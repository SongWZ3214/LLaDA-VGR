import torch
import numpy as np
import json
import os
import sys
import copy
import cv2
import time
import warnings
import spacy
import re
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import asdict
from PIL import Image

# 将当前运行目录添加到系统路径，确保能找到 groundingdino 文件夹
sys.path.append(os.getcwd())
# 硬编码你的路径
sys.path.append("/data2/tyc/GroundingDINO")

# === Grounding DINO Imports ===
try:
    from groundingdino.util.inference import load_model, load_image, predict
    from groundingdino.util.box_ops import box_cxcywh_to_xyxy
    DINO_AVAILABLE = True
except ImportError:
    print("Warning: GroundingDINO not installed. Visual grounding will be skipped.")
    DINO_AVAILABLE = False

# === 全局配置 ===
class Config:
    # Grounding DINO Setup
    dino_config_path = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    dino_weights_path = "weights/groundingdino_swint_ogc.pth"
    box_threshold = 0.35
    text_threshold = 0.25
    SPAN_K = 5

    # 是否直接使用原始 Token Span，不进行 Spacy 核心词提取
    # True: "a small red cat" -> "a small red cat"
    # False: "a small red cat" -> "cat"
    USE_RAW_SPAN = True

    # 裁剪框外扩参数
    # 当检测框面积小于全图面积的 MIN_CROP_RATIO 时，或者为了获取更多语义，强制将框向外扩大 EXPANSION_RATIO (例如 1.5 倍)
    MIN_CROP_RATIO = 0.05  # 如果框小于图的 5%，扩
    EXPANSION_FACTOR = 1.5  # 扩大倍数，1.0 为原大小，1.5 为扩大 50%

# === 核心语义提取器 (Spacy 优化版) ===
class TextMiner:
    _nlp = None
    # 简单的抽象词黑名单，避免提取无意义词汇
    STOP_WORDS = {
        "image", "picture", "photo", "background", "foreground", "view", "scene",
        "detail", "one", "part", "sense", "kind", "sort", "type", "way", "lot",
        "bit", "piece", "pair", "group", "set", "variety", "amount", "number",
        "center", "middle", "left", "right", "top", "bottom", "side", "feature",
        "object", "thing", "item", "element"
    }

    @classmethod
    def get_nlp(cls):
        """单例模式加载 Spacy，避免重复加载"""
        if cls._nlp is None:
            print(">>> Loading Spacy Model (en_core_web_sm)...")
            try:
                cls._nlp = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download
                print(">>> Downloading en_core_web_sm...")
                download("en_core_web_sm")
                cls._nlp = spacy.load("en_core_web_sm")
        return cls._nlp

    @classmethod
    def clean_text(cls, text: str) -> str:
        """清洗文本：去除 Tokenizer 产生的类似 'elephant1' 这种数字后缀"""
        # 移除单词末尾的数字 (e.g., "elephant1" -> "elephant")
        text = re.sub(r'(?<=[a-zA-Z])\d+', '', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @classmethod
    def process_span(cls, text_span: str, use_raw: bool = False) -> str:
        """
        统一入口：根据配置决定是返回 Raw Span 还是提取 Core Phrase
        """
        clean = cls.clean_text(text_span)
        if use_raw:
            return clean
        return cls.extract_core_phrase(clean)

    @classmethod
    def extract_core_phrase(cls, text_span: str) -> str:
        """
        智能核心词提取：利用 Noun Chunks (名词块) 保持形容词+名词的完整性
        """
        # 1. 预清洗文本
        clean_span = cls.clean_text(text_span)
        nlp = cls.get_nlp()
        doc = nlp(clean_span)

        candidates = []

        # 2. 使用 noun_chunks (Spacy 会自动提取 "the black cat" 作为一个块)
        # 如果 span 很短没有 chunk，退化为 token 分析
        chunks = list(doc.noun_chunks)

        if not chunks:
            # 兜底：如果没有名词块，找最长的名词或实词
            for token in doc:
                if token.pos_ in ["NOUN", "PROPN"] and token.text.lower() not in cls.STOP_WORDS:
                    return token.text
            return max([t.text for t in doc if not t.is_stop], key=len) if len(doc) > 0 else "object"

        # 3. 筛选最佳的名词块
        for chunk in chunks:
            # 去除冠词/限定词 (e.g., "the black cat" -> "black cat")
            root_text = chunk.root.text.lower()  # 核心名词
            full_text = chunk.text

            # 这里的逻辑是：去掉开头的冠词
            words = full_text.split()
            if words[0].lower() in ["a", "an", "the", "this", "that"]:
                refined_phrase = " ".join(words[1:])
            else:
                refined_phrase = full_text

            # 如果剩下来的词是空的，或者是纯抽象词，跳过
            if not refined_phrase or refined_phrase.lower() in cls.STOP_WORDS:
                continue

            # 如果核心名词是抽象词 (比如 "piece of cake" 中的 piece)，尝试降低优先级或跳过
            # 这里简单处理：如果核心词在黑名单，就跳过这个chunk
            if root_text in cls.STOP_WORDS:
                continue

            # 评分：长度越长包含信息越多 (优先 "black cat" 而不是 "cat")
            # 且包含名词
            score = len(refined_phrase)
            candidates.append((refined_phrase, score))

        if candidates:
            # 按长度降序排列，取最长的有效名词短语
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        # 再次兜底
        return "object"


# === 波动分析器 (Jitter Analyzer) ===
class JitterAnalyzer:
    @staticmethod
    def calculate_jitter_numpy(confidence_matrix: np.ndarray) -> np.ndarray:
        """向量化计算 Jitter，增加维度检查"""
        # 1. 维度检查
        if confidence_matrix.ndim < 2:
            return np.zeros(confidence_matrix.shape[0]) if confidence_matrix.size > 0 else np.array([])

        # 2. 长度检查：如果 Step 数少于 2，无法计算差分
        if confidence_matrix.shape[1] < 2:
            return np.zeros(confidence_matrix.shape[0])

        # Jitter = sum(|diff|) - (end - start)
        diffs = np.abs(np.diff(confidence_matrix, axis=1))
        sum_changes = np.sum(diffs, axis=1)
        # 注意：这里假设最后一步是收敛的，第一步是发散的
        conf_diff = confidence_matrix[:, -1] - confidence_matrix[:, 0]

        return sum_changes - conf_diff

    @staticmethod
    def extract_confidence_data(intermediate_history: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """
        从历史记录中提取置信度矩阵。
        修复点：不再依赖 token_texts 来确定 token 数量，而是直接使用 confidences 的长度。
        """
        if not intermediate_history:
            return np.array([]), []

        # 获取最后一步的数据
        last_step = intermediate_history[-1]

        # 优先使用 confidences 的长度来确定 num_tokens
        # 因为 loop_initial_version_with_poor_performance.py 可能只传了 confidences 而没有传 token_texts
        confidences_raw = last_step.get('confidences', [])
        num_tokens = len(confidences_raw)

        # 只有在真的没有 confidences 时，才尝试回退查看 token_texts (虽然此时也没用了)
        token_texts = last_step.get('token_texts', [])
        if num_tokens == 0 and token_texts:
            num_tokens = len(token_texts)

        if num_tokens == 0:
            return np.array([]), []

        # 初始化矩阵 (Token数, Step数)
        num_steps = len(intermediate_history)
        confidence_matrix = np.zeros((num_tokens, num_steps))

        for step_idx, step_data in enumerate(intermediate_history):
            confs = step_data.get('confidences', [])
            # 截断或填充逻辑：确保长度一致，防止赋值报错
            length = min(len(confs), num_tokens)
            if length > 0:
                confidence_matrix[:length, step_idx] = confs[:length]

        return confidence_matrix, token_texts


# ====== 视觉定位模块 (GroundingAgent) =======
class GroundingAgent:
    def __init__(self, config: Config):
        self.cfg = config
        self.model = None

    def load(self):
        if not DINO_AVAILABLE: return
        print(f">>> Loading Grounding DINO from {self.cfg.dino_weights_path}...")
        self.model = load_model(self.cfg.dino_config_path, self.cfg.dino_weights_path)

    # 增加 image_path 参数
    def run_grounding(self, core_phrase: str, specific_output_dir: str, image_path: str):
        if not self.model:
            print("Grounding DINO model not loaded.")
            return

        print(f"\n>>> Running Grounding DINO for prompt: '{core_phrase}'")

        # 1. 加载图像 (使用传入的 image_path，而不是 self.cfg.image_path)
        try:
            image_source, image = load_image(image_path)
        except Exception as e:
            print(f"Error loading image: {e}")
            return

        h_img, w_img, _ = image_source.shape

        # 2. 推理
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=core_phrase,
            box_threshold=self.cfg.box_threshold,
            text_threshold=self.cfg.text_threshold
        )

        if len(boxes) == 0:
            print(f"No objects found for '{core_phrase}'.")
            return

        # 选最佳框
        best_idx = torch.argmax(logits).item()
        best_box = boxes[best_idx]  # (cx, cy, w, h) 归一化坐标
        best_logit = logits[best_idx].item()

        # --- 动态外扩逻辑 ---
        cx, cy, w_box, h_box = best_box.tolist()

        # 1. 计算当前面积占比
        area_ratio = (w_box * h_box)

        # 2. 决定是否扩大 (如果面积太小，或者强制开启扩大)

        # 如果框极小，扩大倍数可以更大一点，防止 crop 出来全是马赛克
        if area_ratio < self.cfg.MIN_CROP_RATIO:
            expand_factor = self.cfg.EXPANSION_FACTOR  # 极小框扩大
            new_w = w_box * expand_factor
            new_h = h_box * expand_factor
        else:
            new_w = w_box
            new_h = h_box

        # 3. 转换回 xyxy 并截断边界
        # cx, cy 是归一化的，需要转换
        # new_w, new_h 也是归一化的

        x1 = (cx - new_w / 2) * w_img
        y1 = (cy - new_h / 2) * h_img
        x2 = (cx + new_w / 2) * w_img
        y2 = (cy + new_h / 2) * h_img

        # 整数化 & 边界检查
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))

        if x2 > x1 and y2 > y1:
            crop_img = image_source[y1:y2, x1:x2]

            # image_source 是 RGB 格式 (PIL加载)，但 cv2.imwrite 需要 BGR 格式
            # 所以在保存前需要转换颜色空间
            try:
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Color conversion failed: {e}")

            # 清洗文件名
            safe_phrase = core_phrase.replace(" ", "_")
            # 加上时间戳防止覆盖
            import time
            timestamp = int(time.time())
            filename = f"best_crop_{safe_phrase}_{timestamp}.jpg"
            save_path = os.path.join(specific_output_dir, filename)

            cv2.imwrite(save_path, crop_img)
            print(f"Best Crop Saved: {save_path} (Conf: {best_logit:.2f})")
        else:
            print("Crop failed: Invalid coordinates.")


def manual_annotate(image_source, boxes, logits, phrases):
    """
    手动使用 OpenCV 画框和标签，替代 broken 的 supervise 库函数
    """
    h, w, _ = image_source.shape
    # 转换坐标: (cx, cy, w, h) -> (x1, y1, x2, y2)
    boxes_xyxy = box_cxcywh_to_xyxy(boxes) * torch.Tensor([w, h, w, h])

    annotated_frame = image_source.copy()

    for box, conf, label in zip(boxes_xyxy, logits, phrases):
        x1, y1, x2, y2 = box.int().tolist()

        # 1. 画矩形框 (绿色, 线宽2)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 2. 准备标签文本
        text = f"{label} {conf:.2f}"

        # 3. 画文字背景 (为了看清楚字)
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated_frame, (x1, y1 - text_h - 6), (x1 + text_w, y1), (0, 255, 0), -1)

        # 4. 画文字 (白色)
        cv2.putText(annotated_frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated_frame

# === Mask工具函数 ===
def mask_high_jitter_tokens(token_ids_list, span_range, tokenizer, mask_token_id=126336):
    """把生成的ID序列中的波动部分替换为MASK"""
    new_ids = token_ids_list.copy()
    start, end = span_range
    start = max(0, start)
    end = min(len(new_ids), end)

    for i in range(start, end):
        new_ids[i] = mask_token_id

    masked_text = tokenizer.decode(new_ids, skip_special_tokens=False)
    return masked_text