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
import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import asdict
from PIL import Image

# 将当前运行目录添加到系统路径，确保能找到 groundingdino 文件夹
sys.path.append(os.getcwd())
# 硬编码你的路径
sys.path.append("/data0/swz/GroundingDINO")

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
    dino_config_path = "/data0/swz/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    dino_weights_path = "/data0/swz/GroundingDINO/weights/groundingdino_swint_ogc.pth"
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
    def analyze_token_for_refinement(cls, doc, char_offset: int) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
        """
        策略 V5 (最终修复版): 
        1. 对齐修正: 自动跳过 LLaMA Token 前面的空格，对齐到 Spacy Token 的单词首字母。
        2. 锚点寻找 & 并集覆盖 (保持 V4 逻辑)。
        """
        
        # 边界检查
        if char_offset >= len(doc.text):
            return None, None
            
        # 只要当前字符是空格，就往后挪
        while char_offset < len(doc.text) and doc.text[char_offset].isspace():
            char_offset += 1

        # === 接下来是正常的 Token 查找 ===
        target_token = None
        for token in doc:
            # 宽松匹配：只要 offset 落在 token 范围内即可
            if token.idx <= char_offset < (token.idx + len(token.text)):
                target_token = token
                break
        
        # 如果修正后还是找不到 (比如选中的是纯空格)，则退出
        if not target_token:
            # print(f"    [Debug] No Spacy token found at aligned offset {char_offset}")
            return None, None

        # 辅助函数: 判断 token 是否在某个 chunk 里
        def get_chunk_containing(t):
            for chunk in doc.noun_chunks:
                if chunk.start <= t.i < chunk.end:
                    return chunk
            return None

        # --- 第一阶段: 寻找锚点 (Anchor) ---
        anchor_chunk = None
        direct_chunk = get_chunk_containing(target_token)
        if direct_chunk:
            anchor_chunk = direct_chunk
        else:
            anchor_token = None
            if target_token.pos_ == "VERB":
                for child in target_token.children:
                    if child.dep_ in {"dobj", "attr", "nsubjpass", "nsubj", "pobj"} and child.pos_ in {"NOUN", "PROPN"}:
                        anchor_token = child
                        break
            elif target_token.pos_ == "ADP":
                for child in target_token.children:
                    if child.dep_ == "pobj" and child.pos_ in {"NOUN", "PROPN"}:
                        anchor_token = child
                        break
            else:
                if target_token.head.pos_ in {"NOUN", "PROPN"}:
                    anchor_token = target_token.head
                elif target_token.head.pos_ == "ADJ" and target_token.head.head.pos_ in {"NOUN", "PROPN"}:
                    anchor_token = target_token.head.head

            if anchor_token:
                anchor_chunk = get_chunk_containing(anchor_token)
                if not anchor_chunk:
                    anchor_chunk = doc[anchor_token.i : anchor_token.i + 1]

        if not anchor_chunk:
            return None, None

        # --- 第二阶段: 计算并集范围 (Mask Range) ---
        dino_phrase = anchor_chunk.text
        t_start = target_token.idx
        t_end = target_token.idx + len(target_token.text)
        a_start = anchor_chunk.start_char
        a_end = anchor_chunk.end_char
        union_start = min(t_start, a_start)
        union_end = max(t_end, a_end)
        
        return dino_phrase, (union_start, union_end)

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
    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
        self.cfg = config
        self.model = None
        # 设置日志
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(handler)
        else:
            self.logger = logger

    def load(self):
        if not DINO_AVAILABLE: return
        self.logger.info(f">>> Loading Grounding DINO from {self.cfg.dino_weights_path}...")
        self.model = load_model(self.cfg.dino_config_path, self.cfg.dino_weights_path)

    # 增加 image_path 参数
    def run_grounding(self, core_phrase: str, specific_output_dir: str, image_path: str):
        if not self.model:
            self.logger.warning("Grounding DINO model not loaded.")
            return None, None

        self.logger.info(f"\n>>> Running Grounding DINO for prompt: '{core_phrase}'")

        # 1. 加载图像 (使用传入的 image_path，而不是 self.cfg.image_path)
        try:
            image_source, image = load_image(image_path)
        except Exception as e:
            self.logger.error(f"Error loading image: {e}")
            return None, None

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
            self.logger.info(f"No objects found for '{core_phrase}'.")
            return None, None

        # 选最佳框
        best_idx = torch.argmax(logits).item()
        best_box = boxes[best_idx]  # (cx, cy, w, h) 归一化坐标
        best_logit = logits[best_idx].item()
        
        if best_logit < 0.50:
            return None, best_logit

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
                self.logger.warning(f"Color conversion failed: {e}")

            # 清洗文件名
            safe_phrase = core_phrase.replace(" ", "_")
            # 加上时间戳防止覆盖
            import time
            timestamp = int(time.time())
            filename = f"best_crop_{safe_phrase}_{timestamp}.jpg"
            save_path = os.path.join(specific_output_dir, filename)

            cv2.imwrite(save_path, crop_img)
            self.logger.info(f"Best Crop Saved: {save_path} (Conf: {best_logit:.2f})")
            return save_path, best_logit
        else:
            self.logger.warning("Crop failed: Invalid coordinates.")
            return None, 0.0

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