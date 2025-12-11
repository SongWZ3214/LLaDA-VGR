import torch
import copy
import time
import os
import sys
import numpy as np
import glob
import warnings
import gc
from PIL import Image

from crop import Config, GroundingAgent, JitterAnalyzer, TextMiner, mask_high_jitter_tokens

# LLaVA / LLaDA Imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.hooks.fast_dllm_hook import register_fast_dllm_hook

# -----------------------------------------------------------------------------
# 0. 基础配置
# -----------------------------------------------------------------------------
device = "cuda:1"
torch.cuda.set_device(1)
warnings.filterwarnings("ignore")

pretrained = "jiyatai/ReDiff"
model_name = "llava_llada"
vision_tower_path = "google/siglip2-so400m-patch14-384"
original_image_path = "/data2/tyc/LLaDA-V_Experiment/test/data/images/000002.png"
temp_dir = "/data2/tyc/LLaDA-V_Experiment/train/cropped_image"
os.makedirs(temp_dir, exist_ok=True)

# === 超参数配置 ===
MAX_STEPS = 4   # 最大循环数
JITTER_THRESHOLD = 0.10 # 结束循环的 jitter statistic 阈值
# Analysis & Refinement
SPAN_K = 2  # Span 半径：中心词左右各取 K 个 token
TOP_K_CANDIDATES = 3 # 每轮分析 Top-3 个波动点

SPAN_NMS_RADIUS = 3 # NMS 半径：如果选了位置 i，那么 i±5 的都不能再选；稍微加大 NMS 半径，避免选中相邻词
MASK_EXPANSION = 8  # Mask 额外扩张：Mask的时候，在span基础上左右多Mask 几个词，给重写空间；加大到 4，破坏上下文惯性，强迫重写
GLOBAL_SUPPRESS_RADIUS = 5  # 全局空间抑制半径：如果位置 i 被选中过，下次 i ± GLOBAL_SUPPRESS_RADIUS 都不能再选

MASK_TOKEN_ID = 126336

# 强制 Resize 尺寸，防止 Crop 过大导致处理时 OOM
# SigLIP 通常是 384，设置为 384x384 可以避免大量插值计算显存消耗
CROP_RESIZE_TARGET = (384, 384)
# -----------------------------------------------------------------------------
# 1. 显存管理工具
# -----------------------------------------------------------------------------
def flush():
    """强制清理显存"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def move_model_to_cpu(model_obj):
    """将模型移动到 CPU 以节省显存"""
    if model_obj:
        model_obj.to("cpu")
        flush()

def move_model_to_gpu(model_obj, device):
    """将模型移动回 GPU"""
    if model_obj:
        model_obj.to(device)

# -----------------------------------------------------------------------------
# 2. 模型加载
# -----------------------------------------------------------------------------
# Token NMS (Non-Maximum Suppression)
def get_top_k_separated_indices(jitter_values, k, separation_radius):
    """
    选取Top-K个波动点，但是强制要求它们之间至少相隔 separation_radius 个Token。
    防止选中同一个词的三个部分。
    """
    # 从大到小排序的索引
    sorted_indices = np.argsort(jitter_values)[::-1]

    selected_indices = []
    for idx in sorted_indices:
        # 如果jitter已经是0（被全局mask了），直接跳过
        if jitter_values[idx] == 0:
            continue

        is_too_close = False
        for chosen in selected_indices:
            if abs(idx - chosen) < separation_radius:
                is_too_close = True
                break

        if not is_too_close:
            selected_indices.append(idx)
            if len(selected_indices) >= k:
                break

    return selected_indices

print(">>> Loading LLaDA Model...")
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name,
    attn_implementation="sdpa",
    device_map=None,
    overwrite_config={"mm_vision_tower": vision_tower_path}
)
model = model.to(device)
model.eval()
register_fast_dllm_hook(model)

# 初始化 GroundingDINO (先放在 CPU，用的时候再拿上来)
print(">>> Loading GroundingDINO (Standing by on CPU)...")
dino_config = Config()
dino_config.SPAN_K = SPAN_K
grounder = GroundingAgent(dino_config)
grounder.load()
# 策略 B: 初始时不占用 GPU
if grounder.model:
    grounder.model.to("cpu")

# -----------------------------------------------------------------------------
# 3. 准备工作
# -----------------------------------------------------------------------------
# 仅加载一次原图对象
original_image_obj = Image.open(original_image_path).convert("RGB")
base_instruction = "Please describe the image in detail."

# 状态变量
last_intermediate_confidences = None
current_generated_ids = []
# 记录已经处理过的 phrase，防止循环死磕同一个地方
# 全局空间记忆：记录已经处理过的 Token Index 集合
global_covered_indices = set()

print(f"\n{'=' * 20} Start Intelligent Loop (Threshold: {JITTER_THRESHOLD}) {'=' * 20}")

# -----------------------------------------------------------------------------
# 4. 进入循环
# -----------------------------------------------------------------------------
for step_idx in range(MAX_STEPS + 1):
    # 每轮开始前强力GC
    flush()
    print(f"\n>>> [Phase {step_idx}] ", end="")

    target_span_range = None

    # === 策略 A: 定义本轮使用的图片列表 ===
    # 始终重置为一个新列表，而不是 append
    active_images = [original_image_obj]

    # =========================================================================
    # A. 分析与决策阶段 (Phase > 0)
    # =========================================================================
    if step_idx > 0:
        print("Analyzing Uncertainty...")

        if not last_intermediate_confidences:
            print("  [Error] No confidence history. Breaking.")
            break

        # 1. Jitter 计算
        history_data = [{'confidences': [float(c) for c in step.get('confidences', [])]}
                        for step in last_intermediate_confidences]

        # 显存大扫除！
        # Jitter 算完后，这个巨大的历史数据就没有用了，必须马上删除！
        # 否则它会和下一轮生成的模型同时占用显存，导致 OOM。
        del last_intermediate_confidences
        last_intermediate_confidences = None
        flush()  # 强制清理

        conf_matrix, _ = JitterAnalyzer.extract_confidence_data(history_data)
        jitter_values = JitterAnalyzer.calculate_jitter_numpy(conf_matrix)

        # 用完 matrix 也销毁
        del conf_matrix, history_data

        if len(jitter_values) == 0: break

        # 应用全局 Mask (Global NMS)
        # 将之前所有处理过的位置周围的 Jitter 强制设为 0
        if global_covered_indices:
            # 将 set 转为 list 进行索引操作
            suppressed_mask = np.zeros_like(jitter_values, dtype=bool)
            for idx in global_covered_indices:
                if idx < len(jitter_values):
                    suppressed_mask[idx] = True

            # 将被抑制的位置 Jitter 设为 0
            jitter_values[suppressed_mask] = 0.0
            print(f"  -> Suppressed {sum(suppressed_mask)} tokens based on history.")

        # 2. 获取 Top-1 (既然我们要迭代修补，每次修补最严重的一个即可，无需 Top-K 循环)
        # 此时 jitter_values 已经去除了历史区域
        if np.max(jitter_values) < JITTER_THRESHOLD:
            print(f"  -> Max Jitter ({np.max(jitter_values):.4f}) below threshold. Stability reached.")
            break

        best_idx = np.argmax(jitter_values)
        max_val = jitter_values[best_idx]

        # 提取文本
        s_start = max(0, best_idx - SPAN_K)
        s_end = min(len(current_generated_ids), best_idx + SPAN_K + 1)
        span_ids = current_generated_ids[s_start:s_end]
        span_str = tokenizer.decode(span_ids, skip_special_tokens=True)

        target_phrase = TextMiner.process_span(span_str, use_raw=Config.USE_RAW_SPAN)

        print(f"  -> Selected Focus: '{target_phrase}' (Jitter: {max_val:.4f}, Index: {best_idx})")

        # 更新全局空间记忆
        # 标记当前中心点左右 RADIUS 范围内为“已处理”
        cover_start = max(0, best_idx - GLOBAL_SUPPRESS_RADIUS)
        cover_end = min(len(jitter_values), best_idx + GLOBAL_SUPPRESS_RADIUS + 1)
        for i in range(cover_start, cover_end):
            global_covered_indices.add(i)

        # 3. Grounding
        # 计算 Mask 范围
        mask_start = max(0, s_start - MASK_EXPANSION)
        mask_end = min(len(current_generated_ids), s_end + MASK_EXPANSION)
        target_span_range = [mask_start, mask_end]

        # DINO 推理
        move_model_to_gpu(grounder.model, device)
        grounder.run_grounding(target_phrase, temp_dir, original_image_path)
        move_model_to_cpu(grounder.model)

        # 4. 加载 Crop
        list_of_crops = glob.glob(os.path.join(temp_dir, f"best_crop_*.jpg"))
        if list_of_crops:
            # 找到最新的一张 Crop
            latest_crop_path = max(list_of_crops, key=os.path.getctime)
            print(f"  -> Found Evidence: {latest_crop_path}")

            # 严格限制 Active Images 为 [原图, 新Crop]
            # 不再 append 到旧列表，保证显存不随 Step 增加
            crop_img = Image.open(latest_crop_path).convert("RGB")

            # Resize Crop!
            # 原始 Crop 可能很大或比例奇怪，导致 process_images 产生巨大开销
            crop_img = crop_img.resize(CROP_RESIZE_TARGET, Image.Resampling.LANCZOS)

            active_images.append(crop_img)
            # 此时 len(active_images) == 2
        else:
            print("  -> No crop found. Text will be masked but no new visual evidence added.")

    else:
        print("Initial Generation...")

    # =========================================================================
    # B. 构建 Prompt (根据 active_images 动态调整)
    # =========================================================================

    # 清理上轮残留
    flush()

    # 1. 图像转 Tensor
    image_tensor = process_images(active_images, image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [_img.to(dtype=torch.float16, device=device) for _img in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=torch.float16, device=device)

    # 2. 构建 Image Tokens 字符串 (根据当前图片数量)
    # 如果 active_images 有2张图，这里就是 "<image> <image>"
    img_tokens_str = " ".join([DEFAULT_IMAGE_TOKEN] * len(active_images))

    kwargs_text_ids = {}

    if step_idx == 0:
        # Phase 0: 初始生成
        prompt_text = img_tokens_str + "\n" + base_instruction
    else:
        # Phase > 0: 修正生成
        # 准备 Masked Token IDs
        masked_ids = current_generated_ids.copy()

        if target_span_range:
            s, e = target_span_range
            # 打印调试：看看我们到底 Mask 了什么；先解码原始文本，再进行 Mask 赋值
            original_text_segment = tokenizer.decode(masked_ids[s:e], skip_special_tokens=False)
            print(f"  -> Applied Masking on: '{original_text_segment}' (Indices: {s}-{e})")

            # 赋值 Mask
            for i in range(s, e):
                masked_ids[i] = MASK_TOKEN_ID

        text_token_ids_tensor = torch.tensor([masked_ids], device=device)
        kwargs_text_ids = {"text_token_ids": text_token_ids_tensor}

        # refine_instruction = "\nRefine the description based on the zoomed-in details."
        # 增强 Prompt 引导
        # 告诉模型，第二张图是细节，请根据细节修改被 Mask 的部分
        refine_instruction = (
            "\nI have provided a zoomed-in crop (the second image) highlighting the masked area. "
            "Please regenerate the masked text to accurately describe the details in the crop."
        )
        prompt_text = img_tokens_str + refine_instruction

    # 3. 对话模版处理
    conv = copy.deepcopy(conv_templates["llava_llada"])
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    prompt_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
        0).to(device)

    # =========================================================================
    # C. 模型推理
    # =========================================================================
    # 进入最危险的环节前，再次清理
    flush()

    try:
        with torch.inference_mode():
            result = model.generate(
                prompt_ids,
                images=image_tensor,
                image_sizes=[img.size for img in active_images],
                steps=128,
                gen_length=128,
                block_length=128,
                tokenizer=tokenizer,
                stopping_criteria=['<|eot_id|>'],
                return_confidences=True,
                save_confidence_interval=1,
                is_initial_generation=(step_idx == 0),
                **kwargs_text_ids
            )
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"!!! CUDA OOM at Step {step_idx}. Exiting.")
            # 【修改点 5】OOM 时的清理
            if 'last_intermediate_confidences' in locals():
                del last_intermediate_confidences
            del image_tensor, prompt_ids
            flush()
            break
        else:
            raise e

    # =========================================================================
    # D. 结果处理
    # =========================================================================
    if isinstance(result, tuple):
        if len(result) == 3:
            cont, _, intermediate_confidences = result
        elif len(result) == 2:
            cont, _ = result
    else:
        cont = result
        # 赋值给 last_intermediate_confidences 用于下一轮

    # 注意：这一步是在 generate 之后，所以此时显存是 (Model + Output + Confidences)
    # 我们在下一轮开头立即 delete 它是最安全的。
    last_intermediate_confidences = intermediate_confidences

    if hasattr(cont, 'cpu'):
        current_generated_ids = cont[0].cpu().numpy().tolist()
    elif isinstance(cont, list):
        current_generated_ids = cont[0] if len(cont) > 0 else []

    if current_generated_ids:
        text_res = tokenizer.decode(current_generated_ids, skip_special_tokens=True)
        # 简单清洗输出，防止刷屏
        print(f"--> Result: {text_res.replace(chr(10), ' ')}")

    # 显式删除 Tensor
    del result, cont, intermediate_confidences, image_tensor, prompt_ids
    if 'text_token_ids_tensor' in locals(): del text_token_ids_tensor
    if 'crop_img' in locals(): del crop_img

    # 再次清理
    flush()

print(f"\n{'=' * 20} Loop Completed {'=' * 20}")