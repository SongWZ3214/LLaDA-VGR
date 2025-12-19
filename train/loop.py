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
device = "cuda:0"
torch.cuda.set_device(0)
warnings.filterwarnings("ignore")

pretrained = "/data0/swz/LLaDA-VGR/train/exp/llada_vgr_lora_rank64"
model_base = "jiyatai/ReDiff"
# pretrained = "jiyatai/ReDiff"
model_name = "llava_llada_lora"
# model_name = "llava_llada"
vision_tower_path = "google/siglip2-so400m-patch14-384"
original_image_path = "/data0/swz/LLaDA-VGR/test/data/images/000001.png"
temp_dir = "./cropped_image"
os.makedirs(temp_dir, exist_ok=True)

# === 超参数配置 ===
MAX_STEPS = 10  # 最大循环数
JITTER_THRESHOLD = 0.30 # 结束循环的 jitter statistic 阈值
# Analysis & Refinement
SPAN_K = 3  # Span 半径：中心词左右各取 K 个 token

MASK_EXPANSION = 2  # Mask 额外扩张：Mask的时候，在span基础上左右多Mask 几个词，给重写空间；加大到 4，破坏上下文惯性，强迫重写
GLOBAL_SUPPRESS_RADIUS = 3  # 全局空间抑制半径：如果位置 i 被选中过，下次 i ± GLOBAL_SUPPRESS_RADIUS 都不能再选

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
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

print(">>> Loading LLaDA Model...")
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, model_base, model_name,
    # pretrained, None, model_name,
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
base_instruction = "Please describe the image in detail. Use less absolute directional descriptions."

# 状态变量
last_intermediate_confidences = None
current_generated_ids = []
# 记录已经处理过的 phrase，防止循环死磕同一个地方
# 全局空间记忆：记录已经处理过的 Token Index 集合
global_covered_indices = set()

initial_jitter = []
target_span_range = None

print(f"\n{'=' * 20} Start Intelligent Loop (Threshold: {JITTER_THRESHOLD}) {'=' * 20}")

# -----------------------------------------------------------------------------
# 4. 进入循环
# -----------------------------------------------------------------------------
def map_char_span_to_token_span(token_ids, tokenizer, char_start, char_end):
    """
    将字符级范围 (char_start, char_end) 映射回 Token 索引范围 (token_start, token_end)。
    通过逐个解码 Token 来累加字符位置，确保 100% 对齐。
    """
    current_char_pos = 0
    token_start = None
    token_end = None

    for i, tid in enumerate(token_ids):
        # 解码单个 token (注意：LLaMA token 通常自带前导空格)
        # skip_special_tokens=False 保证长度计算准确
        token_text = tokenizer.decode([tid], skip_special_tokens=False)
        token_len = len(token_text)
        
        # 当前 token 的字符区间
        t_s = current_char_pos
        t_e = current_char_pos + token_len
        
        # 检查是否与目标区间 [char_start, char_end) 有交集
        # 逻辑：Token 结束位置大于目标开始，且 Token 开始位置小于目标结束
        if t_e > char_start and t_s < char_end:
            if token_start is None:
                token_start = i
            token_end = i + 1 # end 是开区间，所以 +1
            
        current_char_pos += token_len
        
        # 优化：如果已经超出了目标范围，提前退出
        if current_char_pos >= char_end and token_start is not None:
            break
            
    # 兜底：如果没匹配到 (理论不应发生)，返回 None
    if token_start is None:
        return None, None
        
    return token_start, token_end

for step_idx in range(MAX_STEPS + 1):
    # 每轮开始前强力GC
    # flush()
    print(f"\n>>> [Phase {step_idx}] ", end="")

    # =========================================================================
    # A. 分析与决策阶段 (Phase > 0)
    # =========================================================================
    if step_idx > 0:
        print("Analyzing Uncertainty...")

        if last_intermediate_confidences:
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
            
            del conf_matrix, history_data
        if step_idx == 1:
            initial_jitter = jitter_values
        else:
            if len(initial_jitter) > 0 and len(jitter_values) == len(initial_jitter):
                s, e = target_span_range
                initial_jitter[s:e] =  jitter_values[s:e]
                jitter_values = initial_jitter.copy()
            else:
                print("  -> Initial jitter and current jitter have different lengths. Skipping.")
                continue
        # print(f"  -> Jitter Values: {jitter_values}")

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

        # 2. 智能选点逻辑 (POS-Guided Selection)
        # ---------------------------------------------------------------------
        # 不再只取 max，而是取排序后的列表，依次检查是否符合语法要求
        sorted_indices = np.argsort(jitter_values)[::-1] # 降序排列
        
        selected_idx = None
        target_phrase = None
        target_span_range = None # [token_start, token_end]
        
        # 解码完整文本，用于 Spacy 分析
        full_text = tokenizer.decode(current_generated_ids, skip_special_tokens=True)
        # 获取 Spacy Doc (只需解析一次)
        nlp = TextMiner.get_nlp()
        doc = nlp(full_text)
        
        print(f"  -> Searching for valid refinement target (NOUN/ADJ)...")

        for idx in sorted_indices:
            # 阈值截断
            if jitter_values[idx] < JITTER_THRESHOLD:
                break 
            
            # 计算该 Token 在 string 中的字符偏移量 (char_offset)
            # 方法：解码该 token 之前的所有 token，看长度
            prefix_ids = current_generated_ids[:idx]
            prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
            char_offset = len(prefix_text)
            
            # 忽略开头结尾的特殊符号带来的偏移误差，这通常是一个近似值，但对定位单词足够了
            # 如果 token 是 " apple" (带空格)，decode后空格会计入长度
            
            # 调用 crop.py 中的新方法进行判断
            chunk_text, char_span = TextMiner.analyze_token_for_refinement(doc, char_offset)
            
            if chunk_text:
                # 找到了合法的名词/形容词！
                selected_idx = idx
                target_phrase = chunk_text
                max_val = jitter_values[idx]
                
                print(f"  -> Valid Target Found: '{tokenizer.decode([current_generated_ids[idx]])}' (POS Valid). Expanding to chunk: '{target_phrase}'")
                
                # --- 核心：反向映射字符 Span 到 Token Span ---
                # char_span 是 (start_char, end_char)
                start_char, end_char = char_span
                
                s_start, s_end = map_char_span_to_token_span(
                    current_generated_ids, 
                    tokenizer, 
                    start_char, 
                    end_char
                )
                
                if s_start is None:
                    print(f"     [Error] Failed to map char span to tokens. Skipping.")
                    continue
                
                # 保存 Mask 范围
                target_span_range = [s_start, s_end]
                
                # 停止搜索
                break
            else:
                print(f"     [Skip] Token '{tokenizer.decode([current_generated_ids[idx]])}' is not a valid target.")
                pass
        
        # ---------------------------------------------------------------------
        if selected_idx is None:
            print(f"  -> No valid refinement target found (Max Jitter below threshold or no NOUN/ADJ found). Stability reached.")
            break

        print(f"  -> Selected Focus: '{target_phrase}' (Jitter: {max_val:.4f})")


        # 3. Grounding
        # DINO 推理
        move_model_to_gpu(grounder.model, device)
        crop_path, dino_conf = grounder.run_grounding(target_phrase, temp_dir, original_image_path)
        move_model_to_cpu(grounder.model)
        
        # 4. 加载 Crop
        if crop_path:
            crop_img = Image.open(crop_path).convert("RGB")
            if hasattr(image_processor, 'image_mean') and image_processor.image_mean is not None:
                mean = image_processor.image_mean
                bg_color = tuple(int(x * 255) for x in mean)
            else:
                bg_color = (122, 116, 104)
            crop_img = expand2square(crop_img, bg_color)
            active_images = [crop_img]
        else:
            if dino_conf:
                print(f"  -> [REJECT] Visual evidence weak (Conf: {dino_conf:.2f} < 0.50). Skipping.")
            else:
                print("  -> No crop found. SKIPPING refinement for this step to avoid regression.")
            for i in range(target_span_range[0], target_span_range[1]):
                global_covered_indices.add(i)
            continue

    else:
        print("Initial Generation...")
        active_images = [original_image_obj]

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
            s = target_span_range[0]
            e = target_span_range[1]
            for _ in range(MASK_EXPANSION):
                if s > 0:
                    prev_token_id = current_generated_ids[s - 1]
                    prev_token = tokenizer.decode([prev_token_id])
                    # 如果遇到标点，停止向左膨胀
                    if '.' in prev_token or '!' in prev_token or '?' in prev_token or ',' in prev_token:
                        break
                    s -= 1
            for _ in range(MASK_EXPANSION):
                if e < len(current_generated_ids):
                    next_token_id = current_generated_ids[e]
                    next_token = tokenizer.decode([next_token_id])
                    # 如果遇到标点，停止向右膨胀
                    if '.' in next_token or '!' in next_token or '?' in next_token or ',' in next_token:
                        break
                    e += 1
            # 打印调试：看看我们到底 Mask 了什么；先解码原始文本，再进行 Mask 赋值
            original_text_segment = tokenizer.decode(masked_ids[s:e], skip_special_tokens=False)
            print(f"  -> Applied Masking on: '{original_text_segment}' (Indices: {s}-{e})")

            # 赋值 Mask
            for i in range(s, e):
                masked_ids[i] = MASK_TOKEN_ID
                global_covered_indices.add(i)

        text_token_ids_tensor = torch.tensor([masked_ids], device=device)
        kwargs_text_ids = {"text_token_ids": text_token_ids_tensor}

        # refine_instruction = "\nRefine the description based on the zoomed-in details."
        # 增强 Prompt 引导
        # 告诉模型，第二张图是细节，请根据细节修改被 Mask 的部分
        refine_instruction = (
            "\nYou are presented with a zoomed-in visual detail and a text description with missing parts. Task: Analyze the specific visual attributes (such as texture, pattern, color, and object shape) in the provided image crop. Instruction: The masked part of the text describes this specific visual evidence. Do not guess based on common language patterns. Instead, look closely at the image crop to accurately restore the missing text. What you see in the crop is the ground truth."
        )
        prompt_text = img_tokens_str + refine_instruction

    # 3. 对话模版处理
    conv = copy.deepcopy(conv_templates["llava_llada"])
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    prompt_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

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
                repetition_penalty=1.15,
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