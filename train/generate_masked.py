from transformers.generation import stopping_criteria
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from llava.cache import dLLMCache, dLLMCacheConfig
from llava.hooks import register_cache_LLaDA_V
from dataclasses import asdict
from llava.hooks.fast_dllm_hook import register_fast_dllm_hook, unregister_fast_dllm_hook

from PIL import Image
import requests
import copy
import torch
import time
import json
import numpy as np
from pathlib import Path
import sys
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test" / "scripts"))
from visualize_confidence_trends import (
    process_single_sample,
)

import warnings

prompt_interval_steps = 25
gen_interval_steps = 7
transfer_ratio = 0.25
use_fast_dllm = True  # using fast-dLLM (https://github.com/NVlabs/Fast-dLLM) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 6s to generate 128 tokens.
use_dllm_cache = False  # using dLLM-Cache(https://github.com/maomaocun/dLLM-cache) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 25s to generate 128 tokens.

warnings.filterwarnings("ignore")
# pretrained = "GSAI-ML/LLaDA-V"
pretrained = os.environ.get("PRETRAINED_MODEL", str(REPO_ROOT / "train" / "exp" / "llada_v_lora_rank64_1227"))
model_base = "GSAI-ML/LLaDA-V"
model_name = "llava_llada_lora"

device = "cuda:0"
device_map = "cuda:0"

# 设置 vision_tower 路径（覆盖配置文件中的路径）
# 如果本地路径不存在，可以尝试使用 HuggingFace Hub 上的模型
# 选项1: 使用 HuggingFace Hub 模型
vision_tower_path = "google/siglip2-so400m-patch14-384"
# 选项2: 使用本地路径（如果存在）
# vision_tower_path = "/data2/models/siglip-so400m-patch14-384"
# 选项3: 使用相对路径（根据训练脚本）
# vision_tower_path = "model/siglip2-so400m-patch14-384"

# 设置默认CUDA设备
torch.cuda.set_device(0)

# 加载模型：先不使用 device_map，加载后再手动移动到设备
# 这样可以避免 device_map 可能导致的设备不一致问题
# 使用 overwrite_config 覆盖 vision_tower 路径
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, model_base, model_name, 
    attn_implementation="sdpa", 
    device_map=None,  # 先不使用 device_map
    overwrite_config={"mm_vision_tower": vision_tower_path}  # 覆盖 vision_tower 路径
)

# 手动将模型移动到指定设备
model = model.to(device)
model.eval()

# 验证模型设备
print(f"模型设备: {next(model.parameters()).device}")

# 支持多图输入：可以是单个图片路径或图片路径列表
# 示例：
# image_paths = "test0.png"  # 单图
# image_paths = ["test0.png", "test.jpg", "bird.png"]  # 多图
image_paths = ["0000.jpg", "image82.jpg"]  # 可以修改为多图列表

# 加载图片
if isinstance(image_paths, str):
    # 单图输入
    images = [Image.open(image_paths)]
    image_paths = [image_paths]
else:
    # 多图输入
    images = [Image.open(path) for path in image_paths]

print(f"加载了 {len(images)} 张图片: {image_paths}")

# 处理图片
image_tensor = process_images(images, image_processor, model.config)
# 确保所有图像张量都在正确的设备上
if isinstance(image_tensor, list):
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
else:
    image_tensor = image_tensor.to(dtype=torch.float16, device=device)

conv_template = "llava_llada" 
# text = "In the image, a curious cat is standing in a bathroom. The cat is positioned near a white toilet, which <|mdm_mask|> <|mdm_mask|> <|mdm_mask|> <|mdm_mask|> <|mdm_mask|> <|mdm_mask|> <|mdm_mask|> <|mdm_mask|> <|mdm_mask|> cover. The toilet seat is open, revealing a white bowl. To the left of the toilet, there's a black trash can. The wall of the bathroom is decorated with pink flamingo decals, adding a playful touch to the room. The floor of the bathroom also features a black and white checkered pattern, matching the overall aesthetic of the bathroom. The cat appears to be peering over the lid of the toilet, adding a touch of whimsy to the scene."
text = "In the image, a curious cat is seen in a bathroom, peering into a toilet with its lid open. The bathroom features a black and white checkered floor, which extends to a brown mat on the right side of the frame. Above the toilet hangs a shower curtain adorned with pink flamingo decorations. The cat, <|mdm_mask|> <|mdm_mask|> <|mdm_mask|> <|mdm_mask|> <|mdm_mask|> <|mdm_mask|> <|mdm_mask|>, appears to be intrigued by the toilet's interior, with its head visible inside the bowl. To the left, a stack of books is placed on the back of the toilet. Additionally, a small trash can is visible on the floor to the left of the toilet."
num_images = len(images)
if num_images == 1:
    image_tokens = DEFAULT_IMAGE_TOKEN
else:
    image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * num_images)

question = image_tokens + "\nYou are presented with a zoomed-in visual detail and a text description with missing parts. Task: Analyze the specific visual attributes (such as texture, pattern, color, and object shape) in the provided image crop. Instruction: The masked part of the text describes this specific visual evidence. Do not guess based on common language patterns. Instead, look closely at the image crop to accurately restore the missing text. What you see in the crop is the ground truth."
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

# 单独处理text，去掉每组的220
text_token_ids = tokenizer.encode(text, add_special_tokens=False)
# 去掉每组的220（220可能是某种分隔符token）
# 查找所有220和126336的组合，去掉220
cleaned_token_ids = []
i = 0
while i < len(text_token_ids):
    if text_token_ids[i] == 220:
        # 检查后面是否有126336（mask token）
        if i + 1 < len(text_token_ids) and text_token_ids[i + 1] == 126336:
            # 跳过220，保留126336
            cleaned_token_ids.append(text_token_ids[i + 1])
            i += 2
        else:
            # 如果220后面不是126336，也跳过220
            i += 1
    else:
        cleaned_token_ids.append(text_token_ids[i])
        i += 1

text_token_ids = cleaned_token_ids
# print(f"Text token IDs (after removing 220): {text_token_ids[:50]}...")  # 只打印前50个

model.eval()
if use_fast_dllm:
    register_fast_dllm_hook(model)
    # 注册 hook 后再次确保模型在正确的设备上
    model = model.to(device)
    print("Testing with Fast dLLM hook enabled")
elif use_dllm_cache:
    dLLMCache.new_instance(
        **asdict(
            dLLMCacheConfig(
                prompt_interval_steps=prompt_interval_steps,
                gen_interval_steps=gen_interval_steps,
                transfer_ratio=transfer_ratio,
            )
        )
    )
    register_cache_LLaDA_V(model, "model.layers")
    print("Testing with cache enabled")
else:
    print("Testing without cache")

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
# print(input_ids)

# 获取所有图片的尺寸
image_sizes = [img.size for img in images]
print(f"图片尺寸: {image_sizes}")

# 验证输入张量设备
print(f"input_ids 设备: {input_ids.device}")
if isinstance(image_tensor, list):
    print(f"image_tensor 设备: {[img.device for img in image_tensor]}")
else:
    print(f"image_tensor 设备: {image_tensor.device}")

# 判断是初次生成还是迭代修改
is_initial_generation = False  # 设置为False表示迭代修改状态
# 将text_token_ids转换为tensor
text_token_ids_tensor = torch.tensor([text_token_ids], dtype=torch.long, device=device) if text_token_ids else None

start_time = time.time()
result = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    steps=128, gen_length=128, block_length=128, tokenizer=tokenizer, stopping_criteria=['<|eot_id|>'], 
    prefix_refresh_interval=32,
    threshold=None,
    return_confidences=True,
    save_confidence_interval=1,
    is_initial_generation=is_initial_generation,  # 新增参数
    text_token_ids=text_token_ids_tensor,  # 新增参数
)
end_time = time.time()
generation_time = end_time - start_time
print(f"Generation time: {generation_time:.4f} seconds")

# 处理返回结果：可能是 (tokens, confidences) 或 (tokens, confidences, intermediate_confidences)
if len(result) == 2:
    cont, confidences = result
    intermediate_confidences = None
elif len(result) == 3:
    cont, confidences, intermediate_confidences = result
else:
    # cont, confidences = result[0], result[1]
    # intermediate_confidences = result[2] if len(result) > 2 else None
    cont = result

# 解码输出
generated_text = tokenizer.decode(cont[0], skip_special_tokens=True)
print(f"\n生成的文本:\n{generated_text}")

# 获取token列表和置信度列表
token_ids = cont[0].cpu().numpy().tolist()
confidence_list = confidences[0].cpu().numpy().tolist()

# 获取每个token的文本和置信度
token_details = []
for token_id, conf in zip(token_ids, confidence_list):
    token_str = tokenizer.decode([token_id])
    token_details.append({
        'token_id': int(token_id),
        'token': token_str,
        'confidence': float(conf)
    })

# 构建结果字典
result_dict = {
    'generated_text': generated_text,
    'token_details': token_details,
    'num_tokens': len(token_ids),
    'average_confidence': float(np.mean(confidence_list)),
    'min_confidence': float(np.min(confidence_list)),
    'max_confidence': float(np.max(confidence_list)),
    'generation_time': float(generation_time),
    'num_images': num_images,  # 图片数量
    'image_paths': image_paths,  # 图片路径列表
    'image_sizes': [(size[0], size[1]) for size in image_sizes],  # 图片尺寸列表
}

# 如果有中间置信度历史，添加到结果中
if intermediate_confidences is not None:
    intermediate_confidence_history = []
    for step_idx, step_confidences in enumerate(intermediate_confidences):
        # step_confidences 是一个字典，包含 'step', 'block', 'tokens', 'confidences', 'token_states'
        tokens = step_confidences.get('tokens', [])
        token_states = step_confidences.get('token_states', [])
        
        # 为每个token生成文本（mask的token显示为特殊标记）
        token_texts = []
        for token, state in zip(tokens, token_states):
            if state == 'mask':
                token_texts.append('<MASK>')
            else:
                token_texts.append(tokenizer.decode([int(token)]))
        
        step_data = {
            'step': int(step_confidences.get('step', step_idx)),
            'block': int(step_confidences.get('block', 0)),
            'tokens': [int(t) for t in tokens],  # 所有token，包括mask
            'token_texts': token_texts,  # 所有token的文本，mask显示为'<MASK>'
            'confidences': [float(c) for c in step_confidences.get('confidences', [])],  # 所有置信度
            'token_states': token_states  # 每个token的状态：'mask' 或 'decoded'
        }
        intermediate_confidence_history.append(step_data)
    result_dict['intermediate_confidence_history'] = intermediate_confidence_history
    print(f"\n保存了 {len(intermediate_confidence_history)} 个中间步骤的置信度历史")

# 保存结果到JSON文件
output_file = Path("generate_masked_output.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(result_dict, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存到: {output_file}")

# 打印摘要信息
print(f"\n生成摘要:")
print(f"  输入图片数: {num_images}")
print(f"  图片路径: {image_paths}")
print(f"  总token数: {len(token_ids)}")
print(f"  平均置信度: {np.mean(confidence_list):.4f}")
print(f"  最小置信度: {np.min(confidence_list):.4f}")
print(f"  最大置信度: {np.max(confidence_list):.4f}")
print(f"  生成时间: {generation_time:.4f} 秒")

# # 生成可视化图表
# print("\n" + "="*60)
# print("开始生成可视化图表...")
# print("="*60)

# # 设置输出目录（与JSON文件在同一目录）
# output_file_path = Path("generate_masked_output.json")
# output_dir = output_file_path.parent.absolute()
# visualization_output_dir = output_dir / "confidence_visualization"
# visualization_output_dir.mkdir(parents=True, exist_ok=True)
# print(f"可视化输出目录: {visualization_output_dir}")

# # 检查是否有中间置信度历史数据
# if 'intermediate_confidence_history' in result_dict and result_dict['intermediate_confidence_history']:
#     try:
#         # 为result_dict添加index字段（如果还没有）
#         if 'index' not in result_dict:
#             result_dict['index'] = 0
        
#         # 调用process_single_sample生成individual_token_trends.png
#         success = process_single_sample(result_dict, str(visualization_output_dir))
        
#         if success:
#             print(f"\n✓ 可视化图表已保存到: {visualization_output_dir}")
#         else:
#             print(f"✗ 生成可视化图表失败")
#     except Exception as e:
#         print(f"✗ 生成可视化图表时出错: {e}")
#         import traceback
#         traceback.print_exc()
# else:
#     print("警告: 没有中间置信度历史数据，无法生成可视化图表")
#     print("提示: 确保 save_confidence_interval 参数已设置")
