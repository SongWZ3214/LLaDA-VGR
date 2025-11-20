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
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple
import numpy as np
import base64
import io
import re
import codecs

import sys
import warnings

prompt_interval_steps = 25
gen_interval_steps = 7
transfer_ratio = 0.25
use_fast_dllm = True  # using fast-dLLM (https://github.com/NVlabs/Fast-dLLM) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 6s to generate 128 tokens.
use_dllm_cache = False  # using dLLM-Cache(https://github.com/maomaocun/dLLM-cache) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 25s to generate 128 tokens.

warnings.filterwarnings("ignore")

# 配置参数
PRETRAINED = "GSAI-ML/LLaDA-V"
MODEL_NAME = "llava_llada"
DEVICE = "cuda:0"
DEVICE_MAP = "cuda:0"

# 数据集路径和输出路径
DATASET_PATH = "/data2/swz/LLaDA-VGR/test/data/DetailCaps-4870/DetailCaps-4870_refined_EN.parquet"
OUTPUT_DIR = "/data2/swz/LLaDA-VGR/test/result/detailcaps_outputs"  # 输出目录，每个样本会保存为单独的文件
LIMIT = None  # 控制处理前n个数据，设置为None处理全部数据，或设置为数字限制处理数量（如10）

# 中间过程保存参数
SAVE_CONFIDENCE_INTERVAL = 1  # 控制每隔k步保存一次各token的置信度，设置为None不保存中间过程，设置为数字表示每隔k步保存一次（如5表示每5步保存一次）

# 生成参数
GEN_STEPS = 128
GEN_LENGTH = 128
BLOCK_LENGTH = 128
PREFIX_REFRESH_INTERVAL = 32
THRESHOLD = 1

# Prompt设置
PROMPT_TEXT = "Please describe the image in detail."

BASE64_RE = re.compile(r'^[A-Za-z0-9+/]+={0,2}$')

def _pil_from_bytes(b: bytes):
    try:
        im = Image.open(io.BytesIO(b))
        im.load()  # 强制读完，避免懒加载带来的句柄问题
        return im.convert("RGB")
    except Exception:
        return None

def load_image_from_data(data):
    """更稳的图像加载：支持 bytes / bytearray / memoryview / np.uint8数组 / Base64 / data URL / URL / 本地路径"""
    # 1) 先把各种“字节类”统一成 bytes
    if isinstance(data, (bytes, bytearray, memoryview)):
        b = bytes(data)
        img = _pil_from_bytes(b)
        if img is not None:
            return img
        # 如果打不开，再继续往下尝试（极少数情况是字节里其实是 ASCII 文本，如 data URL）
        try:
            data = b.decode("utf-8", errors="ignore")
        except Exception:
            pass  # 留给后续分支处理

    # 2) numpy 数组（uint8 原始图像字节 或 HWC 图片）
    try:
        import numpy as np  # 局部导入以免外部没有
        if isinstance(data, np.ndarray):
            if data.dtype == np.uint8:
                if data.ndim == 1:  # 原始字节
                    img = _pil_from_bytes(data.tobytes())
                    if img is not None:
                        return img
                elif data.ndim in (2, 3):  # 直接当像素矩阵
                    return Image.fromarray(data).convert("RGB")
    except Exception:
        pass

    # 3) 字符串类
    if isinstance(data, str):
        s = data.strip()

        # 3.1 data URL
        # data:image/jpeg;base64,/9j/4AAQ...
        if s.lower().startswith("data:image/"):
            try:
                comma = s.find(',')
                if comma != -1:
                    b64 = s[comma+1:].strip()
                    b = base64.b64decode(b64)
                    img = _pil_from_bytes(b)
                    if img is not None:
                        return img
            except Exception:
                pass

        # 3.2 URL
        if s.startswith("http://") or s.startswith("https://"):
            try:
                r = requests.get(s, timeout=15)
                r.raise_for_status()
                img = _pil_from_bytes(r.content)
                if img is not None:
                    return img
            except Exception as e:
                print(f"无法从URL加载图像: {e}")
                return None

        # 3.3 可能是“Python 字节串的转义文本”，形如 "\\xff\\xd8\\xff..."
        # 这个场景常见于把 bytes 列存成了字符串。
        # 只在明显出现大量 \xHH 且不符合 base64 的时候尝试还原。
        looks_hex_escaped = bool(re.search(r'\\x[0-9A-Fa-f]{2}', s))
        is_base64_like = (
            len(s) > 32 and
            len(s.replace("\n", "").replace("\r", "").replace(" ", "")) % 4 == 0 and
            bool(BASE64_RE.match(s.replace("\n", "").replace("\r", "").replace(" ", "")))
        )

        if looks_hex_escaped and not is_base64_like:
            try:
                # 去掉可能的开头 b'...' 外壳
                if s.startswith(("b'", 'b"')) and s.endswith(("'", '"')):
                    s_inner = s[2:-1]
                else:
                    s_inner = s
                # 把 \xHH 转回原字节
                b = codecs.decode(s_inner, 'unicode_escape').encode('latin-1')
                img = _pil_from_bytes(b)
                if img is not None:
                    return img
            except Exception:
                pass

        # 3.4 规范的 Base64（无 data URL 前缀）
        # 额外加常见魔数的快速路径：/9j/ (JPEG), iVBOR (PNG), R0lG (GIF), UklG (WEBP)
        compact = s.replace("\n", "").replace("\r", "").replace(" ", "")
        if (
            compact.startswith(("/9j/", "iVBOR", "R0lG", "UklG")) or
            (len(compact) > 32 and len(compact) % 4 == 0 and BASE64_RE.match(compact))
        ):
            try:
                b = base64.b64decode(compact)  # 不用 validate=True，容错更好
                img = _pil_from_bytes(b)
                if img is not None:
                    return img
            except Exception:
                pass

        # 3.5 本地路径
        if len(s) < 512 and ("/" in s or "\\" in s):
            p = Path(s)
            if p.exists() and p.is_file():
                try:
                    with open(p, "rb") as f:
                        b = f.read()
                    img = _pil_from_bytes(b)
                    if img is not None:
                        return img
                except Exception:
                    pass

        print(f"无法处理图像数据（字符串），长度={len(s)}, 预览={s[:100]}")
        return None

    # 其他类型兜底
    print(f"无法处理的图像数据类型: {type(data)}")
    return None

def process_single_sample(model, tokenizer, image_processor, image_data, prompt_text, device, gen_steps, gen_length, block_length, prefix_refresh_interval, threshold, save_confidence_interval=None):
    """处理单个样本"""
    # 加载图像（支持Base64、URL或本地路径）
    image = load_image_from_data(image_data)
    if image is None:
        return None
    
    # 处理图像
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
    image_sizes = [image.size]
    
    # 构建prompt
    conv_template = "llava_llada"
    question = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    # 编码输入
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    
    # 生成
    try:
        generate_kwargs = {
            'inputs': input_ids,
            'images': image_tensor,
            'image_sizes': image_sizes,
            'steps': gen_steps,
            'gen_length': gen_length,
            'block_length': block_length,
            'tokenizer': tokenizer,
            'stopping_criteria': ['<|eot_id|>'],
            'prefix_refresh_interval': prefix_refresh_interval,
            'threshold': threshold,
            'return_confidences': True
        }
        
        # 如果设置了保存置信度间隔，添加参数
        if save_confidence_interval is not None:
            generate_kwargs['save_confidence_interval'] = save_confidence_interval
        
        result = model.generate(**generate_kwargs)
        
        # 处理返回结果：可能是 (tokens, confidences) 或 (tokens, confidences, intermediate_confidences)
        if len(result) == 2:
            cont, confidences = result
            intermediate_confidences = None
        elif len(result) == 3:
            cont, confidences, intermediate_confidences = result
        else:
            cont, confidences = result[0], result[1]
            intermediate_confidences = result[2] if len(result) > 2 else None
        
        # 解码输出
        generated_text = tokenizer.decode(cont[0], skip_special_tokens=True)
        
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
        
        result_dict = {
            'generated_text': generated_text,
            # 'token_ids': token_ids,
            # 'confidences': confidence_list,
            'token_details': token_details,
            'num_tokens': len(token_ids),
            'average_confidence': float(np.mean(confidence_list)),
            'min_confidence': float(np.min(confidence_list)),
            'max_confidence': float(np.max(confidence_list)),
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
        
        return result_dict
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def find_failed_samples(output_dir: Path) -> List[Tuple[int, dict, Path]]:
    """
    检测输出目录下失败的样本（没有 generated_text 字段或包含 error 字段）
    
    Returns:
        List of (index, data_dict, filepath) tuples for failed samples
    """
    failed_samples = []
    json_files = sorted(output_dir.glob("*.json"))
    
    print(f"正在检测失败的样本...")
    print(f"找到 {len(json_files)} 个 JSON 文件")
    
    for filepath in json_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查是否有 generated_text 字段
            has_generated_text = 'generated_text' in data and data.get('generated_text', '').strip()
            
            # 如果缺少 generated_text ，则认为是失败的样本
            if not has_generated_text:
                index = data.get('index')
                if index is None:
                    # 尝试从文件名提取索引
                    try:
                        index = int(filepath.stem)
                    except ValueError:
                        print(f"警告: 文件 {filepath} 没有有效的索引，跳过")
                        continue
                
                failed_samples.append((index, data, filepath))
                reason = []
                if not has_generated_text:
                    reason.append("缺少 generated_text")
                print(f"  发现失败样本: {filepath.name} (index: {index}) - {', '.join(reason)}")
        except Exception as e:
            print(f"警告: 读取文件 {filepath} 时出错: {e}")
            continue
    
    print(f"共找到 {len(failed_samples)} 个失败的样本")
    return failed_samples

def reprocess_failed_samples(
    model, tokenizer, image_processor, 
    df: pd.DataFrame, 
    failed_samples: List[Tuple[int, dict, Path]],
    output_dir: Path
):
    """
    重新处理失败的样本
    
    Args:
        model: 模型
        tokenizer: tokenizer
        image_processor: 图像处理器
        df: 原始数据集
        failed_samples: 失败的样本列表 [(index, data_dict, filepath), ...]
        output_dir: 输出目录
    """
    print(f"\n开始重新处理 {len(failed_samples)} 个失败的样本...")
    
    successful_count = 0
    failed_count = 0
    
    for idx, (sample_index, old_data, filepath) in enumerate(tqdm(failed_samples, desc="重新处理失败样本")):
        try:
            # 从原始数据集中获取对应的行
            if sample_index not in df.index:
                print(f"  警告: 样本 {sample_index} 不在数据集中，跳过")
                failed_count += 1
                continue
            
            row = df.loc[sample_index]
            
            # 获取图像数据
            image_data = row.get('binary')
            if pd.isna(image_data) or image_data == '':
                print(f"  警告: 样本 {sample_index} 没有有效的图像数据")
                failed_count += 1
                continue
            
            # 检查数据类型并转换
            if isinstance(image_data, bytes):
                pass
            elif isinstance(image_data, str):
                pass
            else:
                try:
                    if hasattr(image_data, 'tobytes'):
                        image_data = image_data.tobytes()
                    else:
                        image_data = str(image_data)
                except Exception as e:
                    print(f"  警告: 样本 {sample_index} 的图像数据类型无法处理: {type(image_data)}, 错误: {e}")
                    failed_count += 1
                    continue
            
            # 处理样本
            result = process_single_sample(
                model, tokenizer, image_processor,
                image_data, PROMPT_TEXT, DEVICE,
                GEN_STEPS, GEN_LENGTH, BLOCK_LENGTH,
                PREFIX_REFRESH_INTERVAL, THRESHOLD,
                save_confidence_interval=SAVE_CONFIDENCE_INTERVAL
            )
            
            if result is None or 'error' in result:
                print(f"  警告: 样本 {sample_index} 重新处理失败")
                failed_count += 1
                continue
            
            # 构建结果，移除 error 字段，保留其他原有字段（如果有）
            data_result = {
                'index': int(sample_index),
                'binary_data_length': len(str(image_data)) if image_data else 0,
                'binary_data_preview': str(image_data)[:50] + '...' if image_data and len(str(image_data)) > 50 else str(image_data),
                **result,  # 包含新的 generated_text, token_details, confidences等
                # 保留原有数据中的其他字段（除了 error 和生成相关的字段）
                **{col: old_data[col] for col in old_data.keys() 
                   if col not in ['error', 'generated_text', 'token_details', 'num_tokens', 
                                 'average_confidence', 'min_confidence', 'max_confidence',
                                 'intermediate_confidence_history', 'binary_data_length', 'binary_data_preview']}
            }
            
            # 保存结果到原文件（覆盖）
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_result, f, ensure_ascii=False, indent=2)
            
            successful_count += 1
            print(f"  ✓ 成功重新处理样本 {sample_index}")
            
        except Exception as e:
            print(f"  ✗ 重新处理样本 {sample_index} 时出错: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
    
    print(f"\n重新处理完成！")
    print(f"成功: {successful_count} 个")
    print(f"失败: {failed_count} 个")

def main():
    # 加载模型
    print(f"正在加载模型: {PRETRAINED}")
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        PRETRAINED, None, MODEL_NAME,
        attn_implementation="sdpa",
        device_map=DEVICE_MAP
    )
    model.eval()
    print("模型加载完成")
    
    # 注册Fast-dLLM hook（如果需要）
    if use_fast_dllm:
        register_fast_dllm_hook(model)
        print("Fast dLLM hook已启用")
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
        print("dLLM-Cache已启用")
    else:
        print("未使用缓存加速")
    
    # 加载数据集
    print(f"正在加载数据集: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    print(f"数据集包含 {len(df)} 条数据")
    print(f"列名: {list(df.columns)}")
    
    # 检查binary列的数据类型
    if 'binary' in df.columns:
        sample_val = df['binary'].iloc[0] if len(df) > 0 else None
        print(f"binary列数据类型: {df['binary'].dtype}")
        if sample_val is not None:
            print(f"binary列示例数据类型: {type(sample_val)}")
            if isinstance(sample_val, str):
                print(f"binary列示例数据长度: {len(sample_val)}")
                print(f"binary列示例数据前100个字符: {sample_val[:100]}")
            elif isinstance(sample_val, bytes):
                print(f"binary列示例数据长度: {len(sample_val)}")
                print(f"binary列示例数据前100个字节: {sample_val[:100]}")
    
    # 保存完整的原始数据集（用于重新处理失败样本）
    df_full = df.copy()
    
    # 创建输出目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 检测并重新处理失败的样本
    print("\n" + "="*60)
    print("步骤 1: 检测失败的样本")
    print("="*60)
    failed_samples = find_failed_samples(output_dir)
    
    if failed_samples:
        print(f"\n找到 {len(failed_samples)} 个失败的样本，开始重新处理...")
        reprocess_failed_samples(
            model, tokenizer, image_processor,
            df_full, failed_samples, output_dir  # 使用完整数据集
        )
    else:
        print("\n没有发现失败的样本，所有文件都包含有效的 generated_text")
    
    # 如果只想重新处理失败的样本，可以在这里 return
    # 否则继续处理新数据
    if LIMIT is None:
        print("\n" + "="*60)
        print("步骤 2: 处理新数据（LIMIT=None，跳过新数据处理）")
        print("="*60)
        print("提示: 如果只想重新处理失败的样本，可以设置 LIMIT=None")
        return
    
    # 限制处理数量（如果指定）- 只用于处理新数据
    df = df.head(LIMIT)
    print(f"\n限制处理前 {LIMIT} 条数据")
    
    print("\n" + "="*60)
    print("步骤 2: 处理新数据")
    print("="*60)
    
    # 处理每条数据
    successful_count = 0
    failed_count = 0
    print(f"\n开始处理 {len(df)} 条数据...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理数据"):
        try:
            # 获取图像数据（可能是Base64字符串或字节数据）
            image_data = row.get('binary')
            if pd.isna(image_data) or image_data == '':
                print(f"警告: 第 {idx} 条数据没有有效的图像数据")
                error_result = {
                    'index': int(idx),
                    'error': 'No valid image data',
                    **{col: str(row[col]) if not pd.isna(row[col]) else None for col in df.columns}
                }
                # 保存错误结果到单独文件
                output_file = output_dir / f"{idx:06d}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(error_result, f, ensure_ascii=False, indent=2)
                failed_count += 1
                continue
            
            # 检查数据类型并转换
            if isinstance(image_data, bytes):
                # 如果已经是字节数据，直接使用（可能是原始图像数据）
                pass
            elif isinstance(image_data, str):
                pass
            else:
                # 其他类型，尝试转换为字节或字符串
                try:
                    if hasattr(image_data, 'tobytes'):
                        # 如果是numpy数组或其他可转换为字节的对象
                        image_data = image_data.tobytes()
                    else:
                        # 尝试转换为字符串再处理
                        image_data = str(image_data)
                except Exception as e:
                    print(f"警告: 第 {idx} 条数据的图像数据类型无法处理: {type(image_data)}, 错误: {e}")
                    error_result = {
                        'index': int(idx),
                        'error': f'Unsupported data type: {type(image_data)}',
                        **{col: str(row[col]) if not pd.isna(row[col]) else None for col in df.columns}
                    }
                    # 保存错误结果到单独文件
                    output_file = output_dir / f"{idx:06d}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(error_result, f, ensure_ascii=False, indent=2)
                    failed_count += 1
                    continue
            
            # 处理样本
            result = process_single_sample(
                model, tokenizer, image_processor,
                image_data, PROMPT_TEXT, DEVICE,
                GEN_STEPS, GEN_LENGTH, BLOCK_LENGTH,
                PREFIX_REFRESH_INTERVAL, THRESHOLD,
                save_confidence_interval=SAVE_CONFIDENCE_INTERVAL
            )
            
            if result is None:
                error_result = {
                    'index': int(idx),
                    'error': 'Failed to load or process image',
                    **{col: str(row[col]) if not pd.isna(row[col]) else None for col in df.columns if col != 'binary'}
                }
                # 保存错误结果到单独文件
                output_file = output_dir / f"{idx:06d}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(error_result, f, ensure_ascii=False, indent=2)
                failed_count += 1
                continue
            
            # 构建结果（不保存完整的Base64数据，因为太大）
            data_result = {
                'index': int(idx),
                'binary_data_length': len(str(image_data)) if image_data else 0,  # 只保存数据长度
                'binary_data_preview': str(image_data)[:50] + '...' if image_data and len(str(image_data)) > 50 else str(image_data),  # 只保存前50个字符作为预览
                **result,  # 包含generated_text, token_details, confidences等
                # 保存原始数据的所有列（不保存完整的binary，因为Base64数据太大）
                **{col: str(row[col]) if not pd.isna(row[col]) else None 
                   for col in df.columns if col != 'binary'}
            }
            
            # 保存每个样本的结果到单独文件
            output_file = output_dir / f"{idx:06d}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_result, f, ensure_ascii=False, indent=2)
            
            successful_count += 1
            
        except Exception as e:
            print(f"\n处理第 {idx} 条数据时出错: {e}")
            import traceback
            traceback.print_exc()
            error_result = {
                'index': int(idx),
                'error': str(e),
                **{col: str(row[col]) if not pd.isna(row[col]) else None for col in df.columns}
            }
            # 保存错误结果到单独文件
            output_file = output_dir / f"{idx:06d}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, ensure_ascii=False, indent=2)
            failed_count += 1
    
    # 统计信息
    total_count = successful_count + failed_count
    print(f"\n处理完成！")
    print(f"总计: {total_count} 条")
    print(f"成功: {successful_count} 条")
    print(f"失败: {failed_count} 条")
    print(f"结果已保存到目录: {output_dir}")
    print(f"每个样本的结果保存在单独的文件中，文件名格式: {idx:06d}.json")

if __name__ == "__main__":
    main()