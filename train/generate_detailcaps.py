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
import tempfile
import os

import sys
import warnings
import logging
from datetime import datetime

from refinement_engine import RefinementEngine

prompt_interval_steps = 25
gen_interval_steps = 7
transfer_ratio = 0.25
use_fast_dllm = True  # using fast-dLLM (https://github.com/NVlabs/Fast-dLLM) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 6s to generate 128 tokens.
use_dllm_cache = False  # using dLLM-Cache(https://github.com/maomaocun/dLLM-cache) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 25s to generate 128 tokens.

warnings.filterwarnings("ignore")

# 配置参数
PRETRAINED = "/data0/swz/LLaDA-VGR/train/exp/llada_vgr_lora_rank64"  # 训练好的模型路径
MODEL_BASE = "jiyatai/ReDiff"  # 基础模型路径
MODEL_NAME = "llava_llada_lora"  # 模型名称
DEVICE = "cuda:0"
DEVICE_MAP = "cuda:0"

# RefinementEngine 配置参数
VISION_TOWER_PATH = "google/siglip2-so400m-patch14-384"
MAX_STEPS = 5  # 最大迭代次数
JITTER_THRESHOLD = 0.35  # Jitter 阈值
SPAN_K = 3  # Span 半径
MASK_EXPANSION = 2  # Mask 扩张
GLOBAL_SUPPRESS_RADIUS = 3  # 全局抑制半径
TEMP_DIR = "./cropped_image"  # 临时文件目录

# 数据集路径和输出路径
DATASET_PATH = "/data0/swz/LLaDA-VGR/exp/DetailCaps/DetailCaps-4870_refined_EN.parquet"
OUTPUT_DIR = "/data0/swz/LLaDA-VGR/exp/DetailCaps/result/llada_vgr_lora_rank64"  # 输出目录，每个样本会保存为单独的文件
START_INDEX = 0  # 起始索引（包含），设置为None表示从0开始
END_INDEX = None  # 终止索引（不包含），设置为None表示处理到末尾

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

# 配置日志
class TeeLogger:
    """同时将输出写入文件和原始流，进度条输出到终端"""
    def __init__(self, file_path, original_stream):
        self.file = open(file_path, 'a', encoding='utf-8')
        self.original_stream = original_stream
        
    def write(self, message):
        # 检查是否是进度条输出（tqdm 的特征：包含 \r 或 \b，或者以 \r 开头）
        is_progress_bar = (
            '\r' in message or  # 回车符（tqdm 用于更新同一行）
            '\b' in message or  # 退格符
            message.startswith('\r') or  # 以回车符开头
            (len(message) > 0 and message[0] == '\r')  # 第一个字符是回车符
        )
        
        if is_progress_bar:
            # 进度条输出：同时写入终端和文件
            self.original_stream.write(message)
            self.original_stream.flush()
            # 也写入文件（但去掉 \r 以避免文件中的格式混乱）
            clean_message = message.replace('\r', '').replace('\b', '')
            if clean_message.strip():
                self.file.write(clean_message)
                self.file.flush()
        else:
            # 普通输出：只写入文件
            if message.strip():  # 忽略空行
                self.file.write(message)
                self.file.flush()
        
    def flush(self):
        self.file.flush()
        if hasattr(self.original_stream, 'flush'):
            self.original_stream.flush()
    
    def close(self):
        self.file.close()
    
    def isatty(self):
        """返回 True 以让 tqdm 认为这是一个终端"""
        return True

def setup_logger(log_file_path: str):
    """配置日志，输出到文件，并重定向标准输出和标准错误"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器
    logger.handlers = []
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 重定向标准输出和标准错误到日志文件
    tee_stdout = TeeLogger(log_file_path, sys.stdout)
    tee_stderr = TeeLogger(log_file_path, sys.stderr)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr
    
    return logger, tee_stdout, tee_stderr

# 全局logger，在main函数中初始化
logger = None
tee_stdout = None
tee_stderr = None

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
                if logger:
                    logger.warning(f"无法从URL加载图像: {e}")
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

        if logger:
            logger.warning(f"无法处理图像数据（字符串），长度={len(s)}, 预览={s[:100]}")
        return None

    # 其他类型兜底
    if logger:
        logger.warning(f"无法处理的图像数据类型: {type(data)}")
    return None

def process_single_sample(refinement_engine: RefinementEngine, image_data, prompt_text):
    """
    处理单个样本，使用 RefinementEngine 进行迭代细化
    
    Args:
        refinement_engine: RefinementEngine 实例
        image_data: 图像数据（bytes、base64字符串或本地路径）
        prompt_text: 提示文本
        
    Returns:
        结果字典，包含 generated_text 和其他信息
    """
    # 加载图像（支持Base64、URL或本地路径）
    image = load_image_from_data(image_data)
    if image is None:
        return None
    
    # 将图像保存为临时文件，因为 RefinementEngine.refine() 需要文件路径
    temp_file = None
    try:
        # 创建临时文件
        temp_fd, temp_file = tempfile.mkstemp(suffix='.png')
        os.close(temp_fd)
        
        # 保存图像到临时文件
        image.save(temp_file, 'PNG')
        
        # 使用 RefinementEngine 进行细化
        final_response, metadata = refinement_engine.refine(
            image_path=temp_file,
            base_instruction=prompt_text
        )
        
        result_dict = {
            'generated_text': final_response,
            'refinement_metadata': metadata,  # 添加细化过程的元数据
        }
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Error during refinement: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'error': str(e)}
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass

def find_failed_samples(output_dir: Path) -> List[Tuple[int, dict, Path]]:
    """
    检测输出目录下失败的样本（没有 generated_text 字段或包含 error 字段）
    
    Returns:
        List of (index, data_dict, filepath) tuples for failed samples
    """
    failed_samples = []
    json_files = sorted(output_dir.glob("*.json"))
    
    logger.info(f"正在检测失败的样本...")
    logger.info(f"找到 {len(json_files)} 个 JSON 文件")
    
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
                        logger.warning(f"警告: 文件 {filepath} 没有有效的索引，跳过")
                        continue
                
                failed_samples.append((index, data, filepath))
                reason = []
                if not has_generated_text:
                    reason.append("缺少 generated_text")
                logger.info(f"  发现失败样本: {filepath.name} (index: {index}) - {', '.join(reason)}")
        except Exception as e:
            logger.warning(f"警告: 读取文件 {filepath} 时出错: {e}")
            continue
    
    logger.info(f"共找到 {len(failed_samples)} 个失败的样本")
    return failed_samples

def reprocess_failed_samples(
    refinement_engine: RefinementEngine,
    df: pd.DataFrame, 
    failed_samples: List[Tuple[int, dict, Path]],
    output_dir: Path
):
    """
    重新处理失败的样本
    
    Args:
        refinement_engine: RefinementEngine 实例
        df: 原始数据集
        failed_samples: 失败的样本列表 [(index, data_dict, filepath), ...]
        output_dir: 输出目录
    """
    logger.info(f"\n开始重新处理 {len(failed_samples)} 个失败的样本...")
    
    successful_count = 0
    failed_count = 0
    
    for idx, (sample_index, old_data, filepath) in enumerate(tqdm(failed_samples, desc="重新处理失败样本")):
        try:
            # 从原始数据集中获取对应的行
            if sample_index not in df.index:
                logger.warning(f"  警告: 样本 {sample_index} 不在数据集中，跳过")
                failed_count += 1
                continue
            
            row = df.loc[sample_index]
            
            # 获取图像数据
            image_data = row.get('binary')
            if pd.isna(image_data) or image_data == '':
                logger.warning(f"  警告: 样本 {sample_index} 没有有效的图像数据")
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
                    logger.warning(f"  警告: 样本 {sample_index} 的图像数据类型无法处理: {type(image_data)}, 错误: {e}")
                    failed_count += 1
                    continue
            
            # 处理样本
            result = process_single_sample(
                refinement_engine,
                image_data, PROMPT_TEXT
            )
            
            if result is None or 'error' in result:
                logger.warning(f"  警告: 样本 {sample_index} 重新处理失败")
                failed_count += 1
                continue
            
            # 构建结果，移除 error 字段，保留其他原有字段（如果有）
            data_result = {
                'index': int(sample_index),
                'binary_data_length': len(str(image_data)) if image_data else 0,
                'binary_data_preview': str(image_data)[:50] + '...' if image_data and len(str(image_data)) > 50 else str(image_data),
                **result,  # 包含新的 generated_text, refinement_metadata等
                # 保留原有数据中的其他字段（除了 error 和生成相关的字段）
                **{col: old_data[col] for col in old_data.keys() 
                   if col not in ['error', 'generated_text', 'token_details', 'num_tokens', 
                                 'average_confidence', 'min_confidence', 'max_confidence',
                                 'intermediate_confidence_history', 'binary_data_length', 'binary_data_preview',
                                 'refinement_metadata']}
            }
            
            # 保存结果到原文件（覆盖）
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_result, f, ensure_ascii=False, indent=2)
            
            successful_count += 1
            logger.info(f"  ✓ 成功重新处理样本 {sample_index}")
            
        except Exception as e:
            logger.error(f"  ✗ 重新处理样本 {sample_index} 时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            failed_count += 1
    
    logger.info(f"\n重新处理完成！")
    logger.info(f"成功: {successful_count} 个")
    logger.info(f"失败: {failed_count} 个")

def main():
    global logger, tee_stdout, tee_stderr
    
    # 创建输出目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化日志
    log_file = output_dir / f"process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger, tee_stdout, tee_stderr = setup_logger(str(log_file))
    logger.info(f"日志文件: {log_file}")
    
    # 初始化 RefinementEngine
    logger.info(f"正在初始化 RefinementEngine...")
    logger.info(f"模型路径: {PRETRAINED}")
    logger.info(f"基础模型: {MODEL_BASE}")
    
    refinement_engine = RefinementEngine(
        model_path=PRETRAINED,
        model_base=MODEL_BASE,
        model_name=MODEL_NAME,
        vision_tower_path=VISION_TOWER_PATH,
        device=DEVICE,
        max_steps=MAX_STEPS,
        jitter_threshold=JITTER_THRESHOLD,
        span_k=SPAN_K,
        mask_expansion=MASK_EXPANSION,
        global_suppress_radius=GLOBAL_SUPPRESS_RADIUS,
        temp_dir=TEMP_DIR,
        logger=logger  # 传递logger给RefinementEngine
    )
    logger.info("RefinementEngine 初始化完成")
    
    # 加载数据集
    logger.info(f"正在加载数据集: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    logger.info(f"数据集包含 {len(df)} 条数据")
    logger.info(f"列名: {list(df.columns)}")
    
    # 检查binary列的数据类型
    if 'binary' in df.columns:
        sample_val = df['binary'].iloc[0] if len(df) > 0 else None
        logger.info(f"binary列数据类型: {df['binary'].dtype}")
        if sample_val is not None:
            logger.info(f"binary列示例数据类型: {type(sample_val)}")
            if isinstance(sample_val, str):
                logger.info(f"binary列示例数据长度: {len(sample_val)}")
                logger.info(f"binary列示例数据前100个字符: {sample_val[:100]}")
            elif isinstance(sample_val, bytes):
                logger.info(f"binary列示例数据长度: {len(sample_val)}")
                logger.info(f"binary列示例数据前100个字节: {sample_val[:100]}")
    
    # 保存完整的原始数据集（用于重新处理失败样本）
    df_full = df.copy()
    
    logger.info(f"输出目录: {output_dir}")
    
    # 检测并重新处理失败的样本
    logger.info("\n" + "="*60)
    logger.info("步骤 1: 检测失败的样本")
    logger.info("="*60)
    failed_samples = find_failed_samples(output_dir)
    
    if failed_samples:
        logger.info(f"\n找到 {len(failed_samples)} 个失败的样本，开始重新处理...")
        reprocess_failed_samples(
            refinement_engine,
            df_full, failed_samples, output_dir  # 使用完整数据集
        )
    else:
        logger.info("\n没有发现失败的样本，所有文件都包含有效的 generated_text")
    
    # 处理数据区间
    if START_INDEX is None and END_INDEX is None:
        logger.info("\n" + "="*60)
        logger.info("步骤 2: 处理新数据（START_INDEX=None, END_INDEX=None，跳过新数据处理）")
        logger.info("="*60)
        logger.info("提示: 如果只想重新处理失败的样本，可以设置 START_INDEX=None, END_INDEX=None")
        return
    
    # 根据索引区间筛选数据
    start_idx = START_INDEX if START_INDEX is not None else 0
    end_idx = END_INDEX if END_INDEX is not None else len(df)
    
    if start_idx >= len(df):
        logger.warning(f"起始索引 {start_idx} 超出数据集范围（共 {len(df)} 条），跳过处理")
        return
    
    end_idx = min(end_idx, len(df))
    df = df.iloc[start_idx:end_idx]
    logger.info(f"\n处理数据区间: [{start_idx}, {end_idx})，共 {len(df)} 条数据")
    
    logger.info("\n" + "="*60)
    logger.info("步骤 2: 处理新数据")
    logger.info("="*60)
    
    # 处理每条数据
    successful_count = 0
    failed_count = 0
    logger.info(f"\n开始处理 {len(df)} 条数据...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理数据"):
        try:
            # 获取图像数据（可能是Base64字符串或字节数据）
            image_data = row.get('binary')
            if pd.isna(image_data) or image_data == '':
                logger.warning(f"警告: 第 {idx} 条数据没有有效的图像数据")
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
                    logger.warning(f"警告: 第 {idx} 条数据的图像数据类型无法处理: {type(image_data)}, 错误: {e}")
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
                refinement_engine,
                image_data, PROMPT_TEXT
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
                **result,  # 包含generated_text, refinement_metadata等
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
            logger.error(f"\n处理第 {idx} 条数据时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
    logger.info(f"\n处理完成！")
    logger.info(f"总计: {total_count} 条")
    logger.info(f"成功: {successful_count} 条")
    logger.info(f"失败: {failed_count} 条")
    logger.info(f"结果已保存到目录: {output_dir}")
    logger.info(f"每个样本的结果保存在单独的文件中，文件名格式: {idx:06d}.json")
    
    # 恢复标准输出和标准错误
    if tee_stdout:
        sys.stdout = tee_stdout.original_stream
        tee_stdout.close()
    if tee_stderr:
        sys.stderr = tee_stderr.original_stream
        tee_stderr.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if logger:
            logger.error(f"程序异常退出: {e}")
            import traceback
            logger.error(traceback.format_exc())
        raise