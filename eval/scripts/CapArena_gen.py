from PIL import Image
import requests
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
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager

import sys
sys.path.insert(0, '/data0/swz/LLaDA-VGR/train')
from refinement_engine import RefinementEngine

prompt_interval_steps = 25
gen_interval_steps = 7
transfer_ratio = 0.25
use_fast_dllm = False  # using fast-dLLM (https://github.com/NVlabs/Fast-dLLM) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 6s to generate 128 tokens.
use_dllm_cache = False  # using dLLM-Cache(https://github.com/maomaocun/dLLM-cache) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 25s to generate 128 tokens.

warnings.filterwarnings("ignore")

# 配置参数
PRETRAINED = "/data0/swz/LLaDA-VGR/train/exp/llada_v_lora_rank64_1227"  # 训练好的模型路径
MODEL_BASE = "GSAI-ML/LLaDA-V"  # 基础模型路径
MODEL_NAME = "llava_llada_lora"  # 模型名称
DEVICE = "cuda:0"
DEVICE_MAP = "cuda:0"
NUM_GPUS = 4  # 使用的GPU数量

# RefinementEngine 配置参数
VISION_TOWER_PATH = "google/siglip2-so400m-patch14-384"
MAX_STEPS = 6  # 最大迭代次数
JITTER_THRESHOLD = 0.35  # Jitter 阈值
MASK_EXPANSION = 2  # Mask 扩张（仅在 mask_mode="span" 时使用）
TEMP_DIR = "./cropped_image"  # 临时文件目录
IMAGE_INPUT_MODE = "both"  # 图像输入模式: "original"（仅原图）、"crop"（仅局部图）、"both"（原图+局部图）
MASK_MODE = "span"  # Mask 模式: "span"（使用TextMiner解析span）、"single"（只mask高波动token）、"expand"（扩展高波动token左右各4个token）
TOKEN_SELECTION_MODE = "confidence"  # Token 选择模式: "jitter"（选择jitter最高的token）、"random"（随机选择一个token）、"confidence"（选择confidence最低的token）、"jitter_confidence"（选择jitter最高的token，如果jitter低于阈值，则选择confidence最低的token）

# CapArena 数据集路径和输出路径
IMAGE_DIR = "/data0/swz/exp/CapArena/data/caparena_auto_docci_600"  # 图片文件目录
OUTPUT_JSON = "/data0/swz/exp/CapArena/data/llada_vgr_0105_confidence.json"  # 输出JSON文件
START_INDEX = 0  # 起始索引（包含），设置为None表示从0开始
END_INDEX = None  # 终止索引（不包含），设置为None表示处理到末尾

# 生成参数
GEN_STEPS = 128
GEN_LENGTH = 128
BLOCK_LENGTH = 128
PREFIX_REFRESH_INTERVAL = 32
THRESHOLD = 1

# Prompt设置
PROMPT_TEXT = "Please describe the image in detail. Use less absolute directional descriptions. Do not repeat information."

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
        if logger:
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

def process_data_chunk(
    gpu_id: int,
    image_files: List[Tuple[str, Path]],  # List of (filename, image_path)
    output_json: str,
    result_queue: Queue,
    log_file: str,
    json_lock: mp.Lock
):
    """
    在单个GPU上处理数据块
    
    Args:
        gpu_id: GPU ID (0-3)
        image_files: 要处理的图片文件列表 [(filename, image_path), ...]
        output_json: 输出JSON文件路径
        result_queue: 用于返回结果的队列
        log_file: 日志文件路径
        json_lock: JSON文件写入锁
    """
    # 为每个进程设置独立的日志
    process_logger = logging.getLogger(f"GPU_{gpu_id}")
    process_logger.setLevel(logging.INFO)
    process_logger.handlers = []
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file.replace('.log', f'_gpu{gpu_id}.log'), encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(f'%(asctime)s - GPU{gpu_id} - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    process_logger.addHandler(file_handler)
    
    device = f"cuda:{gpu_id}"
    process_logger.info(f"GPU {gpu_id}: 开始初始化 RefinementEngine，使用设备: {device}")
    process_logger.info(f"GPU {gpu_id}: 模型路径: {PRETRAINED}")
    process_logger.info(f"GPU {gpu_id}: 基础模型: {MODEL_BASE}")
    
    # 创建临时目录
    temp_dir_gpu = f"{TEMP_DIR}_gpu{gpu_id}"
    os.makedirs(temp_dir_gpu, exist_ok=True)
    process_logger.info(f"GPU {gpu_id}: 临时目录: {temp_dir_gpu}")
    
    try:
        # 初始化 RefinementEngine
        process_logger.info(f"GPU {gpu_id}: 正在加载模型（这可能需要几分钟）...")
        process_logger.info(f"GPU {gpu_id}: 开始创建 RefinementEngine 实例...")
        
        refinement_engine = RefinementEngine(
            model_path=PRETRAINED,
            model_base=MODEL_BASE,
            model_name=MODEL_NAME,
            vision_tower_path=VISION_TOWER_PATH,
            device=device,
            max_steps=MAX_STEPS,
            jitter_threshold=JITTER_THRESHOLD,
            mask_expansion=MASK_EXPANSION,
            temp_dir=temp_dir_gpu,  # 每个GPU使用独立的临时目录
            image_input_mode=IMAGE_INPUT_MODE,
            mask_mode=MASK_MODE,
            token_selection_mode=TOKEN_SELECTION_MODE,
            logger=process_logger
        )
        process_logger.info(f"GPU {gpu_id}: RefinementEngine 初始化完成！")
        
        successful_count = 0
        failed_count = 0
        
        process_logger.info(f"GPU {gpu_id}: 开始处理 {len(image_files)} 张图片...")
        
        for filename, image_path in tqdm(image_files, desc=f"GPU {gpu_id} 处理图片"):
            try:
                if not image_path.exists():
                    process_logger.warning(f"GPU {gpu_id}: 警告: 图片文件不存在: {image_path}")
                    # 更新JSON文件，标记为错误
                    with json_lock:
                        try:
                            with open(output_json, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            data[filename] = None  # 保持为null表示失败
                            with open(output_json, 'w', encoding='utf-8') as f:
                                json.dump(data, f, ensure_ascii=False)
                        except Exception as e:
                            process_logger.error(f"GPU {gpu_id}: 更新JSON文件失败: {e}")
                    failed_count += 1
                    continue
                
                # 使用文件路径作为图像数据
                image_data = str(image_path)
                
                # 处理样本
                result = process_single_sample(
                    refinement_engine,
                    image_data, PROMPT_TEXT
                )
                
                if result is None or 'error' in result:
                    process_logger.warning(f"GPU {gpu_id}: 处理图片失败: {filename}")
                    # 更新JSON文件，标记为错误
                    with json_lock:
                        try:
                            with open(output_json, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            data[filename] = None  # 保持为null表示失败
                            with open(output_json, 'w', encoding='utf-8') as f:
                                json.dump(data, f, ensure_ascii=False)
                        except Exception as e:
                            process_logger.error(f"GPU {gpu_id}: 更新JSON文件失败: {e}")
                    failed_count += 1
                    continue
                
                # 获取生成的caption
                generated_text = result.get('generated_text', '')
                
                # 更新JSON文件
                with json_lock:
                    try:
                        # 读取现有JSON文件
                        with open(output_json, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        # 更新对应文件的caption
                        data[filename] = generated_text
                        # 写回JSON文件
                        with open(output_json, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False)
                    except Exception as e:
                        process_logger.error(f"GPU {gpu_id}: 更新JSON文件失败: {e}")
                        failed_count += 1
                        continue
                
                successful_count += 1
                
            except Exception as e:
                process_logger.error(f"GPU {gpu_id}: 处理图片 {filename} 时出错: {e}")
                import traceback
                process_logger.error(traceback.format_exc())
                # 更新JSON文件，标记为错误
                with json_lock:
                    try:
                        with open(output_json, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        data[filename] = None  # 保持为null表示失败
                        with open(output_json, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False)
                    except Exception as e2:
                        process_logger.error(f"GPU {gpu_id}: 更新JSON文件失败: {e2}")
                failed_count += 1
        
        # 返回结果
        result_queue.put({
            'gpu_id': gpu_id,
            'successful_count': successful_count,
            'failed_count': failed_count,
            'total_count': successful_count + failed_count
        })
        
        process_logger.info(f"GPU {gpu_id}: 处理完成！成功: {successful_count}, 失败: {failed_count}")
        
    except Exception as e:
        process_logger.error(f"GPU {gpu_id}: 进程异常: {e}")
        import traceback
        process_logger.error(traceback.format_exc())
        result_queue.put({
            'gpu_id': gpu_id,
            'successful_count': 0,
            'failed_count': len(image_files),
            'total_count': len(image_files),
            'error': str(e)
        })

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
            
            # 从文件系统读取图像文件
            image_path = Path(IMAGE_DIR) / f"{sample_index:04d}.jpg"
            if not image_path.exists():
                logger.warning(f"  警告: 样本 {sample_index} 的图片文件不存在: {image_path}")
                failed_count += 1
                continue
            
            # 使用文件路径作为图像数据
            image_data = str(image_path)
            
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
                'image_path': str(image_path),
                **result,  # 包含新的 generated_text, refinement_metadata等
                # 保留原有数据中的其他字段（除了 error 和生成相关的字段）
                **{col: old_data[col] for col in old_data.keys() 
                   if col not in ['error', 'generated_text', 'token_details', 'num_tokens', 
                                 'average_confidence', 'min_confidence', 'max_confidence',
                                 'intermediate_confidence_history', 'binary_data_length', 'binary_data_preview',
                                 'refinement_metadata', 'image_path', 'binary']}
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
    
    # 创建日志目录
    log_dir = Path(OUTPUT_JSON).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化日志
    log_file = log_dir / f"caparena_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger, tee_stdout, tee_stderr = setup_logger(str(log_file))
    logger.info(f"日志文件: {log_file}")
    
    # 检查图片目录
    image_dir_path = Path(IMAGE_DIR)
    if not image_dir_path.exists():
        logger.error(f"错误: 图片目录不存在: {IMAGE_DIR}")
        raise FileNotFoundError(f"图片目录不存在: {IMAGE_DIR}")
    logger.info(f"图片目录: {IMAGE_DIR}")
    
    # 检查输出JSON文件
    output_json_path = Path(OUTPUT_JSON)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 读取或创建输出JSON文件
    if output_json_path.exists():
        logger.info(f"读取现有JSON文件: {OUTPUT_JSON}")
        with open(output_json_path, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
    else:
        logger.info(f"创建新的JSON文件: {OUTPUT_JSON}")
        output_data = {}
    
    # 获取所有图片文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    all_image_files = []
    for ext in image_extensions:
        all_image_files.extend(image_dir_path.glob(f"*{ext}"))
        all_image_files.extend(image_dir_path.glob(f"*{ext.upper()}"))
    
    # 按文件名排序
    all_image_files = sorted(all_image_files, key=lambda x: x.name)
    logger.info(f"找到 {len(all_image_files)} 张图片")
    
    # 筛选需要处理的图片（只处理JSON中值为null的）
    image_files_to_process = []
    for image_path in all_image_files:
        filename = image_path.name
        # 如果JSON中没有这个文件，或者值为null，则需要处理
        if filename not in output_data or output_data[filename] is None:
            image_files_to_process.append((filename, image_path))
    
    logger.info(f"需要处理的图片: {len(image_files_to_process)} 张")
    
    # 处理数据区间
    if START_INDEX is not None or END_INDEX is not None:
        start_idx = START_INDEX if START_INDEX is not None else 0
        end_idx = END_INDEX if END_INDEX is not None else len(image_files_to_process)
        if start_idx >= len(image_files_to_process):
            logger.warning(f"起始索引 {start_idx} 超出范围（共 {len(image_files_to_process)} 张），跳过处理")
            return
        end_idx = min(end_idx, len(image_files_to_process))
        image_files_to_process = image_files_to_process[start_idx:end_idx]
        logger.info(f"处理数据区间: [{start_idx}, {end_idx})，共 {len(image_files_to_process)} 张图片")
    
    if len(image_files_to_process) == 0:
        logger.info("没有需要处理的图片")
        return
    
    logger.info("\n" + "="*60)
    logger.info("处理图片（多GPU并行）")
    logger.info("="*60)
    
    # 使用多GPU并行处理
    logger.info(f"\n开始使用 {NUM_GPUS} 张GPU并行处理 {len(image_files_to_process)} 张图片...")
    
    # 将图片文件分割成NUM_GPUS份
    chunk_size = len(image_files_to_process) // NUM_GPUS
    chunks = []
    for i in range(NUM_GPUS):
        start_idx = i * chunk_size
        if i == NUM_GPUS - 1:
            # 最后一份包含所有剩余数据
            end_idx = len(image_files_to_process)
        else:
            end_idx = (i + 1) * chunk_size
        chunk = image_files_to_process[start_idx:end_idx]
        chunks.append(chunk)
        logger.info(f"GPU {i}: 分配 {len(chunk)} 张图片 (索引 {start_idx} 到 {end_idx-1})")
    
    # 创建结果队列和锁
    manager = Manager()
    result_queue = manager.Queue()
    json_lock = manager.Lock()  # 用于保护JSON文件写入
    
    # 创建进程列表
    processes = []
    log_file_base = str(log_file).replace('.log', '')
    
    # 启动多个进程（添加延迟以避免同时加载模型导致资源竞争）
    for gpu_id in range(NUM_GPUS):
        if len(chunks[gpu_id]) > 0:
            p = Process(
                target=process_data_chunk,
                args=(gpu_id, chunks[gpu_id], OUTPUT_JSON, result_queue, log_file_base, json_lock)
            )
            p.start()
            processes.append(p)
            logger.info(f"启动 GPU {gpu_id} 进程 (PID: {p.pid})")
            
            # 添加延迟，避免所有进程同时加载模型
            # 每个进程延迟 gpu_id * 5 秒，让它们错开启动
            if gpu_id < NUM_GPUS - 1:  # 最后一个进程不需要延迟
                delay = (gpu_id + 1) * 5  # 5秒、10秒、15秒...
                logger.info(f"等待 {delay} 秒后启动下一个GPU进程...")
                time.sleep(delay)
        else:
            logger.info(f"GPU {gpu_id}: 没有数据需要处理，跳过")
    
    # 等待所有进程完成
    logger.info(f"\n等待 {len(processes)} 个进程完成...")
    logger.info("提示: 模型加载可能需要几分钟时间，请耐心等待...")
    logger.info("提示: 可以查看各GPU的独立日志文件了解详细进度")
    
    for idx, p in enumerate(processes):
        logger.info(f"等待进程 {p.pid} (GPU {idx}) 完成...")
        p.join()
        logger.info(f"进程 {p.pid} (GPU {idx}) 已完成")
    
    # 收集结果
    total_successful = 0
    total_failed = 0
    total_count = 0
    
    logger.info("\n收集各GPU的处理结果...")
    while not result_queue.empty():
        result = result_queue.get()
        gpu_id = result['gpu_id']
        successful = result['successful_count']
        failed = result['failed_count']
        count = result['total_count']
        
        total_successful += successful
        total_failed += failed
        total_count += count
        
        if 'error' in result:
            logger.error(f"GPU {gpu_id}: 处理过程中出现错误: {result['error']}")
        else:
            logger.info(f"GPU {gpu_id}: 成功 {successful} 条, 失败 {failed} 条, 总计 {count} 条")
    
    # 统计信息
    logger.info(f"\n" + "="*60)
    logger.info(f"所有GPU处理完成！")
    logger.info(f"总计: {total_count} 张")
    logger.info(f"成功: {total_successful} 张")
    logger.info(f"失败: {total_failed} 张")
    logger.info(f"结果已保存到JSON文件: {OUTPUT_JSON}")
    logger.info(f"各GPU的日志文件: {log_file_base}_gpu0.log 到 {log_file_base}_gpu{NUM_GPUS-1}.log")
    logger.info("="*60)
    
    # 恢复标准输出和标准错误
    if tee_stdout:
        sys.stdout = tee_stdout.original_stream
        tee_stdout.close()
    if tee_stderr:
        sys.stderr = tee_stderr.original_stream
        tee_stderr.close()

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    try:
        main()
    except Exception as e:
        if logger:
            logger.error(f"程序异常退出: {e}")
            import traceback
            logger.error(traceback.format_exc())
        raise