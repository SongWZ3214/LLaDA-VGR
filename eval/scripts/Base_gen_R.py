import torch
torch.set_default_dtype(torch.bfloat16)
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

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "train"))
from refinement_engine import RefinementEngine

prompt_interval_steps = 25
gen_interval_steps = 7
transfer_ratio = 0.25
use_fast_dllm = False
use_dllm_cache = False

warnings.filterwarnings("ignore")

PRETRAINED = os.environ.get("PRETRAINED_MODEL", str(WORKSPACE_ROOT / "models" / "LLaDA-V"))
MODEL_BASE = None
MODEL_NAME = "llava_llada_base"
DEVICE = "cuda:0"
DEVICE_MAP = "cuda:0"
NUM_GPUS = 1

# RefinementEngine 配置参数
VISION_TOWER_PATH = "google/siglip2-so400m-patch14-384"
MAX_STEPS = 0  # 彻底关闭 VGR 迭代优化，只做单次生成
JITTER_THRESHOLD = 0.0

MASK_EXPANSION = 2
TEMP_DIR = "./cropped_image"
IMAGE_INPUT_MODE = "both"
MASK_MODE = "span"
TOKEN_SELECTION_MODE = "confidence"

IMAGE_DIR = os.environ.get("AMBER_IMAGE_DIR", str(WORKSPACE_ROOT / "exp" / "AMBER" / "data" / "image"))
BASE_QUERY_JSON = os.environ.get(
    "AMBER_QUERY_JSON",
    str(WORKSPACE_ROOT / "exp" / "AMBER" / "data" / "query" / "query_generative_half0328base.json"),
)

dino_thresh = os.environ.get("DINO_THRESH", None)
if dino_thresh is not None:
    QUERY_JSON = BASE_QUERY_JSON.replace(".json", f"-thresh_{dino_thresh}.json")
else:
    QUERY_JSON = BASE_QUERY_JSON

START_INDEX = 0
END_INDEX = 50

# 生成参数
GEN_STEPS = 128
GEN_LENGTH = 128
BLOCK_LENGTH = 128
PREFIX_REFRESH_INTERVAL = 32
THRESHOLD = 1

DEFAULT_PROMPT_TEXT = "Please describe the image in detail. Use less absolute directional descriptions. Do not repeat information."
BASE64_RE = re.compile(r'^[A-Za-z0-9+/]+={0,2}$')

class TeeLogger:
    def __init__(self, file_path, original_stream):
        self.file = open(file_path, 'a', encoding='utf-8')
        self.original_stream = original_stream

    def write(self, message):
        self.original_stream.write(message)
        if hasattr(self.original_stream, 'flush'):
            self.original_stream.flush()

        is_progress_bar = (
                '\r' in message or
                '\b' in message or
                message.startswith('\r') or
                (len(message) > 0 and message[0] == '\r')
        )

        if is_progress_bar:
            self.original_stream.write(message)
            self.original_stream.flush()
            clean_message = message.replace('\r', '').replace('\b', '')
            if clean_message.strip():
                self.file.write(clean_message)
                self.file.flush()
        else:
            if message.strip():
                self.file.write(message)
                self.file.flush()

    def flush(self):
        self.file.flush()
        if hasattr(self.original_stream, 'flush'):
            self.original_stream.flush()

    def close(self):
        self.file.close()

    def isatty(self):
        return True


def setup_logger(log_file_path: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    tee_stdout = TeeLogger(log_file_path, sys.stdout)
    tee_stderr = TeeLogger(log_file_path, sys.stderr)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    return logger, tee_stdout, tee_stderr


logger = None
tee_stdout = None
tee_stderr = None


def _pil_from_bytes(b: bytes):
    try:
        im = Image.open(io.BytesIO(b))
        im.load()
        return im.convert("RGB")
    except Exception:
        return None

def load_image_from_data(data):
    if isinstance(data, (bytes, bytearray, memoryview)):
        b = bytes(data)
        img = _pil_from_bytes(b)
        if img is not None:
            return img
        try:
            data = b.decode("utf-8", errors="ignore")
        except Exception:
            pass

    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            if data.dtype == np.uint8:
                if data.ndim == 1:
                    img = _pil_from_bytes(data.tobytes())
                    if img is not None:
                        return img
                elif data.ndim in (2, 3):
                    return Image.fromarray(data).convert("RGB")
    except Exception:
        pass

    if isinstance(data, str):
        s = data.strip()
        if s.lower().startswith("data:image/"):
            try:
                comma = s.find(',')
                if comma != -1:
                    b64 = s[comma + 1:].strip()
                    b = base64.b64decode(b64)
                    img = _pil_from_bytes(b)
                    if img is not None:
                        return img
            except Exception:
                pass

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

        looks_hex_escaped = bool(re.search(r'\\x[0-9A-Fa-f]{2}', s))
        is_base64_like = (
                len(s) > 32 and
                len(s.replace("\n", "").replace("\r", "").replace(" ", "")) % 4 == 0 and
                bool(BASE64_RE.match(s.replace("\n", "").replace("\r", "").replace(" ", "")))
        )

        if looks_hex_escaped and not is_base64_like:
            try:
                if s.startswith(("b'", 'b"')) and s.endswith(("'", '"')):
                    s_inner = s[2:-1]
                else:
                    s_inner = s
                b = codecs.decode(s_inner, 'unicode_escape').encode('latin-1')
                img = _pil_from_bytes(b)
                if img is not None:
                    return img
            except Exception:
                pass

        compact = s.replace("\n", "").replace("\r", "").replace(" ", "")
        if (
                compact.startswith(("/9j/", "iVBOR", "R0lG", "UklG")) or
                (len(compact) > 32 and len(compact) % 4 == 0 and BASE64_RE.match(compact))
        ):
            try:
                b = base64.b64decode(compact)
                img = _pil_from_bytes(b)
                if img is not None:
                    return img
            except Exception:
                pass

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

    if logger:
        logger.warning(f"无法处理的图像数据类型: {type(data)}")
    return None

def process_single_sample(refinement_engine: RefinementEngine, image_data, prompt_text, device="cuda:0"):
    image = load_image_from_data(image_data)
    if image is None:
        return None

    temp_file = None
    try:
        temp_fd, temp_file = tempfile.mkstemp(suffix='.png')
        os.close(temp_fd)
        image.save(temp_file, 'PNG')

        torch_device = torch.device(device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch_device)
            torch.cuda.synchronize(torch_device)

        start_time = time.time()

        final_response, metadata = refinement_engine.refine(
            image_path=temp_file,
            base_instruction=prompt_text
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize(torch_device)

        end_time = time.time()
        latency = end_time - start_time

        peak_allocated_mb = 0.0
        peak_reserved_mb = 0.0
        if torch.cuda.is_available():
            peak_allocated_mb = torch.cuda.max_memory_allocated(torch_device) / (1024 * 1024)
            peak_reserved_mb = torch.cuda.max_memory_reserved(torch_device) / (1024 * 1024)

        initial_response = metadata.get('initial_response', '')

        result_dict = {
            'initial_response': initial_response,
            'response': final_response,
            'refinement_metadata': metadata,
            'latency_seconds': latency,
            'peak_allocated_mb': peak_allocated_mb,
            'peak_reserved_mb': peak_reserved_mb
        }

        return result_dict

    except Exception as e:
        logger.error(f"Error during refinement: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'error': str(e)}
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass


def find_failed_samples(output_dir: Path) -> List[Tuple[int, dict, Path]]:
    failed_samples = []
    json_files = sorted(output_dir.glob("*.json"))

    logger.info(f"正在检测失败的样本...")
    logger.info(f"找到 {len(json_files)} 个 JSON 文件")

    for filepath in json_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            has_response = 'response' in data and data.get('response', '').strip()

            if not has_response:
                index = data.get('index')
                if index is None:
                    try:
                        index = int(filepath.stem)
                    except ValueError:
                        logger.warning(f"警告: 文件 {filepath} 没有有效的索引，跳过")
                        continue

                failed_samples.append((index, data, filepath))
                reason = []
                if not has_response:
                    reason.append("缺少 response")
                logger.info(f"  发现失败样本: {filepath.name} (index: {index}) - {', '.join(reason)}")
        except Exception as e:
            logger.warning(f"警告: 读取文件 {filepath} 时出错: {e}")
            continue

    logger.info(f"共找到 {len(failed_samples)} 个失败的样本")
    return failed_samples


def process_data_chunk(
        gpu_id: int,
        tasks: List[Tuple[int, str, str, Path]],
        query_json: str,
        result_queue: Queue,
        log_file: str,
        json_lock: mp.Lock
):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"  # 核心修复：保持为 cuda:0，去掉后面冲突的重新赋值

    process_logger = logging.getLogger(f"GPU_{gpu_id}")
    process_logger.setLevel(logging.INFO)
    process_logger.handlers = []

    file_handler = logging.FileHandler(log_file.replace('.log', f'_gpu{gpu_id}.log'), encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(f'%(asctime)s - GPU{gpu_id} - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    process_logger.addHandler(file_handler)

    process_logger.info(f"GPU {gpu_id}: 开始初始化 RefinementEngine，物理卡号: {gpu_id}, 映射设备: {device}")
    process_logger.info(f"GPU {gpu_id}: 模型路径: {PRETRAINED}")

    temp_dir_gpu = f"{TEMP_DIR}_gpu{gpu_id}"
    os.makedirs(temp_dir_gpu, exist_ok=True)

    try:
        refinement_engine = RefinementEngine(
            model_path=PRETRAINED,
            model_base=MODEL_BASE,
            model_name=MODEL_NAME,
            vision_tower_path=VISION_TOWER_PATH,
            device=device,
            max_steps=MAX_STEPS,
            jitter_threshold=JITTER_THRESHOLD,
            mask_expansion=MASK_EXPANSION,
            temp_dir=temp_dir_gpu,
            image_input_mode=IMAGE_INPUT_MODE,
            mask_mode=MASK_MODE,
            token_selection_mode=TOKEN_SELECTION_MODE,
            logger=process_logger
        )
        process_logger.info(f"GPU {gpu_id}: RefinementEngine 初始化完成！")

        successful_count = 0
        failed_count = 0

        for entry_id, image_filename, query_text, image_path in tqdm(tasks, desc=f"GPU {gpu_id} 处理任务"):
            try:
                if not image_path.exists():
                    process_logger.warning(f"GPU {gpu_id}: 警告: 图片文件不存在: {image_path}")
                    with json_lock:
                        try:
                            with open(query_json, 'r', encoding='utf-8') as f:
                                query_data = json.load(f)
                            for entry in query_data:
                                if entry.get('id') == entry_id:
                                    entry['response'] = None
                                    entry['error'] = f'Image file not found: {image_path}'
                                    break
                            with open(query_json, 'w', encoding='utf-8') as f:
                                json.dump(query_data, f, ensure_ascii=False, indent=2)
                        except Exception as e:
                            process_logger.error(f"GPU {gpu_id}: 更新JSON文件失败: {e}")
                    failed_count += 1
                    continue

                image_data = str(image_path)
                prompt_text = query_text if query_text and query_text.strip() else DEFAULT_PROMPT_TEXT

                result = process_single_sample(
                    refinement_engine,
                    image_data,
                    prompt_text,
                    device=device
                )

                if result is None or 'error' in result:
                    process_logger.warning(f"GPU {gpu_id}: 处理任务 {entry_id} 失败: {image_filename}")
                    with json_lock:
                        try:
                            with open(query_json, 'r', encoding='utf-8') as f:
                                query_data = json.load(f)
                            for entry in query_data:
                                if entry.get('id') == entry_id:
                                    entry['response'] = None
                                    if 'error' in result:
                                        entry['error'] = result['error']
                                    else:
                                        entry['error'] = 'Failed to generate caption'
                                    break
                            with open(query_json, 'w', encoding='utf-8') as f:
                                json.dump(query_data, f, ensure_ascii=False, indent=2)
                        except Exception as e:
                            process_logger.error(f"GPU {gpu_id}: 更新JSON文件失败: {e}")
                    failed_count += 1
                    continue

                response = result.get('response', '')
                initial_response = result.get('initial_response', '')
                latency = result.get('latency_seconds', 0.0)
                peak_allocated = result.get('peak_allocated_mb', 0.0)
                peak_reserved = result.get('peak_reserved_mb', 0.0)

                with json_lock:
                    try:
                        with open(query_json, 'r', encoding='utf-8') as f:
                            query_data = json.load(f)
                        for entry in query_data:
                            if entry.get('id') == entry_id:
                                entry['initial_response'] = initial_response
                                entry['response'] = response
                                entry['latency_seconds'] = latency
                                entry['peak_allocated_mb'] = peak_allocated
                                entry['peak_reserved_mb'] = peak_reserved
                                if 'refinement_metadata' in result:
                                    entry['refinement_metadata'] = result['refinement_metadata']
                                if 'error' in entry:
                                    del entry['error']
                                break
                        with open(query_json, 'w', encoding='utf-8') as f:
                            json.dump(query_data, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        process_logger.error(f"GPU {gpu_id}: 更新JSON文件失败: {e}")
                        failed_count += 1
                        continue

                successful_count += 1

            except Exception as e:
                process_logger.error(f"GPU {gpu_id}: 处理任务 {entry_id} ({image_filename}) 时出错: {e}")
                with json_lock:
                    try:
                        with open(query_json, 'r', encoding='utf-8') as f:
                            query_data = json.load(f)
                        for entry in query_data:
                            if entry.get('id') == entry_id:
                                entry['response'] = None
                                entry['error'] = str(e)
                                break
                        with open(query_json, 'w', encoding='utf-8') as f:
                            json.dump(query_data, f, ensure_ascii=False, indent=2)
                    except Exception as e2:
                        process_logger.error(f"GPU {gpu_id}: 更新JSON文件失败: {e2}")
                failed_count += 1

        result_queue.put({
            'gpu_id': gpu_id,
            'successful_count': successful_count,
            'failed_count': failed_count,
            'total_count': successful_count + failed_count
        })

    except Exception as e:
        process_logger.error(f"GPU {gpu_id}: 进程异常: {e}")
        import traceback
        process_logger.error(traceback.format_exc())
        result_queue.put({
            'gpu_id': gpu_id,
            'successful_count': 0,
            'failed_count': len(tasks),
            'total_count': len(tasks),
            'error': str(e)
        })

def reprocess_failed_samples(
        refinement_engine: RefinementEngine,
        df: pd.DataFrame,
        failed_samples: List[Tuple[int, dict, Path]],
        output_dir: Path
):
    logger.info(f"\n开始重新处理 {len(failed_samples)} 个失败的样本...")

    successful_count = 0
    failed_count = 0

    for idx, (sample_index, old_data, filepath) in enumerate(tqdm(failed_samples, desc="重新处理失败样本")):
        try:
            if sample_index not in df.index:
                logger.warning(f"  警告: 样本 {sample_index} 不在数据集中，跳过")
                failed_count += 1
                continue

            row = df.loc[sample_index]

            image_path = Path(IMAGE_DIR) / f"{sample_index:04d}.jpg"
            if not image_path.exists():
                logger.warning(f"  警告: 样本 {sample_index} 的图片文件不存在: {image_path}")
                failed_count += 1
                continue

            image_data = str(image_path)

            result = process_single_sample(
                refinement_engine,
                image_data,
                DEFAULT_PROMPT_TEXT,
                device="cuda:0"
            )

            if result is None or 'error' in result:
                logger.warning(f"  警告: 样本 {sample_index} 重新处理失败")
                failed_count += 1
                continue

            data_result = {
                'index': int(sample_index),
                'image_path': str(image_path),
                **result,
                **{col: old_data[col] for col in old_data.keys()
                   if col not in ['error', 'response', 'token_details', 'num_tokens',
                                  'average_confidence', 'min_confidence', 'max_confidence',
                                  'intermediate_confidence_history', 'binary_data_length', 'binary_data_preview',
                                  'refinement_metadata', 'image_path', 'binary']}
            }

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

    log_dir = Path(QUERY_JSON).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"lladav_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger, tee_stdout, tee_stderr = setup_logger(str(log_file))
    logger.info(f"日志文件: {log_file}")

    image_dir_path = Path(IMAGE_DIR)
    if not image_dir_path.exists():
        logger.error(f"错误: 图片目录不存在: {IMAGE_DIR}")
        raise FileNotFoundError(f"图片目录不存在: {IMAGE_DIR}")

    query_json_path = Path(QUERY_JSON)
    base_json_path = Path(BASE_QUERY_JSON)
    query_json_path.parent.mkdir(parents=True, exist_ok=True)

    import shutil
    if not query_json_path.exists() and base_json_path.exists():
        logger.info(f"正在从基础文件复制工作副本: {QUERY_JSON}")
        shutil.copy2(base_json_path, query_json_path)

    if not query_json_path.exists():
        logger.error(f"错误: query_generative.json文件不存在: {QUERY_JSON}")
        raise FileNotFoundError(f"query_generative.json文件不存在: {QUERY_JSON}")

    try:
        with open(query_json_path, 'r', encoding='utf-8') as f:
            query_data = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"JSON文件格式错误: {e}")
        raise

    if not isinstance(query_data, list):
        raise ValueError(f"query_generative.json应该是一个数组")

    tasks_to_process = []
    for entry in query_data:
        entry_id = entry.get('id')
        image_filename = entry.get('image')
        query_text = entry.get('query', '')

        if entry_id is None or not image_filename:
            continue

        if 'response' in entry and entry['response'] and entry['response'].strip():
            continue

        image_path = image_dir_path / image_filename
        tasks_to_process.append((entry_id, image_filename, query_text, image_path))

    if START_INDEX is not None or END_INDEX is not None:
        start_idx = START_INDEX if START_INDEX is not None else 0
        end_idx = END_INDEX if END_INDEX is not None else len(tasks_to_process)
        end_idx = min(end_idx, len(tasks_to_process))
        tasks_to_process = tasks_to_process[start_idx:end_idx]

    if len(tasks_to_process) == 0:
        return

    chunk_size = len(tasks_to_process) // NUM_GPUS
    chunks = []
    for i in range(NUM_GPUS):
        start_idx = i * chunk_size
        end_idx = len(tasks_to_process) if i == NUM_GPUS - 1 else (i + 1) * chunk_size
        chunks.append(tasks_to_process[start_idx:end_idx])

    manager = Manager()
    result_queue = manager.Queue()
    json_lock = manager.Lock()

    processes = []
    log_file_base = str(log_file).replace('.log', '')

    for gpu_id in range(NUM_GPUS):
        if len(chunks[gpu_id]) > 0:
            p = Process(
                target=process_data_chunk,
                args=(gpu_id, chunks[gpu_id], QUERY_JSON, result_queue, log_file_base, json_lock)
            )
            p.start()
            processes.append(p)
            if gpu_id < NUM_GPUS - 1:
                time.sleep((gpu_id + 1) * 5)

    for idx, p in enumerate(processes):
        p.join()

    total_successful = 0
    total_failed = 0
    total_count = 0

    while not result_queue.empty():
        result = result_queue.get()
        total_successful += result['successful_count']
        total_failed += result['failed_count']
        total_count += result['total_count']

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
        raise