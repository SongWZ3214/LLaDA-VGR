import json
import os
import sys
import argparse
import subprocess
import math
from pathlib import Path
from typing import List, Tuple

# ==========================================
# 工具函数
# ==========================================

def load_json_files(file_paths: List[Path]) -> List[Tuple[int, dict, str]]:
    """加载指定列表的 JSON 文件"""
    files_data = []
    for filepath in file_paths:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            index = data.get('index')
            if index is None:
                try:
                    index = int(filepath.stem)
                except ValueError:
                    continue
            files_data.append((index, data, str(filepath)))
        except Exception:
            continue
    return files_data

def prepare_data_for_capture(files_data):
    """准备 CAPTURE 库需要的 refs 和 preds"""
    refs = {}
    preds = {}
    file_mapping = [] # (index, original_filepath)
        
    for index, data, filepath in files_data:
        if 'generated_text' not in data: continue
        
        gt_captions = [
            data.get('GT_Caption_GPT4O'),
            data.get('GT_Caption_GPT4V'),
            data.get('GT_Caption_Gemini15Pro')
        ]
        valid_gt = [cap for cap in gt_captions if cap]
        if not valid_gt: continue
        
        generated_text = data.get('generated_text')
        if not generated_text: continue
        
        refs[index] = valid_gt
        preds[index] = [generated_text]
        file_mapping.append((index, filepath))

    return refs, preds, file_mapping

# ==========================================
# Worker 逻辑 (子进程运行的部分)
# ==========================================

def run_worker(gpu_id: str, temp_input_path: str, temp_output_path: str):
    """
    Worker 模式入口。
    在一个独立的进程中运行，只负责计算分数。
    """
    # 1. 设置环境变量，只可见指定的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    # 2. 只有在 Worker 里才导入重型库
    try:
        import torch
        # 必须先导入 torch 再 patch multiprocessing，或者反之，
        # 但为了防止 capture 库内部提前引用，我们需要尽早 Patch
    except ImportError:
        print(f"[GPU {gpu_id}] 错误: 无法导入 torch")
        sys.exit(1)

    # --- 关键修正：完善的伪造 Pool 类 ---
    class SynchronousPool:
        """
        用于欺骗 capture_metric，让它以为自己在多进程跑，实际上是在当前线程串行跑。
        """
        def __init__(self, *args, **kwargs): 
            pass
        
        # 模拟上下文管理器 (with Pool() as pool:)
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def apply_async(self, func, args=(), kwds=None, callback=None, error_callback=None):
            if kwds is None: kwds = {}
            # 立即执行函数
            try:
                res = func(*args, **kwds)
                
                # 返回一个假的 AsyncResult 对象
                class DummyResult:
                    def __init__(self, val): self._val = val
                    def get(self, timeout=None): return self._val
                    def wait(self, timeout=None): pass
                    def ready(self): return True
                    def successful(self): return True
                
                return DummyResult(res)
            except Exception as e:
                if error_callback: error_callback(e)
                raise e

        def close(self): pass
        def join(self): pass
        def terminate(self): pass

    # 3. 进行 Monkey Patch
    import multiprocessing
    multiprocessing.Pool = SynchronousPool

    # 4. 此时再导入 CAPTURE，确保它引用到的是被修改后的 Pool
    try:
        from capture_metric.capture import CAPTURE
    except ImportError:
        print(f"[GPU {gpu_id}] 错误: 无法导入 capture_metric")
        sys.exit(1)

    print(f"[GPU {gpu_id}] Worker 启动，正在加载任务文件: {temp_input_path}")
    
    # 5. 读取分配到的文件列表
    with open(temp_input_path, 'r', encoding='utf-8') as f:
        target_files = json.load(f)
    
    target_paths = [Path(p) for p in target_files]
    files_data = load_json_files(target_paths)
        
    if not files_data:
        print(f"[GPU {gpu_id}] 没有有效数据，退出。")
        with open(temp_output_path, 'w') as f: 
            json.dump({}, f)
            return
        
    # 6. 准备数据
    refs, preds, file_mapping = prepare_data_for_capture(files_data)
    print(f"[GPU {gpu_id}] 准备评估 {len(refs)} 个样本...")

    # 7. 运行评估
    evaluator = CAPTURE()
    # compute_score 会调用 process_samples_multiprocessing
    # 而 process_samples_multiprocessing 会调用我们伪造的 Pool
    result = evaluator.compute_score(refs, preds, return_parse_results=True)
        
    if len(result) == 2:
        _, per_sample_scores = result
    else:
        _, per_sample_scores, _ = result

    # 8. 保存结果
    output_scores = {}
    indices = list(refs.keys())
    for idx, score in zip(indices, per_sample_scores):
        output_scores[idx] = float(score)
    
    print(f"[GPU {gpu_id}] 计算完成，保存结果到 {temp_output_path}")
    with open(temp_output_path, 'w', encoding='utf-8') as f:
        json.dump(output_scores, f)

# ==========================================
# Controller 逻辑 (主进程运行的部分)
# ==========================================

def run_controller(input_dir: str, num_gpus: int, max_samples: int = None, target_indices: List[int] = None):
    """
    主控模式入口。
    
    Args:
        input_dir: 输入目录
        num_gpus: GPU数量
        max_samples: 最大样本数（用于测试）
        target_indices: 要评估的特定索引列表，如果为None则评估所有文件
    """
    directory_path = Path(input_dir)
    # 查找所有 json 文件
    all_json_files = sorted([str(p) for p in directory_path.glob("*.json")])
    
    # 如果指定了目标索引，筛选对应的文件
    if target_indices is not None:
        target_indices_set = set(target_indices)
        json_files = []
        for json_file in all_json_files:
            try:
                # 尝试从文件内容读取index
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    idx = data.get('index')
                    if idx is None:
                        # 如果文件内容没有index，尝试从文件名提取
                        idx = int(Path(json_file).stem)
                    if idx in target_indices_set:
                        json_files.append(json_file)
            except (ValueError, json.JSONDecodeError, KeyError):
                # 如果无法读取，尝试从文件名提取
                try:
                    idx = int(Path(json_file).stem)
                    if idx in target_indices_set:
                        json_files.append(json_file)
                except ValueError:
                    continue
        print(f"[Controller] 指定了 {len(target_indices)} 个目标索引，找到 {len(json_files)} 个匹配的文件")
    else:
        json_files = all_json_files
    
    if max_samples:
        json_files = json_files[:max_samples]
        print(f"[Controller] 测试模式: 仅处理前 {max_samples} 个文件")

    total_files = len(json_files)
    if total_files == 0:
        print("没有找到 JSON 文件")
        return

    # 1. 切分任务
    print(f"[Controller] 将 {total_files} 个文件分配给 {num_gpus} 个 GPU...")
    chunk_size = math.ceil(total_files / num_gpus)
    chunks = [json_files[i:i + chunk_size] for i in range(0, total_files, chunk_size)]
    while len(chunks) < num_gpus: chunks.append([])

    # 2. 创建临时目录
    temp_dir = Path("./temp_capture_tasks")
    temp_dir.mkdir(exist_ok=True)
    
    workers = []
    current_script = sys.argv[0]
    
    # 3. 启动子进程
    for rank in range(num_gpus):
        if not chunks[rank]: continue
        
        task_file = temp_dir / f"task_gpu_{rank}.json"
        result_file = temp_dir / f"result_gpu_{rank}.json"
        
        with open(task_file, 'w', encoding='utf-8') as f:
            json.dump(chunks[rank], f)
        
        cmd = [
            sys.executable, current_script,
            '--worker',
            '--gpu', str(rank),
            '--task_file', str(task_file),
            '--result_file', str(result_file)
        ]
        
        print(f"[Controller] 启动 GPU {rank}...")
        p = subprocess.Popen(cmd)
        workers.append(p)

    # 4. 等待完成
    print("[Controller] 等待所有 Worker 完成...")
    exit_codes = [p.wait() for p in workers]
    
    if any(code != 0 for code in exit_codes):
        print("警告: 有部分 Worker 异常退出，请检查上方报错信息。")
        
    # 5. 收集结果
    print("[Controller] 正在合并结果并写入原文件...")
    updated_count = 0
    new_scores = []  # 本次新评估的分数

    for rank in range(num_gpus):
        result_file = temp_dir / f"result_gpu_{rank}.json"
        if not result_file.exists(): continue
        
        try:
            with open(result_file, 'r') as f:
                scores = json.load(f) # {index_str: score}
            
            # 重新建立 filepath 映射
            chunk_files = chunks[rank]
            file_map = {}
            for fp in chunk_files:
                try:
                    with open(fp, 'r') as f: d = json.load(f)
                    idx = d.get('index', int(Path(fp).stem))
                    file_map[str(idx)] = fp
                except: pass
            
            # 更新
            for idx_str, score in scores.items():
                if idx_str in file_map:
                    fp = file_map[idx_str]
                    try:
                        with open(fp, 'r', encoding='utf-8') as f: d = json.load(f)
                        d['CAPTURE_score'] = score
                        with open(fp, 'w', encoding='utf-8') as f: 
                            json.dump(d, f, ensure_ascii=False, indent=2)
                        updated_count += 1
                        new_scores.append(score)
                    except: pass
        except Exception as e:
            print(f"读取结果文件 {result_file} 失败: {e}")

    # 6. 重新计算所有文件的总平均分
    print("[Controller] 正在重新计算所有文件的总平均分...")
    all_scores = []
    all_json_files = sorted([p for p in directory_path.glob("*.json")])
    
    for json_file in all_json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'CAPTURE_score' in data:
                score = data['CAPTURE_score']
                if isinstance(score, (int, float)):
                    all_scores.append(float(score))
        except Exception as e:
                continue

    # 清理临时目录 (根据需要取消注释)
    # import shutil
    # shutil.rmtree(temp_dir)
    
    if new_scores:
        new_avg = sum(new_scores) / len(new_scores)
        print(f"\n[完成] 成功更新 {updated_count} 个文件。")
        print(f"本次评估的平均 CAPTURE 分数: {new_avg:.6f}")
    
    if all_scores:
        total_avg = sum(all_scores) / len(all_scores)
        print(f"所有文件的总平均 CAPTURE 分数: {total_avg:.6f} (共 {len(all_scores)} 个文件)")
        
        summary_path = Path(input_dir) / "CAPTURE_summary_subprocess.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Average: {total_avg:.6f}\nCount: {len(all_scores)}\n")
            if new_scores:
                f.write(f"New Evaluated: {len(new_scores)}\n")
                f.write(f"New Average: {new_avg:.6f}\n")

# ==========================================
# 主入口
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 这里的默认路径根据你的实际路径修改
    parser.add_argument("--dir", type=str, default='/data0/swz/exp/DetailCaps/result/llada_vgr_0105_mask0')
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None, help="测试时限制样本数")
    parser.add_argument("--indices", type=str, default=None, 
                       help="要评估的样本索引，可以是逗号分隔的字符串（如 '0,1,2'）或文件路径（每行一个索引）")
    
    parser.add_argument("--worker", action='store_true', help="Internal use only")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--task_file", type=str)
    parser.add_argument("--result_file", type=str)
    
    args = parser.parse_args()
    
    if args.worker:
        run_worker(args.gpu, args.task_file, args.result_file)
    else:
        # 解析 indices 参数
        target_indices = None
        if args.indices:
            if Path(args.indices).exists():
                # 从文件读取索引
                with open(args.indices, 'r') as f:
                    target_indices = [int(line.strip()) for line in f if line.strip().isdigit()]
                print(f"[Controller] 从文件读取 {len(target_indices)} 个索引: {args.indices}")
            else:
                # 从逗号分隔的字符串解析
                try:
                    target_indices = [int(x.strip()) for x in args.indices.split(',') if x.strip().isdigit()]
                    print(f"[Controller] 从参数解析 {len(target_indices)} 个索引")
                except ValueError:
                    print(f"[Controller] 警告: 无法解析索引参数 '{args.indices}'，将评估所有文件")
                    target_indices = None
        
        run_controller(args.dir, args.num_gpus, args.max_samples, target_indices)