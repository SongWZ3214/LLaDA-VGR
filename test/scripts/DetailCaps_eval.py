import json
from pathlib import Path
from capture_metric.capture import CAPTURE
from typing import Dict, List, Tuple

def load_all_json_files(directory: str) -> List[Tuple[int, dict, str]]:
    """
    加载目录下所有 JSON 文件。
    
    Returns:
        List of (index, data_dict, filepath) tuples, sorted by index
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    
    files_data = []
    json_files = sorted(directory_path.glob("*.json"))
    
    print(f"找到 {len(json_files)} 个 JSON 文件")
    
    for filepath in json_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            index = data.get('index')
            if index is None:
                # 尝试从文件名提取索引
                try:
                    index = int(filepath.stem)
                except ValueError:
                    print(f"警告: 文件 {filepath} 没有有效的索引，跳过")
                    continue
            
            files_data.append((index, data, str(filepath)))
        except Exception as e:
            print(f"警告: 读取文件 {filepath} 时出错: {e}")
            continue
    
    # 按索引排序
    files_data.sort(key=lambda x: x[0])
    print(f"成功加载 {len(files_data)} 个文件")
    
    return files_data

def prepare_evaluation_data(files_data: List[Tuple[int, dict, str]]) -> Tuple[Dict, Dict, List[Tuple[int, str]]]:
    """
    根据官方格式要求，准备 refs 和 preds 字典。
    
    Returns:
        (refs, preds, file_mapping): refs和preds字典，以及(index, filepath)映射列表
    """
    refs = {}
    preds = {}
    file_mapping = []
        
    for index, data, filepath in files_data:
        # 检查是否有 token_details（可选，用于验证数据完整性）
        td = data.get("token_details", [])
        if not td and 'generated_text' not in data:
            print(f"警告: 文件 {filepath} (index={index}) 没有有效数据，跳过")
            continue
        
        # 准备 Ground Truth (refs) 字典
        # refs 的 value 必须是一个列表
        # 根据数据格式，Ground Truth 是 GT_Caption_GPT4O, GT_Caption_GPT4V, 和 GT_Caption_Gemini15Pro
        gt_captions = [
            data.get('GT_Caption_GPT4O'),
            data.get('GT_Caption_GPT4V'),
            data.get('GT_Caption_Gemini15Pro')
        ]
        
        # 过滤掉可能存在的 None 值
        valid_gt = [cap for cap in gt_captions if cap]
        if not valid_gt:
            print(f"警告: 文件 {filepath} (index={index}) 没有有效的 Ground Truth，跳过")
            continue
        
        refs[index] = valid_gt
        
        # 准备 Prediction (preds) 字典
        # preds 的 value 必须是一个列表，只包含一个预测描述
        generated_text = data.get('generated_text')
        if not generated_text:
            print(f"警告: 文件 {filepath} (index={index}) 没有生成的文本，跳过")
            continue
        
        preds[index] = [generated_text]
        file_mapping.append((index, filepath))

    if not refs or not preds:
        raise ValueError("未能成功准备任何评估数据，请检查您的文件路径和数据格式。")
        
    print(f"成功为 {len(refs)} 个样本准备了评估数据。")
    return refs, preds, file_mapping


def main():
    input_directory = '/data2/swz/LLaDA-VGR/test/result/detailcaps_outputs'
    output_directory = input_directory  # 结果写回原目录
    
    # 控制处理数据的数量，用于测试
    # 设置为 None 表示处理所有数据，设置为数字表示只处理前 N 个样本
    MAX_SAMPLES = 1  # 测试时设置为 10，正式运行时设置为 None
        
    try:
        # --- 1. 加载所有 JSON 文件 ---
        print("正在加载所有 JSON 文件...")
        files_data = load_all_json_files(input_directory)
        
        if not files_data:
            print("错误: 没有找到任何有效的 JSON 文件")
            return
        
        # 限制处理数量（如果指定）
        if MAX_SAMPLES is not None:
            files_data = files_data[:MAX_SAMPLES]
            print(f"测试模式: 只处理前 {MAX_SAMPLES} 个样本")
        
        # --- 2. 准备评估数据 ---
        print("正在准备 refs 和 preds 字典...")
        refs, preds, file_mapping = prepare_evaluation_data(files_data)
        
        if not refs or not preds:
            print("错误: 未能准备有效的评估数据")
            return
        
        # --- 3. 计算 CAPTURE 分数 ---
        print("正在初始化 CAPTURE 评估器...")
        evaluator = CAPTURE()
        
        print(f"正在计算 {len(refs)} 个样本的分数... (这可能需要一些时间，因为需要加载T5和BERT模型)")
        # CAPTURE.compute_score 返回 (overall_score, per_sample_scores) 或 (overall_score, per_sample_scores, parse_results)
        result = evaluator.compute_score(refs, preds, return_parse_results=True)
        
        # 解包返回结果
        if len(result) == 2:
            overall_score, per_sample_scores = result
            parse_results = None
        elif len(result) == 3:
            overall_score, per_sample_scores, parse_results = result
            print(f"parse_results: {parse_results}")
        else:
            raise ValueError(f"意外的返回格式: {type(result)}")
        
        print(f"总体 CAPTURE 分数: {overall_score:.6f}")
        print(f"每个样本的分数列表长度: {len(per_sample_scores)}")
        
        # --- 4. 将分数映射到各个样本并写回文件 ---
        # 获取索引列表，顺序与 refs.keys() 一致（与 per_sample_scores 顺序一致）
        indices_list = list(refs.keys())
        
        if len(per_sample_scores) != len(indices_list):
            print(f"警告: 分数列表长度 ({len(per_sample_scores)}) 与样本数 ({len(indices_list)}) 不匹配")
        
        # 创建索引到分数的映射
        index_to_score = {}
        for idx, score in zip(indices_list, per_sample_scores):
            index_to_score[idx] = float(score)
        
        # 创建索引到文件路径的映射
        index_to_file = {idx: filepath for idx, filepath in file_mapping}
        
        # 将分数写回各自文件
        updated_count = 0
        for index, filepath in file_mapping:
            if index in index_to_score:
                try:
                    # 读取原文件
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 添加 CAPTURE 分数
                    data['CAPTURE_score'] = index_to_score[index]
                    # if parse_results is not None:
                    #     data['CAPTURE_parse_results'] = parse_results[index]
                    # 写回文件
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    
                    updated_count += 1
                except Exception as e:
                    print(f"警告: 更新文件 {filepath} 时出错: {e}")
        
        print(f"成功更新 {updated_count} 个文件")
        
        # --- 5. 保存统计摘要 ---
        valid_scores = list(index_to_score.values())
        if valid_scores:
            average_score = sum(valid_scores) / len(valid_scores)
            print(f"\n平均 CAPTURE 分数: {average_score:.6f}")
            print(f"有效样本数: {len(valid_scores)}")
            
            # 保存摘要到 txt 文件
            summary_file = Path(output_directory) / "CAPTURE_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"CAPTURE 评估结果摘要\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"总样本数: {len(files_data)}\n")
                f.write(f"有效评估样本数: {len(valid_scores)}\n")
                f.write(f"总体 CAPTURE 分数: {overall_score:.6f}\n")
                f.write(f"平均 CAPTURE 分数: {average_score:.6f}\n")
                f.write(f"最高分数: {max(valid_scores):.6f}\n")
                f.write(f"最低分数: {min(valid_scores):.6f}\n")
                f.write(f"\n详细分数已写入各个 JSON 文件的 'CAPTURE_score' 字段\n")
            
            print(f"\n评估摘要已保存到: {summary_file}")

        print("\n--- 评估完成 ---")

    except FileNotFoundError as e:
        print(f"错误：找不到文件或目录: {e}")
    except Exception as e:
        print(f"发生错误：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 确保您已经安装了 capture_metric
    # pip3 install capture_metric
    main()