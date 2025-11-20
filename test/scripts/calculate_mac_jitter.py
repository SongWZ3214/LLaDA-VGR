import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re

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

def calculate_jitter_for_token(confidence_sequence: List[float]) -> float:
    """
    计算单个token的波动率（Jitter）- 计算所有步骤
    
    Args:
        confidence_sequence: token在各个步骤的置信度序列
    
    Returns:
        Jitter值
    """
    if len(confidence_sequence) < 2:
        return 0.0
    
    changes = []
    for t in range(1, len(confidence_sequence)):
        change = abs(confidence_sequence[t] - confidence_sequence[t-1])
        changes.append(change)
        
    conf_diff = confidence_sequence[-1] - confidence_sequence[0]
    jitter = sum(changes) - conf_diff
    return jitter

def extract_confidence_matrix(intermediate_history: List[Dict]) -> Tuple[np.ndarray, List[str], List[List[str]]]:
    """
    从中间置信度历史中提取置信度矩阵和token状态历史
    
    Args:
        intermediate_history: intermediate_confidence_history 列表
    
    Returns:
        (confidence_matrix, token_texts, token_states_history): 
        置信度矩阵、token文本列表、每个步骤的token状态列表
    """
    if not intermediate_history:
        return np.array([]), [], []
    
    # 获取最后一个步骤的token文本（最终生成的token）
    last_step = intermediate_history[-1]
    token_texts = last_step.get('token_texts', [])
    num_tokens = len(token_texts)
    num_steps = len(intermediate_history)
    
    if num_tokens == 0 or num_steps == 0:
        return np.array([]), [], []
    
    # 初始化置信度矩阵: [num_tokens, num_steps]
    confidence_matrix = np.zeros((num_tokens, num_steps))
    token_states_history = []
    
    # 填充矩阵
    for step_idx, step_data in enumerate(intermediate_history):
        confidences = step_data.get('confidences', [])
        token_states = step_data.get('token_states', [])
        
        # 确保长度匹配
        if len(confidences) >= num_tokens:
            confidence_matrix[:, step_idx] = confidences[:num_tokens]
        else:
            # 如果长度不匹配，填充0
            confidence_matrix[:len(confidences), step_idx] = confidences
            confidence_matrix[len(confidences):, step_idx] = 0.0
        
        # 保存token状态
        if len(token_states) >= num_tokens:
            token_states_history.append(token_states[:num_tokens])
        else:
            # 如果长度不匹配，填充'mask'
            padded_states = token_states + ['mask'] * (num_tokens - len(token_states))
            token_states_history.append(padded_states)
    
    return confidence_matrix, token_texts, token_states_history

def find_marked_token_indices(
    generated_text: str,
    token_texts: List[str],
    inconsistencies: List[Dict],
    fine_grained_phrases: List[Dict]
) -> Tuple[List[int], List[int]]:
    """
    找到需要标记的token索引
    
    Args:
        generated_text: 生成的文本
        token_texts: token文本列表
        inconsistencies: 不一致列表（标记为红色）
        fine_grained_phrases: 细粒度短语列表（标记为绿色）
    
    Returns:
        (inconsistent_indices, fine_grained_indices): 
        不一致token的索引列表和细粒度token的索引列表
    """
    inconsistent_indices = []
    fine_grained_indices = []
    
    # 清理token文本，移除特殊标记，但保留空格信息
    clean_tokens = []
    for token_text in token_texts:
        # 保留原始token文本（可能包含前导空格）
        clean_token = token_text.replace('<MASK>', '').replace('<|eot_id|>', '').strip()
        # 如果原始token以空格开头，在clean_token前加空格
        if token_text.startswith(' ') and not clean_token.startswith(' '):
            clean_token = ' ' + clean_token
        clean_tokens.append(clean_token)
    
    def find_token_indices_for_text(text_list: List[Dict]) -> List[int]:
        """辅助函数：为给定的文本列表找到对应的token索引"""
        matched_indices = []
        
        for item in text_list:
            original_text = item.get('original_text', '').strip()
            if not original_text:
                continue
            
            # 在生成文本中查找原始文本的位置（精确匹配）
            start_pos = generated_text.find(original_text)
            if start_pos == -1:
                # 如果找不到精确匹配，尝试不区分大小写
                start_pos = generated_text.lower().find(original_text.lower())
                if start_pos == -1:
                    # 如果还是找不到，尝试查找关键短语（至少3个连续单词）
                    words = original_text.split()
                    if len(words) >= 3:
                        # 尝试查找前3个单词和后3个单词
                        phrase_start = ' '.join(words[:3])
                        phrase_end = ' '.join(words[-3:])
                        start_match = generated_text.find(phrase_start)
                        end_match = generated_text.find(phrase_end)
                        if start_match != -1 and end_match != -1 and start_match <= end_match:
                            start_pos = start_match
                            end_pos = end_match + len(phrase_end)
                        else:
                            # 如果还是找不到，跳过这个项
                            continue
                    else:
                        # 如果单词太少，跳过
                        continue
                else:
                    end_pos = start_pos + len(original_text)
            else:
                end_pos = start_pos + len(original_text)
            
            # 通过字符位置找到对应的token索引
            char_pos = 0
            matched_tokens = []
            
            for token_idx, clean_token in enumerate(clean_tokens):
                if not clean_token:  # 跳过空token
                    continue
                    
                # 在生成文本中查找当前token的位置
                token_start = generated_text.find(clean_token, char_pos)
                if token_start == -1:
                    # 如果找不到，尝试不区分大小写
                    token_start = generated_text.lower().find(clean_token.lower(), char_pos)
                    if token_start == -1:
                        # 如果还是找不到，使用估算位置
                        token_start = char_pos
                
                token_end = token_start + len(clean_token)
                
                # 检查token是否与文本有重叠
                # 只有当token的至少50%与文本重叠时才认为匹配
                overlap_start = max(token_start, start_pos)
                overlap_end = min(token_end, end_pos)
                if overlap_start < overlap_end:
                    overlap_length = overlap_end - overlap_start
                    token_length = token_end - token_start
                    # 如果重叠长度超过token长度的50%，则认为匹配
                    if overlap_length >= token_length * 0.5:
                        matched_tokens.append(token_idx)
                
                # 更新字符位置（使用实际找到的位置）
                char_pos = max(char_pos, token_end)
            
            # 如果找到了匹配的token，添加到结果中
            if matched_tokens:
                matched_indices.extend(matched_tokens)
        
        # 去重并排序
        return sorted(list(set(matched_indices)))
    
    # 处理inconsistencies（红色标记）
    if inconsistencies:
        inconsistent_indices = find_token_indices_for_text(inconsistencies)
    
    # 处理fine_grained_phrases（绿色标记）
    if fine_grained_phrases:
        fine_grained_indices = find_token_indices_for_text(fine_grained_phrases)
    
    return inconsistent_indices, fine_grained_indices

def calculate_jitter(
    data: dict
) -> Tuple[Dict, np.ndarray, List[str], List[int], List[int]]:
    """
    计算所有token的Jitter值
    
    Args:
        data: 单个样本的数据字典
    
    Returns:
        (jitter_dict, confidence_matrix, token_texts, inconsistent_indices, fine_grained_indices):
        Jitter字典、置信度矩阵、token文本列表、不一致token索引、细粒度token索引
    """
    intermediate_history = data.get('intermediate_confidence_history', [])
    
    if not intermediate_history:
        return {}, np.array([]), [], [], []
    
    # 提取置信度矩阵和token状态历史
    confidence_matrix, token_texts, token_states_history = extract_confidence_matrix(intermediate_history)
    
    if confidence_matrix.size == 0:
        return {}, np.array([]), [], [], []
    
    # 计算每个token的Jitter值
    jitter_dict = {}
    num_tokens, num_steps = confidence_matrix.shape
    
    for token_idx in range(num_tokens):
        confidence_sequence = confidence_matrix[token_idx, :].tolist()
        jitter_value = calculate_jitter_for_token(confidence_sequence)
        
        jitter_dict[token_idx] = {
            'jitter': float(jitter_value),
            'token_text': token_texts[token_idx] if token_idx < len(token_texts) else f'Token_{token_idx}'
        }
    
    # 找到需要标记的token索引
    generated_text = data.get('generated_text', '')
    inconsistencies = data.get('inconsistencies', [])
    fine_grained_phrases = data.get('fine_grained_phrases', [])
    inconsistent_indices, fine_grained_indices = find_marked_token_indices(
        generated_text, token_texts, inconsistencies, fine_grained_phrases
    )
    
    return jitter_dict, confidence_matrix, token_texts, inconsistent_indices, fine_grained_indices

def plot_jitter_barchart(
    jitter_values: np.ndarray,
    token_texts: List[str],
    inconsistent_indices: List[int],
    fine_grained_indices: List[int],
    output_file: str,
    sample_index: int
):
    """
    绘制Jitter值条形图
    
    Args:
        jitter_values: Jitter值数组
        token_texts: token文本列表
        inconsistent_indices: 不一致token的索引列表（红色）
        fine_grained_indices: 细粒度token的索引列表（绿色）
        output_file: 输出文件路径
        sample_index: 样本索引
    """
    num_tokens = len(jitter_values)
    
    if num_tokens == 0:
        print(f"  警告: 样本 {sample_index} 没有token数据，跳过绘图")
        return
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(max(16, num_tokens * 0.2), 8))
    
    # 准备颜色数组：默认蓝色，不一致的token为红色，细粒度的token为绿色
    colors = []
    for i in range(num_tokens):
        if i in inconsistent_indices:
            colors.append('red')
        elif i in fine_grained_indices:
            colors.append('green')
        else:
            colors.append('blue')
    
    # 创建x轴位置
    x_positions = np.arange(num_tokens)
    
    # 绘制条形图
    bars = ax.bar(x_positions, jitter_values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # 设置标签
    ax.set_xlabel('Token Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Jitter', fontsize=12, fontweight='bold')
    title = f'Jitter Bar Chart (Sample {sample_index})'
    title_parts = []
    if inconsistent_indices:
        title_parts.append(f'{len(inconsistent_indices)} inconsistent')
    if fine_grained_indices:
        title_parts.append(f'{len(fine_grained_indices)} fine-grained')
    if title_parts:
        title += f'\nRed: inconsistent, Green: fine-grained'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 设置x轴标签
    if num_tokens <= 100:
        # 使用token文本作为标签
        display_texts = []
        for text in token_texts:
            clean_text = text.replace('<MASK>', '[MASK]').replace('<|eot_id|>', '[EOT]')
            # 限制长度
            if len(clean_text) > 15:
                clean_text = clean_text[:12] + '...'
            display_texts.append(clean_text)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(display_texts, rotation=45, ha='right', fontsize=7)
    else:
        # 如果token太多，只显示位置索引
        step = max(1, num_tokens // 30)
        ax.set_xticks(x_positions[::step])
        ax.set_xticklabels([f'Pos {i}' for i in range(0, num_tokens, step)], 
                          rotation=45, ha='right', fontsize=7)
    
    # 添加网格
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Normal tokens'),
        Patch(facecolor='red', alpha=0.7, label='Inconsistent tokens'),
        Patch(facecolor='green', alpha=0.7, label='Fine-grained tokens')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  已保存条形图: {output_file}")

def process_all_samples(
    input_directory: str,
    output_directory: str,
    max_samples: Optional[int] = None
):
    """
    处理所有样本，计算Jitter值并生成可视化
    
    Args:
        input_directory: 输入目录路径
        output_directory: 输出目录路径
        max_samples: 最大处理样本数量（None表示处理所有）
    """
    # 创建输出目录
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_path}")
    
    # 加载所有文件
    print("正在加载所有 JSON 文件...")
    files_data = load_all_json_files(input_directory)
    
    if not files_data:
        print("错误: 没有找到任何有效的 JSON 文件")
        return
    
    # 限制处理数量
    if max_samples is not None:
        files_data = files_data[:max_samples]
        print(f"测试模式: 只处理前 {max_samples} 个样本")
    
    # 统计信息
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # 处理每个文件
    total = len(files_data)
    for idx, (sample_index, data, filepath) in enumerate(files_data):
        print(f"\n[{idx+1}/{total}] 正在处理样本 {sample_index}...")
        
        try:
            # 检查是否有中间置信度历史
            if 'intermediate_confidence_history' not in data:
                print(f"  跳过: 文件没有 intermediate_confidence_history")
                skipped_count += 1
                continue
            
            # 计算Jitter值
            jitter_dict, confidence_matrix, token_texts, inconsistent_indices, fine_grained_indices = calculate_jitter(data)
            
            if not jitter_dict:
                print(f"  跳过: 无法计算Jitter值")
                skipped_count += 1
                continue
            
            # 提取Jitter值数组
            num_tokens = len(jitter_dict)
            jitter_values = np.array([jitter_dict[i]['jitter'] for i in range(num_tokens)])
            
            # 构建包含token id和对应词的标记列表
            inconsistent_tokens = [
                {
                    'token_id': idx,
                    'token_text': token_texts[idx] if idx < len(token_texts) else f'Token_{idx}'
                }
                for idx in inconsistent_indices
            ]
            
            fine_grained_tokens = [
                {
                    'token_id': idx,
                    'token_text': token_texts[idx] if idx < len(token_texts) else f'Token_{idx}'
                }
                for idx in fine_grained_indices
            ]
            
            # 计算各种平均值
            all_jitter_mean = float(np.mean(jitter_values))
            
            # 红色词（不一致）的波动平均值
            if inconsistent_indices:
                inconsistent_jitter_values = np.array([jitter_values[idx] for idx in inconsistent_indices])
                inconsistent_jitter_mean = float(np.mean(inconsistent_jitter_values))
            else:
                inconsistent_jitter_mean = 0.0
            
            # 绿色词（细粒度）的波动平均值
            if fine_grained_indices:
                fine_grained_jitter_values = np.array([jitter_values[idx] for idx in fine_grained_indices])
                fine_grained_jitter_mean = float(np.mean(fine_grained_jitter_values))
            else:
                fine_grained_jitter_mean = 0.0
            
            # 红绿总平均值（红色和绿色词的合并平均值）
            marked_indices = inconsistent_indices + fine_grained_indices
            if marked_indices:
                marked_jitter_values = np.array([jitter_values[idx] for idx in marked_indices])
                marked_jitter_mean = float(np.mean(marked_jitter_values))
            else:
                marked_jitter_mean = 0.0
            
            # 保存Jitter数据到JSON文件
            jitter_output_file = output_path / f"{sample_index:06d}_jitter.json"
            jitter_data = {
                'index': sample_index,
                'num_tokens': num_tokens,
                'jitter_values': {str(k): v for k, v in jitter_dict.items()},
                'inconsistent_token': inconsistent_tokens,
                'fine_grained_token': fine_grained_tokens,
                'statistics': {
                    'all_tokens_mean': all_jitter_mean,
                    'inconsistent_tokens_mean': inconsistent_jitter_mean,
                    'fine_grained_tokens_mean': fine_grained_jitter_mean,
                    'marked_tokens_mean': marked_jitter_mean,
                    'max_jitter': float(np.max(jitter_values)),
                    'min_jitter': float(np.min(jitter_values))
                }
            }
            
            with open(jitter_output_file, 'w', encoding='utf-8') as f:
                json.dump(jitter_data, f, ensure_ascii=False, indent=2)
            
            print(f"  已保存Jitter数据: {jitter_output_file}")
            stats = jitter_data['statistics']
            # print(f"  所有词平均Jitter: {stats['all_tokens_mean']:.6f}")
            # print(f"  红色词平均Jitter: {stats['inconsistent_tokens_mean']:.6f}")
            # print(f"  绿色词平均Jitter: {stats['fine_grained_tokens_mean']:.6f}")
            # print(f"  红绿总平均Jitter: {stats['marked_tokens_mean']:.6f}")
            # print(f"  最大Jitter: {stats['max_jitter']:.6f}, 最小Jitter: {stats['min_jitter']:.6f}")
            # if inconsistent_indices:
            #     print(f"  不一致token数量: {len(inconsistent_indices)}")
            # if fine_grained_indices:
            #     print(f"  细粒度token数量: {len(fine_grained_indices)}")
            
            # 绘制条形图
            chart_output_file = output_path / f"{sample_index:06d}_jitter_barchart.png"
            plot_jitter_barchart(
                jitter_values,
                token_texts,
                inconsistent_indices,
                fine_grained_indices,
                str(chart_output_file),
                sample_index
            )
            
            processed_count += 1
            
        except Exception as e:
            print(f"  ✗ 处理出错: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
        
        # 每处理10个文件打印一次进度
        if (idx + 1) % 10 == 0:
            print(f"\n进度: 已处理 {idx + 1}/{total} 个文件 [成功: {processed_count}, 跳过: {skipped_count}, 错误: {error_count}]")
    
    # 打印最终统计
    print("\n" + "="*60)
    print("处理完成")
    print("="*60)
    print(f"总文件数: {total}")
    print(f"成功处理: {processed_count}")
    print(f"跳过: {skipped_count}")
    print(f"错误: {error_count}")
    print(f"\n所有结果已保存到: {output_path}")

def main():
    """主函数"""
    input_directory = '/data2/swz/LLaDA-VGR/test/result/detailcaps_outputs'
    output_directory = '/data2/swz/LLaDA-VGR/test/results/jitter_analysis'
    
    # 控制处理数据的数量，用于测试
    # 设置为 None 表示处理所有数据，设置为数字表示只处理前 N 个样本
    MAX_SAMPLES = None  # 测试时设置为 5，正式运行时设置为 None
    
    try:
        process_all_samples(
            input_directory, 
            output_directory, 
            max_samples=MAX_SAMPLES
        )
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

