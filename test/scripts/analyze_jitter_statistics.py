import json
from pathlib import Path
from typing import List, Dict, Any

def load_all_json_files(directory: str) -> List[Dict[str, Any]]:
    """
    加载目录下所有 JSON 文件
    
    Returns:
        List of data dictionaries
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
            files_data.append(data)
        except Exception as e:
            print(f"警告: 读取文件 {filepath} 时出错: {e}")
            continue
    
    print(f"成功加载 {len(files_data)} 个文件")
    return files_data

def analyze_jitter_statistics(input_directory: str, output_file: str):
    """
    分析 Jitter 统计信息
    
    Args:
        input_directory: 输入目录路径
        output_file: 输出文件路径
    """
    # 加载所有 JSON 文件
    print("正在加载所有 JSON 文件...")
    files_data = load_all_json_files(input_directory)
    
    if not files_data:
        print("错误: 没有找到任何有效的 JSON 文件")
        return
    
    # 统计变量
    total_samples = 0
    inconsistent_greater_count = 0
    fine_grained_greater_count = 0
    marked_greater_count = 0
    
    # 详细统计（用于输出）
    detailed_stats = []
    
    # 处理每个文件
    for data in files_data:
        statistics = data.get('statistics', {})
        
        # 检查是否有必要的字段
        if 'all_tokens_mean' not in statistics:
            continue
        
        all_tokens_mean = statistics.get('all_tokens_mean', 0.0)
        inconsistent_tokens_mean = statistics.get('inconsistent_tokens_mean', 0.0)
        fine_grained_tokens_mean = statistics.get('fine_grained_tokens_mean', 0.0)
        marked_tokens_mean = statistics.get('marked_tokens_mean', 0.0)
        
        # 跳过无效数据（如果 all_tokens_mean 为 0 或不存在，可能数据有问题）
        if all_tokens_mean == 0.0:
            continue
        
        total_samples += 1
        
        # 检查各种均值是否大于 all_tokens_mean
        inconsistent_greater = inconsistent_tokens_mean > all_tokens_mean
        fine_grained_greater = fine_grained_tokens_mean > all_tokens_mean
        marked_greater = marked_tokens_mean > all_tokens_mean
        
        if inconsistent_greater:
            inconsistent_greater_count += 1
        if fine_grained_greater:
            fine_grained_greater_count += 1
        if marked_greater:
            marked_greater_count += 1
        
        # 保存详细信息
        sample_index = data.get('index', 'unknown')
        detailed_stats.append({
            'index': sample_index,
            'all_tokens_mean': all_tokens_mean,
            'inconsistent_tokens_mean': inconsistent_tokens_mean,
            'fine_grained_tokens_mean': fine_grained_tokens_mean,
            'marked_tokens_mean': marked_tokens_mean,
            'inconsistent_greater': inconsistent_greater,
            'fine_grained_greater': fine_grained_greater,
            'marked_greater': marked_greater
        })
    
    # 计算比例
    if total_samples == 0:
        print("错误: 没有有效的样本数据")
        return
    
    inconsistent_ratio = inconsistent_greater_count / total_samples * 100
    fine_grained_ratio = fine_grained_greater_count / total_samples * 100
    marked_ratio = marked_greater_count / total_samples * 100
    
    # 准备输出内容
    output_content = []
    output_content.append("=" * 80)
    output_content.append("Jitter 统计信息分析报告")
    output_content.append("=" * 80)
    output_content.append("")
    output_content.append(f"总样本数: {total_samples}")
    output_content.append("")
    output_content.append("-" * 80)
    output_content.append("统计结果:")
    output_content.append("-" * 80)
    output_content.append("")
    output_content.append(f"1. inconsistent_tokens_mean > all_tokens_mean:")
    output_content.append(f"   数量: {inconsistent_greater_count} / {total_samples}")
    output_content.append(f"   比例: {inconsistent_ratio:.2f}%")
    output_content.append("")
    output_content.append(f"2. fine_grained_tokens_mean > all_tokens_mean:")
    output_content.append(f"   数量: {fine_grained_greater_count} / {total_samples}")
    output_content.append(f"   比例: {fine_grained_ratio:.2f}%")
    output_content.append("")
    output_content.append(f"3. marked_tokens_mean > all_tokens_mean:")
    output_content.append(f"   数量: {marked_greater_count} / {total_samples}")
    output_content.append(f"   比例: {marked_ratio:.2f}%")
    output_content.append("")
    output_content.append("=" * 80)
    output_content.append("")
    
    # 保存到文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_content))
    
    # 同时保存 JSON 格式的详细统计
    json_output_path = output_path.with_suffix('.json')
    json_output = {
        'summary': {
            'total_samples': total_samples,
            'inconsistent_greater': {
                'count': inconsistent_greater_count,
                'ratio': inconsistent_ratio
            },
            'fine_grained_greater': {
                'count': fine_grained_greater_count,
                'ratio': fine_grained_ratio
            },
            'marked_greater': {
                'count': marked_greater_count,
                'ratio': marked_ratio
            }
        },
        'detailed_stats': detailed_stats
    }
    
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    
    # 打印结果
    print("\n" + "=" * 80)
    print("统计结果:")
    print("=" * 80)
    print(f"总样本数: {total_samples}")
    print()
    print(f"1. inconsistent_tokens_mean > all_tokens_mean:")
    print(f"   数量: {inconsistent_greater_count} / {total_samples}")
    print(f"   比例: {inconsistent_ratio:.2f}%")
    print()
    print(f"2. fine_grained_tokens_mean > all_tokens_mean:")
    print(f"   数量: {fine_grained_greater_count} / {total_samples}")
    print(f"   比例: {fine_grained_ratio:.2f}%")
    print()
    print(f"3. marked_tokens_mean > all_tokens_mean:")
    print(f"   数量: {marked_greater_count} / {total_samples}")
    print(f"   比例: {marked_ratio:.2f}%")
    print()
    print(f"结果已保存到: {output_path}")
    print(f"详细统计（JSON格式）已保存到: {json_output_path}")

def main():
    """主函数"""
    input_directory = '/data2/swz/LLaDA-VGR/test/result/jitter_analysis'
    output_file = '/data2/swz/LLaDA-VGR/test/results/jitter_analysis_statistics.txt'
    
    try:
        analyze_jitter_statistics(input_directory, output_file)
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

