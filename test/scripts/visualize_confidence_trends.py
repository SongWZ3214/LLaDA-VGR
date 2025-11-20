import json
import numpy as np
from pathlib import Path

# 尝试导入matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    from matplotlib import cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib未安装，将无法生成图表。请运行: pip install matplotlib")

# 尝试导入seaborn（可选）
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# 设置中文字体（如果matplotlib可用）
if HAS_MATPLOTLIB:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

def load_all_json_files(directory: str) -> list:
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

def load_data_from_sample(sample: dict) -> dict:
    """从样本字典中加载数据（用于验证）"""
    print(f"样本索引: {sample.get('index')}")
    print(f"生成文本长度: {len(sample.get('generated_text', ''))}")
    
    history = sample.get('intermediate_confidence_history', [])
    print(f"中间历史步骤数: {len(history)}")
    
    if not history:
        raise ValueError("没有中间置信度历史数据")
    
    return sample

def prepare_confidence_matrix(sample: dict) -> tuple:
    """准备置信度矩阵"""
    history = sample.get('intermediate_confidence_history', [])
    if not history:
        return None, None, None
    
    # 获取token数量（从第一步获取）
    first_step = history[0]
    num_tokens = len(first_step.get('tokens', []))
    num_steps = len(history)
    
    print(f"\nToken数量: {num_tokens}, 步骤数: {num_steps}")
    
    # 创建置信度矩阵: [num_tokens, num_steps]
    confidence_matrix = np.zeros((num_tokens, num_steps))
    state_matrix = []  # 记录每个token在每个步骤的状态
    
    # 填充矩阵
    for step_idx, step_data in enumerate(history):
        confidences = step_data.get('confidences', [])
        token_states = step_data.get('token_states', [])
        tokens = step_data.get('tokens', [])
        
        # 确保长度一致
        min_len = min(len(confidences), len(token_states), len(tokens), num_tokens)
        
        for token_idx in range(min_len):
            confidence_matrix[token_idx, step_idx] = confidences[token_idx]
        
        # 记录状态
        if step_idx == 0:
            state_matrix = [[state] for state in token_states[:num_tokens]]
        else:
            for token_idx in range(min(min_len, num_tokens)):
                if token_idx < len(state_matrix):
                    state_matrix[token_idx].append(token_states[token_idx])
    
    return confidence_matrix, state_matrix, history

def plot_confidence_heatmap(confidence_matrix: np.ndarray, output_file: str, sample: dict):
    """绘制置信度热力图"""
    print(f"\n正在绘制置信度热力图...")
    
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # 绘制热力图
    im = ax.imshow(confidence_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
    
    # 设置标签
    num_tokens, num_steps = confidence_matrix.shape
    ax.set_xlabel('Step (步骤)', fontsize=12)
    ax.set_ylabel('Token Position (Token位置)', fontsize=12)
    ax.set_title('Token Confidence Evolution Over Steps\n(每个Token的置信度随步骤变化)', fontsize=14, fontweight='bold')
    
    # 设置刻度
    step_interval = max(1, num_steps // 20)
    ax.set_xticks(range(0, num_steps, step_interval))
    ax.set_xticklabels(range(0, num_steps, step_interval))
    
    token_interval = max(1, num_tokens // 20)
    ax.set_yticks(range(0, num_tokens, token_interval))
    ax.set_yticklabels(range(0, num_tokens, token_interval))
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Confidence (置信度)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"热力图已保存到: {output_file}")
    plt.close()

def plot_individual_token_trends(confidence_matrix: np.ndarray, state_matrix: list, output_file: str, sample: dict, history: list):
    """绘制所有token的单个置信度趋势"""
    print(f"\n正在绘制所有token的置信度趋势...")
    
    num_tokens, num_steps = confidence_matrix.shape
    
    # 获取最后一步的token文本（最终生成的词）
    last_step = history[-1] if history else {}
    token_texts = last_step.get('token_texts', [])
    
    # 如果token_texts不够，尝试从sample的token_details获取
    if len(token_texts) < num_tokens:
        token_details = sample.get('token_details', [])
        if len(token_details) >= num_tokens:
            token_texts = [td.get('token', f'Token {i}') for i, td in enumerate(token_details[:num_tokens])]
        else:
            # 如果还是不够，使用索引
            token_texts = [f'Token {i}' for i in range(num_tokens)]
    
    # 确保token_texts长度足够
    while len(token_texts) < num_tokens:
        token_texts.append(f'Token {len(token_texts)}')
    
    # 计算子图布局
    cols = 5
    rows = (num_tokens + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    steps = np.arange(num_steps)
    
    for token_idx in range(num_tokens):
        ax = axes[token_idx]
        
        # 获取该token的置信度序列
        confidences = confidence_matrix[token_idx, :]
        
        # 获取该token的状态序列
        states = state_matrix[token_idx] if token_idx < len(state_matrix) else []
        
        # 获取该token的最终文本
        token_text = token_texts[token_idx] if token_idx < len(token_texts) else f'Token {token_idx}'
        # 清理文本，移除特殊字符，限制长度
        token_text = token_text.replace('<MASK>', 'MASK').replace('<|eot_id|>', 'EOT')
        if len(token_text) > 20:
            token_text = token_text[:17] + '...'
        
        # 绘制置信度曲线
        ax.plot(steps, confidences, linewidth=1.5, alpha=0.7)
        
        # 标记mask和decoded的区域
        if states:
            for step_idx, state in enumerate(states):
                if step_idx < len(confidences):
                    if state == 'mask':
                        ax.scatter(step_idx, confidences[step_idx], c='red', s=10, alpha=0.5, marker='x')
                    elif state == 'decoded':
                        ax.scatter(step_idx, confidences[step_idx], c='green', s=10, alpha=0.5, marker='o')
        
        # 设置标签
        ax.set_xlabel('Step', fontsize=8)
        ax.set_ylabel('Confidence', fontsize=8)
        ax.set_title(token_text, fontsize=9)  # 使用token文本而不是"Token n"
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(num_tokens, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Individual Token Confidence Trends\n(单个Token的置信度趋势)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"所有token趋势图已保存到: {output_file}")
    plt.close()

def plot_selected_tokens(confidence_matrix: np.ndarray, state_matrix: list, output_file: str, sample: dict, token_indices: list = None):
    """绘制选定的token的详细趋势"""
    print(f"\n正在绘制选定token的详细趋势...")
    
    num_tokens, num_steps = confidence_matrix.shape
    steps = np.arange(num_steps)
    
    # 如果没有指定token，选择一些代表性的token
    if token_indices is None:
        # 选择前10个、中间10个、后10个
        token_indices = list(range(min(10, num_tokens))) + \
                       list(range(num_tokens // 2, min(num_tokens // 2 + 10, num_tokens))) + \
                       list(range(max(0, num_tokens - 10), num_tokens))
        token_indices = sorted(list(set(token_indices)))[:30]  # 最多30个
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 使用不同颜色绘制不同token
    colors = cm.get_cmap('tab20', len(token_indices))
    
    for idx, token_idx in enumerate(token_indices):
        if token_idx >= num_tokens:
            continue
        
        confidences = confidence_matrix[token_idx, :]
        states = state_matrix[token_idx] if token_idx < len(state_matrix) else []
        
        # 绘制置信度曲线
        ax.plot(steps, confidences, linewidth=1.5, alpha=0.7, 
                label=f'Token {token_idx}', color=colors(idx))
        
        # 标记状态变化点
        if states:
            for step_idx, state in enumerate(states):
                if step_idx < len(confidences):
                    if state == 'mask' and step_idx > 0 and states[step_idx-1] == 'decoded':
                        # mask -> decoded 的转换点
                        ax.scatter(step_idx, confidences[step_idx], c='green', s=50, 
                                 marker='o', zorder=5, edgecolors='black', linewidths=1)
                    elif state == 'decoded' and step_idx > 0 and states[step_idx-1] == 'mask':
                        # decoded -> mask 的转换点（不太可能，但保留）
                        ax.scatter(step_idx, confidences[step_idx], c='red', s=50, 
                                 marker='x', zorder=5, edgecolors='black', linewidths=1)
    
    ax.set_xlabel('Step (步骤)', fontsize=12)
    ax.set_ylabel('Confidence (置信度)', fontsize=12)
    ax.set_title('Selected Token Confidence Trends\n(选定Token的置信度趋势)', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"选定token趋势图已保存到: {output_file}")
    plt.close()

def process_single_sample(sample: dict, output_dir: str):
    """处理单个样本，生成individual_token_trends.png"""
    try:
        # 验证数据
        load_data_from_sample(sample)
        
        # 准备置信度矩阵
        confidence_matrix, state_matrix, history = prepare_confidence_matrix(sample)
        
        if confidence_matrix is None:
            print(f"  跳过: 无法准备置信度矩阵")
            return False
        
        # 生成individual_token_trends.png
        sample_index = sample.get('index', 0)
        output_file = Path(output_dir) / f"{sample_index:06d}_individual_token_trends.png"
        
        plot_individual_token_trends(
            confidence_matrix,
            state_matrix,
            str(output_file),
            sample,
            history
        )
        
        return True
    except Exception as e:
        print(f"  处理出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    if not HAS_MATPLOTLIB:
        print("错误: matplotlib未安装，无法生成可视化图表")
        print("请运行: pip install matplotlib")
        return
    
    # 配置参数
    input_directory = '/data2/swz/LLaDA-VGR/test/result/detailcaps_outputs'
    output_dir = '/data2/swz/LLaDA-VGR/test/result/confidence_visualization_batch'
    
    # 控制处理数据的数量，用于测试
    # 设置为 None 表示处理所有数据，设置为数字表示只处理前 N 个样本
    MAX_SAMPLES = 10  # 测试时设置为 10，正式运行时设置为 None
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 加载所有 JSON 文件
    print("\n" + "="*60)
    print("正在加载所有 JSON 文件...")
    print("="*60)
    files_data = load_all_json_files(input_directory)
    
    if not files_data:
        print("错误: 没有找到任何有效的 JSON 文件")
        return
    
    # 限制处理数量
    if MAX_SAMPLES is not None:
        files_data = files_data[:MAX_SAMPLES]
        print(f"测试模式: 只处理前 {MAX_SAMPLES} 个样本")
    
    # 统计信息
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # 处理每个文件
    total = len(files_data)
    print("\n" + "="*60)
    print("开始生成可视化图表...")
    print("="*60)
    
    for idx, (sample_index, data, filepath) in enumerate(files_data):
        print(f"\n[{idx+1}/{total}] 正在处理样本 {sample_index}...")
        
        # 检查是否有中间置信度历史
        if 'intermediate_confidence_history' not in data:
            print(f"  跳过: 文件没有 intermediate_confidence_history")
            skipped_count += 1
            continue
        
        # 处理样本
        success = process_single_sample(data, output_dir)
        if success:
            processed_count += 1
        else:
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
    print(f"\n所有结果已保存到: {output_dir}")
    print(f"每个样本的图表文件名格式: {{index:06d}}_individual_token_trends.png")

if __name__ == "__main__":
    main()

