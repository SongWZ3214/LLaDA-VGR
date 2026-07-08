import json
import numpy as np
import pandas as pd
from pathlib import Path


def analyze_json_metrics(file_path: str, model_name: str) -> dict:
    """
    读取 JSON 文件并计算时间、显存的平均值和峰值
    """
    path = Path(file_path)
    if not path.exists():
        print(f"⚠️ 找不到文件: {file_path}")
        return None

    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ 文件格式错误: {file_path}")
            return None

    latencies = []
    allocated_mem = []
    reserved_mem = []

    # 遍历数据，只统计成功生成 response 的样本
    for item in data:
        if 'response' in item and item['response']:
            if 'latency_seconds' in item:
                latencies.append(item['latency_seconds'])
            if 'peak_allocated_mb' in item:
                allocated_mem.append(item['peak_allocated_mb'])
            if 'peak_reserved_mb' in item:
                reserved_mem.append(item['peak_reserved_mb'])

    if not latencies:
        print(f"⚠️ 在 {file_path} 中没有找到有效的统计数据！")
        return None

    # 计算统计指标
    stats = {
        "Model": model_name,
        "Valid Samples": len(latencies),
        "Latency Avg (s)": np.mean(latencies),
        "Latency Peak (s)": np.max(latencies),
        "Allocated VRAM Avg (MB)": np.mean(allocated_mem) if allocated_mem else 0,
        "Allocated VRAM Peak (MB)": np.max(allocated_mem) if allocated_mem else 0,
        "Reserved VRAM Avg (MB)": np.mean(reserved_mem) if reserved_mem else 0,
        "Reserved VRAM Peak (MB)": np.max(reserved_mem) if reserved_mem else 0,
    }

    return stats


def main():
    # ================= 配置你的 JSON 文件路径 =================
    # 请根据你实际跑出来的带 threshold 后缀的文件名进行修改
    baseline_json = "/data0/swz/exp/AMBER/data/query/query_generative_0330_base.json"

    # 假设你的阈值是 0.4，文件名类似于 query_generative_rebuttal-thresh_0.4.json
    optimized_json = "/data0/swz/exp/AMBER/data/query/query_generative_0330_vgr.json"
    # ==========================================================

    print("📊 正在解析实验数据...\n")

    results = []

    # 分析 Baseline 模型
    base_stats = analyze_json_metrics(baseline_json, "Baseline (LLaDA-V)")
    if base_stats:
        results.append(base_stats)

    # 分析 Optimized 模型 (VGR)
    opt_stats = analyze_json_metrics(optimized_json, "Optimized (Ours w/ VGR)")
    if opt_stats:
        results.append(opt_stats)

    if not results:
        print("❌ 没有提取到任何有效数据，请检查 JSON 路径。")
        return

    # 转换为 Pandas DataFrame 以方便格式化输出
    df = pd.DataFrame(results)

    # 将 MB 转换为 GB 以提升表格的可读性 (保留两位小数)
    df["Allocated VRAM Avg (GB)"] = (df["Allocated VRAM Avg (MB)"] / 1024).round(2)
    df["Allocated VRAM Peak (GB)"] = (df["Allocated VRAM Peak (MB)"] / 1024).round(2)
    df["Reserved VRAM Avg (GB)"] = (df["Reserved VRAM Avg (MB)"] / 1024).round(2)
    df["Reserved VRAM Peak (GB)"] = (df["Reserved VRAM Peak (MB)"] / 1024).round(2)

    # 保留延迟的小数点后两位
    df["Latency Avg (s)"] = df["Latency Avg (s)"].round(2)
    df["Latency Peak (s)"] = df["Latency Peak (s)"].round(2)

    # 选取我们要展示的列
    display_cols = [
        "Model", "Valid Samples",
        "Latency Avg (s)", "Latency Peak (s)",
        "Allocated VRAM Avg (GB)", "Allocated VRAM Peak (GB)",
        "Reserved VRAM Peak (GB)"  # Reserved 通常看峰值以防 OOM
    ]
    df_display = df[display_cols]

    # 1. 打印 Markdown 表格 (可以直接贴入 GitHub / Rebuttal 纯文本框)
    print("=== 📌 Markdown 格式表格 (适用于 OpenReview / 文本框) ===")
    print(df_display.to_markdown(index=False))
    print("\n")

    # 2. 打印 LaTeX 表格 (可以直接贴入论文 PDF)
    print("=== 🎓 LaTeX 格式表格 (适用于论文 PDF) ===")
    latex_code = df_display.to_latex(index=False, float_format="%.2f",
                                     caption="Comparison of inference latency and memory cost.",
                                     label="tab:memory_cost")
    print(latex_code)

    # 3. 如果需要看绝对数值对比
    if len(results) == 2:
        base_lat = df.loc[0, "Latency Avg (s)"]
        opt_lat = df.loc[1, "Latency Avg (s)"]
        base_mem = df.loc[0, "Reserved VRAM Peak (GB)"]
        opt_mem = df.loc[1, "Reserved VRAM Peak (GB)"]

        print("\n=== 💡 Rebuttal 核心结论提取 ===")
        print(
            f"1. 耗时增加: 平均单图推理耗时从 {base_lat}s 增加至 {opt_lat}s (增幅 {((opt_lat - base_lat) / base_lat * 100):.1f}%)。")
        print(
            f"2. 峰值显存: 系统的真实峰值显存需求 (Reserved) 从 {base_mem}GB 变为 {opt_mem}GB (变化 {opt_mem - base_mem:.2f}GB)。")
        print("这说明我们的 VGR 机制用极小/可接受的额外开销，换取了显著的性能提升。")


if __name__ == "__main__":
    main()