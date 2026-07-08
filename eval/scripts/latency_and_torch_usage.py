import json
import numpy as np
from pathlib import Path

# 你的输出文件所在目录
OUTPUT_DIR = "/data0/swz/exp/AMBER/data/query/"


def analyze_hardware_metrics():
    # 假设你的文件命名规则是这样的
    thresholds = ["0.0", "0.2", "0.6", "0.8"]

    print(f"{'Threshold':<10} | {'Avg Latency (s)':<18} | {'Avg Peak Memory (MB)':<20} | {'Valid Samples'}")
    print("-" * 75)

    for thresh in thresholds:
        # file_path = Path(OUTPUT_DIR) / f"query_generative_rebuttal-thresh_{thresh}.json"
        file_path = "/data0/swz/exp/AMBER/data/query/query_generative_rebuttal-thresh_0.4.json"


        # if not file_path.exists():
        #     print(f"{thresh:<10} | File not found: {file_path.name}")
        #     continue

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        latencies = []
        memories = []

        for entry in data:
            if 'latency_seconds' in entry and 'peak_memory_mb' in entry:
                latencies.append(entry['latency_seconds'])
                memories.append(entry['peak_memory_mb'])

        if latencies:
            avg_lat = np.mean(latencies)
            avg_mem = np.mean(memories)
            print(f"{thresh:<10} | {avg_lat:<18.2f} | {avg_mem:<20.2f} | {len(latencies)}")
        else:
            print(f"{thresh:<10} | No valid data")


if __name__ == "__main__":
    analyze_hardware_metrics()