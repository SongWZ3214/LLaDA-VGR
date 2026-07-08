#!/bin/bash

# 定义要测试的阈值
THRESHOLDS=(0.4)

for thresh in "${THRESHOLDS[@]}"; do
    echo "========================================="
    echo "Starting experiment with DINO_THRESH=${thresh}"
    echo "========================================="

    # 将阈值导出为环境变量，Python 代码里的 os.environ 就能读到了
    export DINO_THRESH=$thresh

    # 运行你的主测试脚本 (你需要确保主脚本里保存的 JSON 文件名也能加上这个阈值后缀，防止覆盖)
    python /data0/swz/LLaDA-VGR/eval/scripts/AMBER_gen_R.py

    echo "Finished experiment with threshold ${thresh}"
done