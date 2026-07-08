#!/bin/bash

# 定义要测试的 Batch Size 列表
BATCH_SIZES=(1 2 4 8)

# 你的“干净模板”JSON文件路径（请确保这个文件里的 model_answer 都是空的！）
TEMPLATE_JSON="/data0/swz/exp/MMHal-Bench/response_template_20260327_rebuttal.json"

for bs in "${BATCH_SIZES[@]}"; do
    echo "========================================================"
    echo "  [Rebuttal R3] Starting inference with BATCH_SIZE = ${bs}"
    echo "========================================================"

    # 1. 为当前 batch size 准备一个专属的 JSON 文件
    # 比如：response_template_20260327_rebuttal_bs1.json
    TARGET_JSON="${TEMPLATE_JSON%.json}_bs${bs}.json"

    # 将干净的模板复制过去
    cp "$TEMPLATE_JSON" "$TARGET_JSON"
    echo ">> 已创建专属任务文件: $TARGET_JSON"

    # 2. 记录开始时间
    START_TIME=$(date +%s)

    # 3. 将 BATCH_SIZE 和专属的 QUERY_JSON 路径作为环境变量传入 Python 脚本
    BATCH_SIZE=${bs} QUERY_JSON=${TARGET_JSON} python /data0/swz/LLaDA-VGR/eval/scripts/MMhal_gen_R.py

    # 4. 记录结束时间并计算耗时
    END_TIME=$(date +%s)
    ELAPSED_TIME=$(($END_TIME - $START_TIME))
    echo ">> BATCH_SIZE = ${bs} 耗时: ${ELAPSED_TIME} 秒"

    # 等待几秒释放一下显存
    sleep 3
done

echo "所有 Batch Size 测试完毕！"