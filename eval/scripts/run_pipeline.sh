#!/bin/bash
# 自动化执行 CapArena 和 AMBER 评估流程
# 按顺序执行：CapArena_gen.py -> caparena_auto_eval.py -> AMBER_gen.py -> inference.py

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAPARENA_SCRIPTS_DIR="/data0/swz/LLaDA-VGR/eval/scripts"
CAPARENA_DIR="/data0/swz/exp/CapArena"
AMBER_DIR="/data0/swz/exp/AMBER"

log_info "=========================================="
log_info "开始执行自动化评估流程"
log_info "=========================================="

# 步骤1: 运行 CapArena_gen.py (在 /data0/swz/LLaDA-VGR/eval/scripts)
log_info "步骤 1/4: 运行 CapArena_gen.py"
log_info "----------------------------------------"
log_info "工作目录: $CAPARENA_SCRIPTS_DIR"
if (cd "$CAPARENA_SCRIPTS_DIR" && python3 CapArena_gen.py); then
    log_info "✓ CapArena_gen.py 执行成功"
else
    log_error "✗ CapArena_gen.py 执行失败"
    exit 1
fi

# 步骤2: 运行 caparena_auto_eval.py (在 /data0/swz/exp/CapArena)
log_info ""
log_info "步骤 2/4: 运行 caparena_auto_eval.py"
log_info "----------------------------------------"
CAPARENA_EVAL_SCRIPT="$CAPARENA_DIR/caparena_auto_eval.py"
if [ ! -f "$CAPARENA_EVAL_SCRIPT" ]; then
    log_error "找不到 caparena_auto_eval.py: $CAPARENA_EVAL_SCRIPT"
    exit 1
fi

log_info "工作目录: $CAPARENA_DIR"
if (cd "$CAPARENA_DIR" && python3 caparena_auto_eval.py \
    --test_model LLaDA-VGR-0105-mask0 \
    --result_path /data0/swz/exp/CapArena/data/llada_vgr_0105_mask0.json \
    --imgs_dir /data0/swz/exp/CapArena/data/caparena_auto_docci_600); then
    log_info "✓ caparena_auto_eval.py 执行成功"
else
    log_error "✗ caparena_auto_eval.py 执行失败"
    exit 1
fi

# 步骤3: 运行 AMBER_gen.py (在 /data0/swz/LLaDA-VGR/eval/scripts)
log_info ""
log_info "步骤 3/4: 运行 AMBER_gen.py"
log_info "----------------------------------------"
log_info "工作目录: $CAPARENA_SCRIPTS_DIR"
if (cd "$CAPARENA_SCRIPTS_DIR" && python3 AMBER_gen.py); then
    log_info "✓ AMBER_gen.py 执行成功"
else
    log_error "✗ AMBER_gen.py 执行失败"
    exit 1
fi

# 步骤4: 运行 inference.py (在 /data0/swz/exp/AMBER)
log_info ""
log_info "步骤 4/4: 运行 inference.py"
log_info "----------------------------------------"
INFERENCE_SCRIPT="$AMBER_DIR/inference.py"
if [ ! -f "$INFERENCE_SCRIPT" ]; then
    log_error "找不到 inference.py: $INFERENCE_SCRIPT"
    exit 1
fi

log_info "工作目录: $AMBER_DIR"
if (cd "$AMBER_DIR" && python3 inference.py \
    --inference_data /data0/swz/exp/AMBER/data/query/llada-vgr-0105-mask0.json \
    --evaluation_type g); then
    log_info "✓ inference.py 执行成功"
else
    log_error "✗ inference.py 执行失败"
    exit 1
fi

log_info ""
log_info "=========================================="
log_info "所有步骤执行完成！"
log_info "=========================================="

