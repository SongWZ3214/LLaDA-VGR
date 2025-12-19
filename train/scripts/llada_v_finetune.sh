# export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=WARN
# export NCCL_DEBUG_SUBSYS=ALL

# 修复 DeepSpeed 兼容性问题
# 禁用 GDS (GPU Direct Storage) 以避免 OpBuilder.has_function() 错误
export DS_SKIP_CUDA_CHECK=0
export DS_BUILD_OPS=0
# 设置 DeepSpeed 跳过某些操作的兼容性检查
export DS_SKIP_GDS_CHECK=1
# 禁用 async_io 以避免 libaio 相关错误
export DS_BUILD_ASYNC_IO=0
# 设置 DeepSpeed 跳过 GDS 检查
export DS_SKIP_GDS=1

# 节点和GPU数量（如果未提供参数，使用默认值）
num_node=${1:-1}  # 默认1个节点
gpu_num=${2:-8}  # 默认8个GPU

# 获取脚本所在目录的绝对路径，并设置 PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${TRAIN_DIR}:${PYTHONPATH}"

# 切换到训练目录
cd "${TRAIN_DIR}"

# 建议：如果是A100 80G，gradient_accumulation_steps 可以设小一点，增加 per_device_batch_size
# 目标是 Global Batch Size 最好在 128 左右
# 当前：2 (per_device) * 8 (gpu) * 4 (accum) = 64。有点小，建议 accum 改为 8，或者 per_device 改为 4。

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29199"}
RANK=${RANK:-"0"}

echo "master_addr ${MASTER_ADDR}"
echo "master_port ${MASTER_PORT}"
echo "node_rank ${RANK}"
echo "gpu_num ${gpu_num}"
echo "num_node ${num_node}"

LLM_VERSION="jiyatai/ReDiff"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip2-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Finetune ################

PROMPT_VERSION="llava_llada"

BASE_RUN_NAME="llada_vgr_lora_rank64"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

# 数据路径
DATA_PATH="/data0/swz/refCOCO+/refcoco+_training_data.json"
IMAGE_FOLDER="/data0/swz/refCOCO+/images/cropped"

# DeepSpeed 配置文件选择
# 如果显存够（A100 80G），建议用 zero2.json 速度更快；显存吃紧用 zero3.json
DS_CONFIG="scripts/zero2.json" 

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=${gpu_num} --nnodes=${num_node} --master_addr=${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank=${RANK} \
    "${TRAIN_DIR}/llava/train/train_mem.py" \
    --deepspeed ${DS_CONFIG} \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --video_folder "" \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --modules_to_save "mm_projector" \
    --mm_tunable_parts="mm_mlp_adapter" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio pad \
    --mm_patch_merge_type spatial_unpad \
    --revise True \
    --bf16 True \
    --run_name $BASE_RUN_NAME \
    --output_dir "exp/$BASE_RUN_NAME" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --torch_compile False \
    --dataloader_drop_last True \
    --attn_implementation sdpa \
    --use_conversation_mask False