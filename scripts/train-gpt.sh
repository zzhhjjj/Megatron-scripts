#!/bin/bash

# debug or training
MODE="train"

# Parse arguments
for arg in "$@"
do
    if [ "$arg" == "-d" ] || [ "$arg" == "--debug" ]; then
        MODE="debug"
    fi
done

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NODE_RANK=0
DISTRIBUTED_ARGS=(
    --nproc_per_node 1
    --nnodes 1 
    --master_addr localhost 
    --master_port 6000
)

# Source argument values
# source /fsx/haojun/Megatron-files/config/qwen_moe/moe_250m.sh
# source /fsx/haojun/Megatron-files/config/qwen_moe/moe_250m_aux_loss.sh

## dense model
# 104M dense
source /fsx/haojun/Megatron-files/config/dense/megatron/dense_104M.sh
# source /fsx/haojun/Megatron-files/config/dense/megatron/dense_1B.sh
# source /fsx/haojun/Megatron-files/config/dense/megatron/dense_8B.sh

train_script=/fsx/haojun/Megatron-LM/pretrain_gpt.py

if [ "$MODE" == "train" ]; then
    echo "Training mode"
    torchrun "${DISTRIBUTED_ARGS[@]}" $train_script \
        "${GPT_MODEL_ARGS[@]}" \
        "${TRAINING_ARGS[@]}" \
        "${MODEL_PARALLEL_ARGS[@]}" \
        "${DATA_ARGS[@]}" \
        "${EVAL_AND_LOGGING_ARGS[@]}" \
        "${OTHER_ARGS[@]}" \
        "${MOE_ARGS[@]}" \
        "${wandb_args[@]}"
else
    echo "Debug mode"
    ./kill_listener.sh
    debugpy-run -m torch.distributed.run -p 5678 -- --nproc_per_node 1 \
    --nnodes 1 --rdzv_endpoint=localhost:29800 --rdzv_backend c10d --max_restarts 0 --tee 3 $train_script \
    "${GPT_MODEL_ARGS[@]}" \
        "${TRAINING_ARGS[@]}" \
        "${MODEL_PARALLEL_ARGS[@]}" \
        "${DATA_ARGS[@]}" \
        "${EVAL_AND_LOGGING_ARGS[@]}" \
        "${OTHER_ARGS[@]}" \
        "${MOE_ARGS[@]}"
fi
