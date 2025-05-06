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
NUM_NODES=1
NODE_RANK=0
GPUS_PER_NODE=1
WORLD_SIZE=$(($GPUS_PER_NODE * $NUM_NODES))

# Source argument values
# source /fsx/haojun/Megatron-files/config/qwen_moe/moe_250m.sh
source /fsx/haojun/Megatron-files/config/qwen_moe/moe_250m_aux_loss.sh

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
    debugpy-run -m torch.distributed.run -p 5678 -- --nproc_per_node $GPUS_PER_NODE \
    --nnodes 1 --rdzv_endpoint=localhost:29800 --rdzv_backend c10d --max_restarts 0 --tee 3 $train_script \
    "${GPT_MODEL_ARGS[@]}" \
        "${TRAINING_ARGS[@]}" \
        "${MODEL_PARALLEL_ARGS[@]}" \
        "${DATA_ARGS[@]}" \
        "${EVAL_AND_LOGGING_ARGS[@]}" \
        "${OTHER_ARGS[@]}" \
        "${MOE_ARGS[@]}"
fi
