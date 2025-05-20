#!/bin/bash
#SBATCH --job-name=dense-megatron
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hopper-prod
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --output=/fsx/haojun/logs/dense/tmp/%x-%j.out
#SBATCH --error=/fsx/haojun/logs/dense/tmp/%x-%j.out
#SBATCH --reservation=moes_benchmarks # reservation for benchmark

# ======== BEGIN: Handle Argument ========
# Usage: sbatch --gres=gpu:N submit_job.slurm
# Example: sbatch --gres=gpu:1 submit_job.slurm

# Activate conda environment
source /admin/home/haojun_zhao/miniconda3/etc/profile.d/conda.sh
conda activate /admin/home/haojun_zhao/miniconda3/envs/megatron

# script for a benchmark
set -x -e
CONFIG_ID=$(echo "$@" | sed -n 's/.*--\([a-zA-Z0-9_]*\).*/\1/p')
NUM_NODES=$(scontrol show job $SLURM_JOB_ID | grep -oP 'NumNodes=\K\d+' | head -1)
if [ $NUM_NODES -gt 1 ]; then
    NUM_GPUS=8
else
    NUM_GPUS=$(scontrol show job $SLURM_JOB_ID | grep -oP 'gres/gpu=\K\d+' | head -1)
fi
echo "Running with $NUM_GPUS GPUs on $NUM_NODES nodes"
echo "START TIME: $(date)"

# SLURM stuff
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9001
export FI_PROVIDER="efa" 

DISTRIBUTED_ARGS=(
    --nproc_per_node $NUM_GPUS
    --nnodes $NUM_NODES 
    --master_addr localhost 
    --master_port 6000
)

echo "NCCL version: $(python -c "import torch;print(torch.cuda.nccl.version())")"
echo "CUDA version: $(python -c "import torch;print(torch.version.cuda)")"

# Config
# source /fsx/haojun/Megatron-files/config/qwen_moe/moe_250m_long.sh


## Dense model
case $CONFIG_ID in
    104M)
        CONFIG_FILE=/fsx/haojun/Megatron-files/config/dense/megatron/dense_104M.sh
        ;;
    1B)
        CONFIG_FILE=/fsx/haojun/Megatron-files/config/dense/megatron/dense_1B.sh
        ;;
    8B)
        CONFIG_FILE=/fsx/haojun/Megatron-files/config/dense/megatron/dense_8B.sh
        ;;
    8B_2nodes)
        CONFIG_FILE=/fsx/haojun/Megatron-files/config/dense/megatron/dense_8B_2nodes.sh
        ;;
    8B_4nodes)
        CONFIG_FILE=/fsx/haojun/Megatron-files/config/dense/megatron/dense_8B_4nodes.sh
        ;;
    *)
        echo "Unsupported Config name: $CONFIG_ID"
        exit 1
        ;;
esac

source $CONFIG_FILE

train_script=/fsx/haojun/Megatron-LM/pretrain_gpt.py

# torchrun "${DISTRIBUTED_ARGS[@]}" $train_script \
#     "${GPT_MODEL_ARGS[@]}" \
#     "${TRAINING_ARGS[@]}" \
#     "${MODEL_PARALLEL_ARGS[@]}" \
#     "${DATA_ARGS[@]}" \
#     "${EVAL_AND_LOGGING_ARGS[@]}" \
#     "${OTHER_ARGS[@]}" \
#     "${MOE_ARGS[@]}" \
#     "${wandb_args[@]}"

srun torchrun --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
   --max_restarts=0 --tee=3 $train_script \
   "${GPT_MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${EVAL_AND_LOGGING_ARGS[@]}" \
    "${OTHER_ARGS[@]}" \
    "${wandb_args[@]}"

# move logs
mv /fsx/haojun/logs/moe/tmp/*${SLURM_JOB_ID}* /fsx/haojun/logs/moe/succeed