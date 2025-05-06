#!/bin/bash
#SBATCH --job-name=moe_train_megatron
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hopper-prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --output=/fsx/haojun/logs/moe/tmp/%x-%j.out
#SBATCH --error=/fsx/haojun/logs/moe/tmp/%x-%j.out


MODE="train"


# script for a benchmark
set -x -e

# SLURM stuff
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export FI_PROVIDER="efa" 
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

echo "NCCL version: $(python -c "import torch;print(torch.cuda.nccl.version())")"
echo "CUDA version: $(python -c "import torch;print(torch.version.cuda)")"

# Config
# source /fsx/haojun/Megatron-files/config/qwen_moe/moe_250m_long.sh
source /fsx/haojun/Megatron-files/config/qwen_moe/moe_250m_aux_loss_long.sh

train_script=/fsx/haojun/Megatron-LM/pretrain_gpt.py

torchrun "${DISTRIBUTED_ARGS[@]}" $train_script \
    "${GPT_MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${EVAL_AND_LOGGING_ARGS[@]}" \
    "${OTHER_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${wandb_args[@]}"

# move logs
mv /fsx/haojun/logs/moe/tmp/*${SLURM_JOB_ID}* /fsx/haojun/logs/moe/succeed