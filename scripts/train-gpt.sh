#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))


CHECKPOINT_PATH=/fsx/haojun/Megatron-files/checkpoints
VOCAB_FILE=/fsx/haojun/Megatron-files/tokenizers/gpt2/vocab.json
MERGE_FILE=/fsx/haojun/Megatron-files/tokenizers/gpt2/merges.txt
DATA_PATH=/fsx/haojun/Megatron-files/datasets/TinyStories/processed/tiny_stories_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 12 
    --hidden-size 512 
    --num-attention-heads 8 
    # --num-query-groups 4
    --seq-length 1024 
    --max-position-embeddings 1024 
    --attention-backend flash # Can use (flash/fused/unfused/local)
    --normalization RMSNorm
    # --attention-backend auto # Can use (flash/fused/unfused/local)
)

TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 8 
    # --rampup-batch-size 16 16 5859375 
    --train-iters 200 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 1.0e-3 
    --lr-decay-style cosine 
    --min-lr 1.0e-4
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1 
	--pipeline-model-parallel-size 1
    # --transformer-impl local
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    # --load $CHECKPOINT_PATH 
    --eval-iters 10
    # --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

# Disable transformer engine, Apex, Fused kernel 
OTHER_ARGS=(
    # --transformer-impl local
    # --no-persist-layer-norm
    # --no-gradient-accumulation-fusion
    # --no-masked-softmax-fusion
)

train_script=/fsx/haojun/Megatron-LM/pretrain_gpt.py

torchrun ${DISTRIBUTED_ARGS[@]} $train_script \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${OTHER_ARGS[@]}