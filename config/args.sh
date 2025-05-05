# args.sh

# === Paths ===
CHECKPOINT_PATH=/fsx/haojun/Megatron-files/checkpoints

# GPT2 and TinyStories
# VOCAB_FILE=/fsx/haojun/Megatron-files/tokenizers/gpt2/vocab.json
# MERGE_FILE=/fsx/haojun/Megatron-files/tokenizers/gpt2/merges.txt
# DATA_PATH=/fsx/haojun/Megatron-files/datasets/TinyStories/processed/tiny_stories_text_document

# Qwen/Qwen1.5-MoE-A2.7B and fineweb-edu-CC-MAIN-2024-51
VOCAB_FILE=/fsx/haojun/Megatron-files/tokenizers/Qwen1.5-MoE-A2.7B/vocab.json
MERGE_FILE=/fsx/haojun/Megatron-files/tokenizers/Qwen1.5-MoE-A2.7B/merges.txt
DATA_PATH=/fsx/haojun/Megatron-files/datasets/fineweb-edu-CC-MAIN-2024-51/processed/fineweb-edu-CC-MAIN-2024-51_text_document

# === Argument groups ===

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
    --group-query-attention
    --num-query-groups 2
    --seq-length 1024 
    --max-position-embeddings 1024 
    --attention-backend flash # Can use (flash/fused/unfused/local)
    --normalization RMSNorm
    --norm-epsilon 1e-6
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
    --bf16
    --lr 1.0e-3 
    --lr-decay-style cosine 
    --min-lr 1.0e-4
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
    --disable-bias-linear
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1 
    --pipeline-model-parallel-size 1
)

MOE_ARGS=(
    --num-experts 8
    --expert-model-parallel-size 1
    --moe-use-legacy-grouped-gemm # a bug when DP=4 with TEGroupedMLP
    --moe-grouped-gemm
    # --moe-permute-fusion
    --moe-router-topk 1
    --moe-router-pre-softmax
    --use-distributed-optimizer
    --moe-token-dispatcher-type alltoall
    # auxiliary loss
    --moe-router-load-balancing-type aux_loss # options: aux_loss, sinkhorn, none. Default is aux_loss.
    --moe-aux-loss-coeff 1e-2
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
    --num-workers 0
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 100000 
    --eval-interval 100000 
    --save $CHECKPOINT_PATH 
    # --load $CHECKPOINT_PATH 
    --eval-iters 5
    # --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

wandb_args=(
    --wandb-project qwen_moe
    --wandb-exp-name qwen_moe_megatron
)

OTHER_ARGS=(
    # --transformer-impl local
    --no-persist-layer-norm
    --no-gradient-accumulation-fusion
    --no-masked-softmax-fusion
)
