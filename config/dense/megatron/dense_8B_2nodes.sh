# args.sh

# === Paths ===
CHECKPOINT_PATH=/fsx/haojun/Megatron-files/checkpoints
TOKENIZER_MODEL=unsloth/Llama-3.2-1B
TENSORBOARD_LOGS_PATH=/fsx/haojun/logs/tensorboard/megatron-moe
# VOCAB_FILE=/fsx/haojun/Megatron-files/tokenizers/Llama-3.2-1B/vocab.json
# MERGE_FILE=/fsx/haojun/Megatron-files/tokenizers/Llama-3.2-1B/merges.txt
DATA_PATH=/fsx/haojun/Megatron-files/datasets/fineweb-edu-CC-MAIN-2024-51/processed/Llama-3.2-1B/fineweb-edu-CC-MAIN-2024-51_text_document

# === Argument groups ===

# https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
GPT_MODEL_ARGS=(
    --num-layers 32
    --hidden-size 4096 
    --ffn-hidden-size 14336
    --vocab-size 128256
    --num-attention-heads 32 
    --num-query-groups 4
    --group-query-attention
    --seq-length 8192 
    --max-position-embeddings 8192 
    --attention-backend flash # Can use (flash/fused/unfused/local)
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --position-embedding-type rope
    --rotary-base 10000
    --overlap-grad-reduce
    --disable-bias-linear
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    # --attention-backend auto # Can use (flash/fused/unfused/local)
)

# number of tokens per step = 64 * 8192 = 512K
TRAINING_ARGS=(
    --micro-batch-size 2
    --global-batch-size 512 # micro-batch-size * num-accumulation-steps * DP
    # --rampup-batch-size 16 16 5859375 
    --train-iters 100 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --adam-eps 1e-8
    --init-method-std 0.02
    --clip-grad 1.0 
    --bf16
    --lr 1.0e-3 
    --lr-decay-style cosine 
    --min-lr 1.0e-4
    --lr-warmup-iters 500
    --lr-decay-iters 5000 # warmup is contained in the decay. so 500 warmup + 4500 decay = 5000
    --disable-bias-linear
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 4 
    --pipeline-model-parallel-size 1
    --sequence-parallel
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    # --vocab-file $VOCAB_FILE 
    # --merge-file $MERGE_FILE 
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_MODEL
    --split 949,50,1
    --num-workers 0
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 100000 
    --eval-interval 100000 
    --save $CHECKPOINT_PATH 
    # --load $CHECKPOINT_PATH 
    --eval-iters 5
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --tensorboard-log-interval 1
    --log-throughput
    --log-timers-to-tensorboard
)

wandb_args=(
    --wandb-project qwen
    --wandb-exp-name 8B-2nodes-megatron-hf
)

OTHER_ARGS=(
    # --transformer-impl local
    --no-persist-layer-norm
    --no-gradient-accumulation-fusion
    --no-masked-softmax-fusion
    # --no-rope-fusion
)
