# convert hf checkpoint to megatron checkpoint
# tools/checkpoint/loader_llama_mistral.py
export PYTHONPATH=/fsx/haojun/Megatron-LM:$PYTHONPATH

tools_path=/fsx/haojun/Megatron-LM/tools/checkpoint/convert.py

model_name=Meta-Llama-3-8B
model_size=llama3
checkpoint_type=hf
hf_checkpoints_dir=/fsx/haojun/Megatron-files/hf_checkpoints/${model_name}
tokenizer_model=/fsx/haojun/Megatron-files/tokenizers/Meta-Llama-3-8B
megatron_checkpoints_dir=/fsx/haojun/Megatron-files/megatron_checkpoints/${model_name}
loader=llama_mistral
saver=core
TP=1
PP=1

python ${tools_path} \
    --model-type GPT \
    --loader ${loader} \
    --load-dir ${hf_checkpoints_dir} \
    --saver ${saver} \
    --model-size ${model_size} \
    --tokenizer-model ${tokenizer_model} \
    --checkpoint-type ${checkpoint_type} \
    --save-dir ${megatron_checkpoints_dir} \
    --target-tensor-parallel-size ${TP} \
    --target-pipeline-parallel-size ${PP} \
    --bf16