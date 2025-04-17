# convert hf checkpoint to megatron checkpoint
# tools/checkpoint/loader_llama_mistral.py

Megatron-files_path=/fsx/haojun/Megatron-files
Megatron-LM_path=/fsx/haojun/Megatron-LM

export PYTHONPATH=${Megatron-LM_path}:$PYTHONPATH

tools_path=${Megatron-LM_path}/tools/checkpoint/convert.py

model_name=Meta-Llama-3-8B
model_size=llama3
checkpoint_type=hf
hf_checkpoints_dir=${Megatron-files_path}/hf_checkpoints/${model_name}
tokenizer_model=${Megatron-files_path}/tokenizers/${model_name}
megatron_checkpoints_dir=${Megatron-files_path}/checkpoints/${model_name}
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