# The script processes the dataset and saves it as a JSON file.

Megatron_files_path=/fsx/haojun/Megatron-files
Megatron_LM_path=/fsx/haojun/Megatron-LM


tools_dir=${Megatron_LM_path}/tools
dataset_folder=fineweb-edu-CC-MAIN-2024-51 # TinyStories, fineweb-edu-CC-MAIN-2024-51
raw_data_path=${Megatron_files_path}/datasets/${dataset_folder}/raw/dataset.json
tokenizer_model=unsloth/Llama-3.2-1B #openai-community/gpt2, Qwen/Qwen1.5-MoE-A2.7B
tokenizer_name=$(echo ${tokenizer_model} | awk -F/ '{print $NF}')
output_path=${Megatron_files_path}/datasets/${dataset_folder}/processed/${tokenizer_name}/${dataset_folder}

num_proc=24

dir_path=$(dirname "$output_path")
mkdir -p $dir_path

python $tools_dir/preprocess_data.py \
    --input $raw_data_path \
    --output-prefix $output_path \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $tokenizer_model \
    --workers $num_proc \
    # --append-eod