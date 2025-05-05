# The script processes the TinyStories dataset and saves it as a JSON file.

Megatron_files_path=/fsx/haojun/Megatron-files
Megatron_LM_path=/fsx/haojun/Megatron-LM


tools_dir=${Megatron_LM_path}/tools
dataset_folder=fineweb-edu-CC-MAIN-2024-51 # TinyStories
raw_data_path=${Megatron_files_path}/datasets/${dataset_folder}/raw/dataset.json
output_path=${Megatron_files_path}/datasets/${dataset_folder}/processed/${dataset_folder}
tokenizer_model=Qwen/Qwen1.5-MoE-A2.7B #openai-community/gpt2
num_proc=24

dir_path=$(dirname "$output_path")
mkdir -p $dir_path

python $tools_dir/preprocess_data.py \
    --input $raw_data_path \
    --output-prefix $output_path \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $tokenizer_model \
    --workers $num_proc \
    --append-eod
       