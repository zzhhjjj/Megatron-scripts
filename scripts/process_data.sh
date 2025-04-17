# The script processes the TinyStories dataset and saves it as a JSON file.

Megatron-files_path=/fsx/haojun/Megatron-files
Megatron-LM_path=/fsx/haojun/Megatron-LM

tools_dir=${Megatron-LM_path}/tools
raw_data_path=${Megatron-files_path}/datasets/TinyStories/raw/dataset.json
output_path=${Megatron-files_path}/datasets/TinyStories/processed/tiny_stories
tokenizer_model=openai-community/gpt2
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
       