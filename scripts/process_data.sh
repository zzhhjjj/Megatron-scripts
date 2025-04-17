# The script processes the TinyStories dataset and saves it as a JSON file.
tools_dir=/fsx/haojun/Megatron-LM/tools
raw_data_path=/fsx/haojun/Megatron-files/datasets/TinyStories/raw/dataset.json
output_path=/fsx/haojun/Megatron-files/datasets/TinyStories/processed/tiny_stories
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
       