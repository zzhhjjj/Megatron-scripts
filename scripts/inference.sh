
script_path=/fsx/haojun/Megatron-LM/examples/inference/llama_mistral/run_text_generation_llama3.sh
path_to_converted_core_checkpoint=/fsx/haojun/Megatron-files/megatron_checkpoints/Meta-Llama-3-8B/iter_0000001
path_to_downloaded_huggingface_checkpoint=/fsx/haojun/Megatron-files/hf_checkpoints/Meta-Llama-3-8B

${script_path} ${path_to_converted_core_checkpoint} ${path_to_downloaded_huggingface_checkpoint}