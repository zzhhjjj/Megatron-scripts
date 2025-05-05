from transformers import AutoTokenizer, AutoModelForCausalLM
import os

megatron_files_path='/fsx/haojun/Megatron-files'

# hf_model_name = "openai-community/gpt2"
# hf_model_name = 'meta-llama/Meta-Llama-3-8B'
hf_model_name = 'Qwen/Qwen1.5-MoE-A2.7B'
model_name = hf_model_name.split("/")[-1]  # gpt2 for example

# Paths to save the tokenizer and model
tokenizer_path = f"{megatron_files_path}/tokenizers/{model_name}"
model_path = f"{megatron_files_path}/hf_checkpoints/{model_name}"

def download_tokenizer(hf_model_name, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    tokenizer.save_pretrained(tokenizer_path)

def download_model(hf_model_name, model_path):
    model = AutoModelForCausalLM.from_pretrained(hf_model_name)
    model.save_pretrained(model_path)

save_model = False
save_tokenizer = True

if save_tokenizer:
    download_tokenizer(hf_model_name, tokenizer_path)

if save_model:
    download_model(hf_model_name, model_path)
