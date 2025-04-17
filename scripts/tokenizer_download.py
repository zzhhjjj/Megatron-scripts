from transformers import AutoTokenizer, AutoModelForCausalLM
import os

megatron_files_path='/fsx/haojun/Megatron-files'

# hf_model_name = "openai-community/gpt2"
hf_model_name = 'meta-llama/Meta-Llama-3-8B'
model_name = hf_model_name.split("/")[-1]  # gpt2

# Paths to save the tokenizer and model
tokenizer_path = f"{megatron_files_path}/tokenizers/{model_name}"
model_path = f"{megatron_files_path}/hf_checkpoints/{model_name}"

save_model = True
save_tokenizer = True

if save_tokenizer:
    # Load and save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    tokenizer.save_pretrained(tokenizer_path)

if save_model:
    # Load and save model
    model = AutoModelForCausalLM.from_pretrained(hf_model_name)
    model.save_pretrained(model_path)
