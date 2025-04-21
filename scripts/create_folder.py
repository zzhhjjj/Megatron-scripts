import os

megatron_files_path='/fsx/haojun/Megatron-files'

os.makedirs(megatron_files_path, exist_ok=True)
os.makedirs(f"{megatron_files_path}/hf_checkpoints", exist_ok=True)
os.makedirs(f"{megatron_files_path}/tokenizers", exist_ok=True)
os.makedirs(f"{megatron_files_path}/datasets", exist_ok=True)
os.makedirs(f"{megatron_files_path}/checkpoints", exist_ok=True)

