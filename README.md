# Megatron-files
This repository contains the details for setting up the Megatron environment as well as the scripts related to Megatron

## Install
```bash
conda create -n megatron python=3.10
conda activate megatron
pip install flash-attn --no-build-isolation
pip install --no-build-isolation transformer_engine[pytorch]
git clone https://github.com/NVIDIA/apex
cd apex
NVCC_APPEND_FLAGS="--threads 4" pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" ./
# if error, switch to CUDA 12.4 
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
cd Megatron-LM
pip install -e .

# to log 
pip install tensorboard
pip install wandb
```

## Path: 
Create the folder structure
```bash
python scripts/create_folder.py
```

**hf_checkpoints**: hf model checkpoints  
**tokenizers**: tokenizers  
**datasets**: hf datasets and processed datasets  
**megatron_checkpoints**: megatron model checkpoints  

## Quick Start

### 1. Login to Hugging Face
```bash
huggingface-cli login
```

### 2. Download the Dataset
Save the dataset to the `datasets/dataset_name/raw` folder.  
The output file will be in **JSONL** format. Only the `"text"` field is needed.
```bash
python download.py
```

### 3. Process the Dataset
Save the processed dataset to the `datasets/dataset_name/processed/tokenizer_name/` folder. These variables correspond to the tokenizer and tokenized dataset.
```bash
./process_data.sh
```

### 4. Download the Tokenizer
Save the tokenizer files to the `tokenizers/model_name` folder.

**Example paths:**
```bash
VOCAB_FILE=/fsx/haojun/Megatron-files/tokenizers/gpt2/vocab.json
MERGE_FILE=/fsx/haojun/Megatron-files/tokenizers/gpt2/merges.txt
```

Download using:
```bash
python tokenizer_download.py
```

### 5. Train the Model from Scratch
Set the `VOCAB_FILE`, `MERGE_FILE`, and `DATA_PATH` in the `train-gpt.sh` file.
```bash
./train-gpt.sh
```

### 6. Debug 
```
./train-gpt.sh -d
```

**Note:**  
The output will include `.bin` and `.idx` files.  
The `--data-path` argument for later BERT training should use the full path **and new filename (without the file extension)**.


## Inference
Convert the checkpoint to megatron format
```bash
./convert_checkpoint.sh
```

Setup inference server 
Under Megatron-LM
```bash
./examples/inference/llama_mistral/run_text_generation_llama3.sh
```

Run inference
```bash
curl 'http://127.0.0.1:5000/api' \
  -X PUT \
  -H 'Content-Type: application/json; charset=UTF-8' \
  -d '{"prompts":["Hello"], "tokens_to_generate":100, "top_k":1}'
```

Run the following command to test the inference server
```bash
python examples/inference/llama_mistral/huggingface_reference.py --model-path <PATH_TO_DOWNLOADED_HUGGINGFACE_CHECKPOINT> --prompt <SOME_PROMPT>
python examples/inference/llama_mistral/huggingface_reference.py --model-path meta-llama/Meta-Llama-3-8B --prompt "Hello"
```


## Fine-tune
TODO

## Side Notes
The `setup.py` file is modified to include the following packages:
```python 
packages=setuptools.find_namespace_packages(include=["megatron.core", "megatron.core.*", "megatron.training", "megatron.training.*", "megatron.legacy", "megatron.legacy.*", "megatron.inference", "megatron.inference.*"]),
```

### Disable transformer engine, Apex, Fused kernel(if necessary)
Add the following arguments to the training script:
```bash
OTHER_ARGS=(
    --transformer-impl local
    --no-persist-layer-norm
    --no-gradient-accumulation-fusion
    --no-masked-softmax-fusion
)
```

## MoE training
### Start:
Follow the instructions in quick start.

## Compare Megatron with Nanotron
### 1. Download Dataset: 
Run the following command to download the **Fineweb-edu CC-MAIN-2024-51** dataset.
```bash
python dataset_download.py
```

### 2. Tokenizer:
Use the same tokenizer as the Nanotron training. Run the following command to tokenize the dataset into bytes.  
```bash
python tokenize_nanotron.py
```

### 3. Config:
Use the same config file as the Nanotron training. Run the following command to compare the config files. 
```bash
python compare_config.py
```
Right now, some of the config files are under /fsx/haojun/Megatron-files/config.  

### 4. Log:
Use wandb to log the training metrics. To use wandb, make sure to install tensorboard and wandb, and set aruments for both of them.  
Search for `wandb_writer.log` to find the places to log the metrics.   

### Note:
Make sure to use the same tokenizer and dataset as the Nanotron training as well as the same config file.  
Trying to match the config files, but there could be some differences.

Some known differences:
- Megatron don't add weight decay to the RMSNorm parameters, while Nanotron does.
- Dataset is the same. But the order of the dataset is different. Still need to check if Nanotron can use the same dataset as Megatron.


### Qwen Tokenizer:
To use the Qwen tokenizer, the eos token id can be hardcoded in the `tokenizer.py` file.
```python
self.eod_id = 151643 # hardcode the eod id for Qwen/Qwen1.5-MoE-A2.7B
```