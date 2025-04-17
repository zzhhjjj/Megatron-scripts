# Megatron-files
This repository contains the scripts used to download and process the dataset for the Megatron-files project.

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
```

## Quick Start
1. Login to huggingface
```bash
huggingface-cli login 
```
2. Download the dataset
Save the dataset to the `datasets/dataset_name/raw` folder. The output file is in jsonl format. only "text" field is needed.
```bash
python download.py
```
3. Process the dataset
Save the processed dataset to the `datasets/dataset_name/processed` folder
```bash
./process_data.sh
```
3. Download the tokenizer
Save the tokenizer to the `tokenizers/model_name` folder   
Example:  
VOCAB_FILE=/fsx/haojun/Megatron-files/tokenizers/gpt2/vocab.json   
MERGE_FILE=/fsx/haojun/Megatron-files/tokenizers/gpt2/merges.txt    
```bash
./tokenizer_download.py
```
4. Train the model from scratch
```bash
./train-gpt.sh
```
The output contains `xxx.bin` and `xxx.idx`. The `--data-path` specified in later BERT training is the full path and new filename, but without the file extension.


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

To not use implementation from transformer engine, Apex, Fused kernel, add the following arguments to the training script:
```bash
OTHER_ARGS=(
    --transformer-impl local
    --no-persist-layer-norm
    --no-gradient-accumulation-fusion
    --no-masked-softmax-fusion
)
```

## Path: 
hf_checkpoints: for hf model checkpoints  
tokenizers: for tokenizers  
datasets: for hf datasets and processed datasets  
checkpoints: megatron model checkpoints  


