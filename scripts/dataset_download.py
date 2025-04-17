"""
This script downloads the TinyStories dataset and saves it as a JSONL file.
"""

from datasets import load_dataset
import json
import os

megatron_files_path='/fsx/haojun/Megatron-files'
hf_dataset_name = "roneneldan/TinyStories"
dataset_name = hf_dataset_name.split("/")[-1]
split = "train"

save_path = f"{megatron_files_path}/datasets/{dataset_name}/raw/dataset.json"

dataset = load_dataset(hf_dataset_name, split=split)  
data = dataset.to_list() # Convert to list of dicts (serializable)

# Make sure the directory exists
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Write JSONL
with open(save_path, "w", encoding="utf-8") as f:
    for i, example in enumerate(dataset):
        json_obj = {
            "src": "TinyStories Dataset",
            "text": example["text"],
            "type": "Eng",
            "id": str(i),
        }
        f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

print(f"Dataset saved to {save_path}")

# Read and print first 3 lines for review
print("\nFirst 3 lines in the saved file:")
with open(save_path, "r", encoding="utf-8") as f:
    for _ in range(3):
        print(f.readline().strip())
