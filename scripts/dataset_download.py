"""
This script downloads datasets from Hugging Face in streaming mode and saves them as JSONL files:
1. TinyStories
2. fineweb-edu (CC-MAIN-2024-51 subset, train split)
"""

from datasets import load_dataset
import json
import os


megatron_files_path = '/fsx/haojun/Megatron-files'

# def save_dataset_as_jsonl(hf_dataset_name, subset_or_split, split, src_label, save_filename):
#     """
#     Load a dataset and save it as JSONL file.
    
#     Parameters:
#         hf_dataset_name (str): Hugging Face dataset path.
#         subset_or_split (str): Optional config (subset); pass None if not needed.
#         split (str): Split to use (e.g., 'train').
#         src_label (str): String for the 'src' field in the output JSON.
#         save_filename (str): Output filename (full path).
#     """
#     if subset_or_split:
#         dataset = load_dataset(hf_dataset_name, subset_or_split, split=split)
#         dataset_name = f"{hf_dataset_name.split('/')[-1]}-{subset_or_split}"
#     else:
#         dataset = load_dataset(hf_dataset_name, split=split)
#         dataset_name = hf_dataset_name.split("/")[-1]

#     data = dataset.to_list()
#     os.makedirs(os.path.dirname(save_filename), exist_ok=True)

#     with open(save_filename, "w", encoding="utf-8") as f:
#         for i, example in enumerate(data):
#             json_obj = {
#                 "src": src_label,
#                 "text": example["text"],
#                 "type": "Eng",
#                 "id": str(i),
#             }
#             f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

#     print(f"Dataset '{dataset_name}' saved to {save_filename}")
#     print(f"\nFirst 3 lines from {dataset_name}:")
#     with open(save_filename, "r", encoding="utf-8") as f:
#         for _ in range(3):
#             print(f.readline().strip())

def save_streaming_dataset_as_jsonl(hf_dataset_name, subset_or_split, split, src_label, save_filename):
    """
    Load a dataset in streaming mode and save it as a JSONL file.
    """
    if subset_or_split:
        dataset = load_dataset(hf_dataset_name, subset_or_split, split=split, streaming=True)
        dataset_name = f"{hf_dataset_name.split('/')[-1]}-{subset_or_split}"
    else:
        dataset = load_dataset(hf_dataset_name, split=split, streaming=True)
        dataset_name = hf_dataset_name.split("/")[-1]

    os.makedirs(os.path.dirname(save_filename), exist_ok=True)

    with open(save_filename, "w", encoding="utf-8") as f:
        for i, example in enumerate(dataset):
            if "text" not in example:
                continue  # Skip corrupted or malformed records

            json_obj = {
                "src": src_label,
                "text": example["text"],
                "type": "Eng",
                "id": str(i),
            }
            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

            if i <=2:
                print(f"Sample record {i}: {json_obj}")  # print first 3 examples
            if i % 100000 == 0 and i != 0:
                print(f"Written {i} examples...")

    print(f"Streaming dataset '{dataset_name}' saved to {save_filename}")


# # === TinyStories ===
# save_dataset_as_jsonl(
#     hf_dataset_name="roneneldan/TinyStories",
#     subset_or_split=None,
#     split="train",
#     src_label="TinyStories Dataset",
#     save_filename=f"{megatron_files_path}/datasets/TinyStories/raw/dataset.json"
# )

# === fineweb-edu (CC-MAIN-2024-51) ===
save_streaming_dataset_as_jsonl(
    hf_dataset_name="HuggingFaceFW/fineweb-edu",
    subset_or_split="CC-MAIN-2024-51",
    split="train",
    src_label="fineweb-edu - CC-MAIN-2024-51",
    save_filename=f"{megatron_files_path}/datasets/fineweb-edu-CC-MAIN-2024-51/raw/dataset.json"
)
