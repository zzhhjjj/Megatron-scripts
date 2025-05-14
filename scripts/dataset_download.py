"""
This script downloads datasets from Hugging Face in streaming mode and saves them as JSONL files:
1. TinyStories
2. fineweb-edu (CC-MAIN-2024-51 subset, train split)
"""

from datasets import load_dataset
import json
import os
from datasets import load_from_disk
from multiprocessing import Process, cpu_count

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

# def save_streaming_dataset_as_jsonl(hf_dataset_name, subset_or_split, split, src_label, save_filename):
#     """
#     Load a dataset in streaming mode and save it as a JSONL file.
#     """
#     if subset_or_split:
#         dataset = load_dataset(hf_dataset_name, subset_or_split, split=split, streaming=True)
#         dataset_name = f"{hf_dataset_name.split('/')[-1]}-{subset_or_split}"
#     else:
#         dataset = load_dataset(hf_dataset_name, split=split, streaming=True)
#         dataset_name = hf_dataset_name.split("/")[-1]

#     os.makedirs(os.path.dirname(save_filename), exist_ok=True)

#     with open(save_filename, "w", encoding="utf-8") as f:
#         for i, example in enumerate(dataset):
#             if "text" not in example:
#                 continue  # Skip corrupted or malformed records

#             json_obj = {
#                 "src": src_label,
#                 "text": example["text"],
#                 "type": "Eng",
#                 "id": str(i),
#             }
#             f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

#             if i <=2:
#                 print(f"Sample record {i}: {json_obj}")  # print first 3 examples
#             if i % 100000 == 0 and i != 0:
#                 print(f"Written {i} examples...")

#     print(f"Streaming dataset '{dataset_name}' saved to {save_filename}")


# # === TinyStories ===
# save_dataset_as_jsonl(
#     hf_dataset_name="roneneldan/TinyStories",
#     subset_or_split=None,
#     split="train",
#     src_label="TinyStories Dataset",
#     save_filename=f"{megatron_files_path}/datasets/TinyStories/raw/dataset.json"
# )

# === fineweb-edu (CC-MAIN-2024-51) ===
# save_streaming_dataset_as_jsonl(
#     hf_dataset_name="HuggingFaceFW/fineweb-edu",
#     subset_or_split="CC-MAIN-2024-51",
#     split="train",
#     src_label="fineweb-edu - CC-MAIN-2024-51",
#     save_filename=f"{megatron_files_path}/datasets/fineweb-edu-CC-MAIN-2024-51/raw/dataset.json"
# )


# save the dataset to disk
dataset_local_path = "/fsx/haojun/Megatron-files/datasets/local/huggingface"
os.makedirs(dataset_local_path, exist_ok=True)
json_path = "/fsx/haojun/Megatron-files/datasets/local/raw"
dataset = load_dataset("HuggingFaceFW/fineweb-edu", "CC-MAIN-2024-51", split="train")  # no streaming
dataset.save_to_disk(dataset_local_path)  # save to disk

# process the dataset in parallel
def process_shard(dataset_path, start, end, save_path, src_label, shard_id):
    dataset = load_from_disk(dataset_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    output_file = f"{save_path}.part{shard_id}"
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(start, end):
            example = dataset[i]
            if "text" not in example:
                continue
            json_obj = {
                "src": src_label,
                "text": example["text"],
                "type": "Eng",
                "id": str(i),
            }
            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
    print(f"[Shard {shard_id}] Done: {output_file}")

def parallel_process_dataset(dataset_path, save_path, src_label, num_processes=None):
    dataset = load_from_disk(dataset_path)
    total = len(dataset)
    num_processes = num_processes or cpu_count()
    chunk_size = (total + num_processes - 1) // num_processes

    processes = []
    for i in range(num_processes):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        p = Process(target=process_shard, args=(dataset_path, start, end, save_path, src_label, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Merge all parts
    with open(save_path, "w", encoding="utf-8") as outfile:
        for i in range(num_processes):
            part_file = f"{save_path}.part{i}"
            with open(part_file, "r", encoding="utf-8") as infile:
                for line in infile:
                    outfile.write(line)
            os.remove(part_file)

    print(f"[Main] All parts merged into {save_path}")

parallel_process_dataset(
    dataset_path=dataset_local_path,
    save_path=f"{json_path}/dataset.json",
    src_label="fineweb-edu - CC-MAIN-2024-51",
    num_processes=8  # or None to use all CPU cores
)