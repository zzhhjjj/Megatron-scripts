import json
import struct
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count

# Initialize tokenizer globally in worker
tokenizer = None

def init_tokenizer():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")

def process_batch(lines):
    global tokenizer
    result = bytearray()

    for line in lines:
        try:
            data = json.loads(line)
            text = data.get("text", "")
            token_ids = tokenizer.encode(text, add_special_tokens=False)

            for token_id in token_ids:
                result += struct.pack("<I", token_id)

        except Exception as e:
            continue  # Skip malformed lines or tokenization errors

    return result

def chunked_iterable(iterable, size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def parallel_tokenize_jsonl(input_path, output_path, batch_size=1000):
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "wb") as outfile:
        with Pool(processes=cpu_count(), initializer=init_tokenizer) as pool:
            for i, result in enumerate(pool.imap(process_batch, chunked_iterable(infile, batch_size))):
                outfile.write(result)
                if i % 10 == 0:
                    print(f"Written {i * batch_size} lines...")

input_path = "/fsx/haojun/Megatron-files/datasets/fineweb-edu-CC-MAIN-2024-51/raw/dataset.json"
output_path = "/fsx/haojun/Megatron-files/datasets/fineweb-edu-CC-MAIN-2024-51/nanotron/streamed_output.ds"

parallel_tokenize_jsonl(input_path, output_path, batch_size=1000)