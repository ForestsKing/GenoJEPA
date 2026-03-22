import argparse
import os
import shutil
import warnings
from concurrent.futures import as_completed, ProcessPoolExecutor

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from webdataset.writer import TarWriter

warnings.filterwarnings("ignore")


def worker(file_list, worker_idx, tokenizer_dir, src_dir, tar_dir):
    print(f"[Worker {worker_idx}] Start, handling {len(file_list)} files.")
    dna_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

    for idx, data_name in enumerate(file_list):
        print(f"[Worker {worker_idx}] Tokenizing {idx + 1}/{len(file_list)} {data_name}")

        src_file = f"{src_dir}/{data_name}"
        tar_file = f"{tar_dir}/train_{data_name.split("_")[1].split(".")[0]}_{data_name.split("_")[0]}.tar"

        ds = load_dataset("parquet", data_files=src_file, split="train", streaming=False)
        tokenized_ds = ds.map(
            lambda x: dna_tokenizer(x["sequence"]),
            batched=True,
            num_proc=8,
            remove_columns=["group", "specie", "sequence"]
        )

        with TarWriter(tar_file) as sink:
            for ii, input_ids in enumerate(tokenized_ds["input_ids"]):
                sample = {
                    "__key__": f"{data_name.split("_")[0]}_{ii + 1:04d}",
                    "npy": np.array(input_ids, dtype=np.uint8),
                }
                sink.write(sample)

    print(f"[Worker {worker_idx}] Done.")
    return f"worker_{worker_idx}_done"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer_dir", type=str, default="../output/tokenizer/")
    parser.add_argument("--src_dir", type=str, default="../dataset/GenoJEPA-Pretraining/DNA_4K_shuffled/")
    parser.add_argument("--tar_dir", type=str, default="../dataset/GenoJEPA-Pretraining/DNA_4K_Tokenized/")

    parser.add_argument("--worker_num", type=int, default=10)

    args = parser.parse_args()
    shutil.rmtree(args.tar_dir) if os.path.exists(args.tar_dir) else None
    os.makedirs(args.tar_dir)

    data_files = sorted([f for f in os.listdir(args.src_dir) if f.endswith(".parquet")])
    chunks = [[] for _ in range(args.worker_num)]
    for i, file in enumerate(data_files):
        chunks[i % args.worker_num].append(file)

    results = []
    with ProcessPoolExecutor(max_workers=args.worker_num) as exe:
        futures = [
            exe.submit(worker, chunks[i], i, args.tokenizer_dir, args.src_dir, args.tar_dir)
            for i in range(args.worker_num)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())

    print("Done!")
