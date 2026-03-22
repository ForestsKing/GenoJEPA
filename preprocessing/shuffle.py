import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def worker(files, wid, bucket_num, batch_size, tmp_dir, seed):
    rng = np.random.default_rng(seed + wid)
    wdir = f"{tmp_dir}/worker_{wid}"
    os.makedirs(wdir, exist_ok=True)

    writers = {}
    schema = None

    for f in tqdm(files, desc=f"Worker {wid}", position=wid, leave=False):
        pf = pq.ParquetFile(f)

        for batch in pf.iter_batches(batch_size=batch_size):
            if schema is None:
                schema = batch.schema

            rows = batch.num_rows
            bucket_ids = rng.integers(0, bucket_num, rows)
            sort_indices = np.argsort(bucket_ids)
            take_indices = pa.array(sort_indices)
            sorted_batch = batch.take(take_indices)

            sorted_bucket_ids = bucket_ids[sort_indices]
            unique_buckets, split_indices = np.unique(sorted_bucket_ids, return_index=True)

            for i, b_id in enumerate(unique_buckets):
                start = split_indices[i]
                end = split_indices[i + 1] if i < len(split_indices) - 1 else rows
                sub_batch = sorted_batch.slice(offset=start, length=end - start)

                if b_id not in writers:
                    writers[b_id] = pq.ParquetWriter(
                        f"{wdir}/bucket_{b_id}.parquet",
                        schema,
                        compression="snappy"
                    )
                writers[b_id].write_batch(sub_batch)

    for w in writers.values():
        w.close()

    return wdir


def merge_worker(b_id, bucket_num, tmp_dirs, tar_dir):
    out_path = f"{tar_dir}/{b_id + 1:04d}_{bucket_num:04d}.parquet"

    tmp_files = []
    for d in tmp_dirs:
        p = f"{d}/bucket_{b_id}.parquet"
        if os.path.exists(p):
            tmp_files.append(p)

    if not tmp_files:
        return None

    first_pf = pq.ParquetFile(tmp_files[0])
    schema = first_pf.schema_arrow

    with pq.ParquetWriter(out_path, schema, compression="snappy") as writer:
        for tf in tmp_files:
            pf = pq.ParquetFile(tf)
            for batch in pf.iter_batches():
                writer.write_batch(batch)

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dir", type=str, default="../dataset/GenoJEPA-Pretraining/DNA_4K/")
    parser.add_argument("--tmp_dir", type=str, default="../dataset/GenoJEPA-Pretraining/tmp/")
    parser.add_argument("--tar_dir", type=str, default="../dataset/GenoJEPA-Pretraining/DNA_4K_shuffled/")

    parser.add_argument("--bucket_num", type=int, default=1_000)
    parser.add_argument("--worker_num", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    shutil.rmtree(args.tmp_dir) if os.path.exists(args.tmp_dir) else None
    shutil.rmtree(args.tar_dir) if os.path.exists(args.tar_dir) else None
    os.makedirs(args.tmp_dir)
    os.makedirs(args.tar_dir)

    files = sorted(glob(f"{args.src_dir}/*.parquet"))
    chunks = [[] for _ in range(args.worker_num)]
    for i, f in enumerate(files):
        chunks[i % args.worker_num].append(f)

    print("Phase 1: Scattering data into buckets (Parallel)...")
    tmp_dirs = []
    with ProcessPoolExecutor(max_workers=args.worker_num) as exe:
        futures = []
        for i in range(args.worker_num):
            if chunks[i]:
                futures.append(exe.submit(
                    worker, chunks[i], i, args.bucket_num, args.batch_size, args.tmp_dir, args.seed
                ))

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Workers"):
            tmp_dirs.append(fut.result())

    print("\n\nPhase 2: Merging buckets (Parallel)...")
    merge_workers = min(args.worker_num, args.bucket_num)

    with ProcessPoolExecutor(max_workers=merge_workers) as exe:
        futures = [
            exe.submit(merge_worker, b, args.bucket_num, tmp_dirs, args.tar_dir)
            for b in range(args.bucket_num)
        ]

        for _ in tqdm(as_completed(futures), total=args.bucket_num, desc="Merging"):
            pass

    print("Cleaning up temporary files...")
    shutil.rmtree(args.tmp_dir)
    print("Done!")
