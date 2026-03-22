import argparse
import os
import re
import shutil

import pandas as pd
from pyfaidx import Fasta
from tqdm import tqdm


def fasta2sequence(gene, win_size, win_overlap, max_ratio):
    gene = str(gene).upper()
    gene = re.sub(r"[^ATCG]", "N", gene)

    res_list = []
    for i in range(0, len(gene), win_size - win_overlap):
        seq = gene[i: i + win_size]

        if len(seq) == win_size:
            ratio = seq.count("N") / win_size

            if ratio <= max_ratio:
                res_list.append(seq)

    return res_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--stat_file", type=str, default="../dataset/GenoJEPA-Pretraining/statistics.csv")
    parser.add_argument("--src_dir", type=str, default="../dataset/GenoJEPA-Pretraining/fasta/")
    parser.add_argument("--tar_dir", type=str, default="../dataset/GenoJEPA-Pretraining/DNA_4K/")

    parser.add_argument("--win_size", type=int, default=4096)
    parser.add_argument("--win_overlap", type=int, default=512)
    parser.add_argument("--max_ratio", type=float, default=0.15)
    parser.add_argument("--parquet_size", type=int, default=1_000_000)

    args = parser.parse_args()

    shutil.rmtree(args.tar_dir) if os.path.exists(args.tar_dir) else None
    os.makedirs(args.tar_dir)

    df = pd.read_csv(args.stat_file)

    buffer = []
    parquet_idx = 1
    bp_nums, seq_nums = [], []

    for i in range(len(df)):
        bp_num, seq_num = 0, 0
        group = df["Taxonomic Group"].values[i]
        specie = df["Specie Name"].values[i]
        fasta_file = df["Fasta File"].values[i]

        genes = Fasta(f"{args.src_dir}/{fasta_file}", sequence_always_upper=True, as_raw=True)

        for key in tqdm(genes.keys(), desc=f"[{i + 1:03d}/{len(df):03d}]: "):
            seqs = fasta2sequence(genes[key], args.win_size, args.win_overlap, args.max_ratio)

            for seq in seqs:
                buffer.append((group, specie, seq))
                bp_num += len(seq)
                seq_num += 1

                if len(buffer) >= args.parquet_size:
                    print(f"{parquet_idx:04d}.parquet is being saved ...")
                    df_buf = pd.DataFrame(buffer, columns=["group", "specie", "sequence"])
                    df_buf.to_parquet(f"{args.tar_dir}/{parquet_idx:04d}.parquet")
                    buffer.clear()
                    parquet_idx += 1

        if seq_num == 0:
            print(f"[{i:03d}] {specie} has been completely discarded !!!")

        bp_nums.append(bp_num)
        seq_nums.append(seq_num)

    if len(buffer) > 0:
        print(f"{parquet_idx:04d}.parquet is being saved ...")
        df_buf = pd.DataFrame(buffer, columns=["group", "specie", "sequence"])
        df_buf.to_parquet(f"{args.tar_dir}/{parquet_idx:04d}.parquet")

    df["Processed Sequences"] = seq_nums
    df["Processed Nucleotides"] = bp_nums
    df.to_csv(f"{args.stat_file.replace(".csv", "_new.csv")}", index=False)
