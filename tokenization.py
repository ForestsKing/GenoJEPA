import argparse
import json
import os
import shutil
import warnings

from transformers import AutoTokenizer

from utils.tokenizer import GenoJEPATokenizer

warnings.filterwarnings("ignore")


def construct_tokenizer(tokenizer_dir):
    dna_tokenizer = GenoJEPATokenizer()
    dna_tokenizer.save_pretrained(tokenizer_dir)
    shutil.copy("./utils/tokenizer.py", f"{tokenizer_dir}/tokenizer.py")

    with open(f"{tokenizer_dir}/tokenizer_config.json", "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)
        tokenizer_config["auto_map"] = {"AutoTokenizer": ["tokenizer.GenoJEPATokenizer", None]}

    with open(f"{tokenizer_dir}/tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2)


def validate_tokenizer(dna_tokenizer):
    seqs = [
        "CAGGCGCAGAGACACATGCTACCGCGTCCAGG",
        "CANGCGCAGAGACACATGCTACCGCGTCCAGG",
        "CANGXGCAGAGACACATGCTACCGCGTCCAGG",
        "CANGXGCAGAGACACATGCTA",
    ]
    tokenized_seqs = dna_tokenizer(seqs, padding=True)

    for idx in range(len(seqs)):
        print("Original sequence: ", seqs[idx])
        print("Attention mask: ", tokenized_seqs["attention_mask"][idx])
        print("Tokenized ids: ", tokenized_seqs["input_ids"][idx])
        print("Decoded sequence: ", dna_tokenizer.decode(tokenized_seqs["input_ids"][idx]))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_dir", type=str, default="./output/tokenizer/")
    args = parser.parse_args()

    shutil.rmtree(args.tokenizer_dir) if os.path.exists(args.tokenizer_dir) else None
    os.makedirs(args.tokenizer_dir)

    construct_tokenizer(args.tokenizer_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    validate_tokenizer(tokenizer)
