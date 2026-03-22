import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import swanlab
import torch
import yaml
from catboost import CatBoostClassifier
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from sklearn.neighbors import KNeighborsClassifier
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def update_args_from_yaml(args, config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file)

    for key, value in config_dict.items():
        current_value = getattr(args, key, None)
        if current_value is None:
            setattr(args, key, value)

    return args


def process_data(train_set, valid_set, test_set, dna_model, dna_tokenizer, batch_size):
    train_data, train_label = train_set["sequence"], train_set["label"]
    valid_data, valid_label = valid_set["sequence"], valid_set["label"]
    test_data, test_label = test_set["sequence"], test_set["label"]

    embed_train_set = {
        "data": np.array(dna_model.encode(train_data, tokenizer=dna_tokenizer, batch_size=batch_size)),
        "label": np.array(train_label)
    }
    embed_valid_set = {
        "data": np.array(dna_model.encode(valid_data, tokenizer=dna_tokenizer, batch_size=batch_size)),
        "label": np.array(valid_label)
    }
    embed_test_set = {
        "data": np.array(dna_model.encode(test_data, tokenizer=dna_tokenizer, batch_size=batch_size)),
        "label": np.array(test_label)
    }

    return embed_train_set, embed_valid_set, embed_test_set


def evaluate(classifier, train_x, train_y, test_x, test_y, valid_x=None, valid_y=None):
    if valid_x is None:
        classifier.fit(train_x, train_y)
    else:
        classifier.fit(train_x, train_y, eval_set=(valid_x, valid_y))

    pred = classifier.predict(test_x)

    acc = accuracy_score(test_y, pred)
    mcc = matthews_corrcoef(test_y, pred)
    f1 = f1_score(test_y, pred, average="macro")

    return acc, mcc, f1, pred, test_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, default="./config/probing_config.yaml")
    parser.add_argument("--eval_idx", type=int, default=54)

    args = parser.parse_args()
    args = update_args_from_yaml(args, args.config_path)
    args.model_name = args.model_dir.split("/")[-1]

    df = pd.read_csv(f"{args.data_dir}/statistics.csv")
    benchmark_name = df["Benchmark"].values[args.eval_idx]
    task_name = df["Task"].values[args.eval_idx]

    fix_seed(args.random_seed)

    print("\n===================== Initializing SwanLab =====================\n")
    swanlab.init(
        project=f"Evaluate_{args.model_name}_Embedding",
        experiment_name=f"{args.eval_idx:02d}-{benchmark_name}-{task_name}",
        config=args,
        mode="online" if args.online_swanlab == 1 else "offline",
    )
    print("\n===================== SwanLab Initialized =====================\n")

    print("\n===================== Loading Model =====================\n")
    device, _, _ = get_backend()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Parameters: {total_params:,}, Trainable Parameters: {trainable_params:,}\n")
    print(model)
    print("\n===================== Model Loaded =====================\n")

    print("\n===================== Constructing Dataset =====================\n")
    dataset = load_dataset("parquet", data_files={
        "train": f"{args.data_dir}/{benchmark_name}/{task_name}/train.parquet",
        "test": f"{args.data_dir}/{benchmark_name}/{task_name}/test.parquet",
    })
    split = dataset["train"].train_test_split(test_size=0.1, seed=args.random_seed)
    train_dataset, valid_dataset, test_dataset = split["train"], split["test"], dataset["test"]
    embed_train_dataset, embed_valid_dataset, embed_test_dataset = process_data(
        train_dataset, valid_dataset, test_dataset, model, tokenizer, batch_size=args.eval_batch_size)

    train_embeddings = embed_train_dataset["data"]
    valid_embeddings = embed_valid_dataset["data"]
    test_embeddings = embed_test_dataset["data"]
    train_labels = embed_train_dataset["label"]
    valid_labels = embed_valid_dataset["label"]
    test_labels = embed_test_dataset["label"]

    print(f"\nTrain embeddings shape: {train_embeddings.shape}, labels shape: {train_labels.shape}")
    print(f"Valid embeddings shape: {valid_embeddings.shape}, labels shape: {valid_labels.shape}")
    print(f"Test embeddings shape: {test_embeddings.shape}, labels shape: {test_labels.shape}")
    print("\n===================== Dataset Constructed =====================\n")

    print("\n===================== Evaluating Model =====================\n")
    knn_classifier = KNeighborsClassifier()
    knn_acc, knn_mcc, knn_f1, knn_pred, knn_label = evaluate(
        knn_classifier, train_embeddings, train_labels, test_embeddings, test_labels)
    swanlab.log({"knn_acc": knn_acc, "knn_mcc": knn_mcc, "knn_f1": knn_f1})
    print(f"Task {task_name} KNN || ACC: {knn_acc:.4f} || MCC: {knn_mcc:.4f} || F1: {knn_f1:.4f}")

    lr_classifier = LogisticRegression(random_state=args.random_seed)
    lr_acc, lr_mcc, lr_f1, lr_pred, lr_label = evaluate(
        lr_classifier, train_embeddings, train_labels, test_embeddings, test_labels)
    swanlab.log({"lr_acc": lr_acc, "lr_mcc": lr_mcc, "lr_f1": lr_f1})
    print(f"Task {task_name} LR  || ACC: {lr_acc:.4f} || MCC: {lr_mcc:.4f} || F1: {lr_f1:.4f}")

    cb_classifier = CatBoostClassifier(random_seed=args.random_seed, verbose=False)
    cb_acc, cb_mcc, cb_f1, cb_pred, cb_label = evaluate(
        cb_classifier, train_embeddings, train_labels, test_embeddings, test_labels, valid_embeddings, valid_labels)
    swanlab.log({"cb_acc": cb_acc, "cb_mcc": cb_mcc, "cb_f1": cb_f1})
    print(f"Task {task_name} CB  || ACC: {cb_acc:.4f} || MCC: {cb_mcc:.4f} || F1: {cb_f1:.4f}\n")

    swanlab.finish()
    print("\n===================== Task Finished =====================\n")
