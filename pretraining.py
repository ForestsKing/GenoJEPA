import argparse
import os
import random
import shutil
import warnings
from glob import glob

import numpy as np
import pandas as pd
import swanlab
import torch
import yaml
from accelerate.test_utils.testing import get_backend
from datasets import load_dataset, Dataset
from lightly.utils.scheduler import cosine_schedule
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from webdataset.compat import WebDataset

from utils.model import GenoJEPAConfig, GenoJEPAForPreTraining

warnings.filterwarnings("ignore")


def init_dir(tar_dir):
    shutil.rmtree(tar_dir) if os.path.exists(tar_dir) else None
    os.makedirs(tar_dir)


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


class WebDatasetAdapter(IterableDataset):
    def __init__(self, wds_dataset):
        self.dataset = wds_dataset

    def __iter__(self):
        for sample in self.dataset:
            yield {"input_ids": sample[0]}


def compute_score(output, random_seed=42):
    data, label = output.predictions, output.label_ids
    train_data, test_data, train_label, test_label = train_test_split(
        data, label, test_size=0.2, random_state=random_seed)

    lr_classifier = LogisticRegression(random_state=random_seed)
    lr_classifier.fit(train_data, train_label)
    lr_pred = lr_classifier.predict(test_data)

    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(train_data, train_label)
    knn_pred = knn_classifier.predict(test_data)

    result = {
        "lr_mcc": matthews_corrcoef(test_label, lr_pred),
        "knn_mcc": matthews_corrcoef(test_label, knn_pred),
    }

    return result


class CustomCallback(TrainerCallback):
    def __init__(self, config):
        self.weight_decay = config.weight_decay
        self.metric = {"pred_loss": [], "sigreg_loss": []}

    def on_step_begin(self, args, state, control, **kwargs):
        weight_decay = cosine_schedule(
            step=state.global_step,
            max_steps=state.max_steps,
            start_value=self.weight_decay[0],
            end_value=self.weight_decay[1],
        )

        optimizer = kwargs.get("optimizer")
        for group in optimizer.param_groups:
            group["weight_decay"] = weight_decay if group["weight_decay"] != 0.0 else 0.0

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        cur_metric = model.get_detailed_metric()
        self.metric["pred_loss"].append(cur_metric["pred_loss"])
        self.metric["sigreg_loss"].append(cur_metric["sigreg_loss"])

        if state.global_step % args.logging_steps == 0:
            optimizer = kwargs.get("optimizer")
            weight_decay = max(group["weight_decay"] for group in optimizer.param_groups)

            swanlab.log(
                data={
                    "train/weight_decay": weight_decay,
                    "train/pred_loss": np.mean(self.metric["pred_loss"]),
                    "train/sigreg_loss": np.mean(self.metric["sigreg_loss"]),
                },
                step=state.global_step,
            )
            self.metric = {"pred_loss": [], "sigreg_loss": []}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, default="./config/pretraining_config_tiny.yaml")

    args = parser.parse_args()
    args = update_args_from_yaml(args, args.config_path)
    args.model_name = args.model_dir.split("/")[-1]

    init_dir(args.model_dir)
    fix_seed(args.random_seed)

    print("\n===================== Initializing SwanLab =====================\n")
    swanlab.init(
        project=f"Pretrain_GenoJEPA",
        experiment_name=f"{args.model_name}",
        config=args,
        mode="online" if args.online_swanlab == 1 else "offline",
    )
    print("\n===================== SwanLab Initialized =====================\n")

    print("\n===================== Constructing Model =====================\n")
    device, _, _ = get_backend()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    model = GenoJEPAForPreTraining(GenoJEPAConfig(
        vocab_num=tokenizer.vocab_size,
        pad_value=tokenizer.pad_token_id,

        global_num=args.global_num,
        global_scale=args.global_scale,
        local_num=args.local_num,
        local_scale=args.local_scale,

        patch_size=args.patch_size,
        embed_size=args.embed_size,
        head_size=args.head_size,

        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        layer_num=args.layer_num,
        head_num=args.head_num,

        slice_num=args.slice_num,
        point_num=args.point_num,
        max_t=args.max_t,
        alpha=args.alpha,
    )).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Parameters: {total_params:,}, Trainable Parameters: {trainable_params:,}\n")
    print(model)
    print("\n===================== Model Constructed =====================\n")

    print("\n===================== Constructing Dataset =====================\n")
    print("Train data will be streamed from the disk during training...")
    train_data_files = sorted(glob(f"{args.train_data_dir}/*.tar"))
    tokenized_train_dataset = WebDatasetAdapter(
        WebDataset(train_data_files, resampled=True).shuffle(args.buffer_size).decode("torch").to_tuple("npy")
    )

    valid_df = pd.read_csv(f"{args.valid_data_dir}/statistics.csv")
    benchmark_name, task_name = valid_df["Benchmark"].values[args.valid_idx], valid_df["Task"].values[args.valid_idx]
    valid_parquet = load_dataset(
        path="parquet",
        data_files=f"{args.valid_data_dir}/{benchmark_name}/{task_name}/train.parquet"
    )
    valid_data, valid_label = valid_parquet["train"]["sequence"], valid_parquet["train"]["label"]
    valid_dataset = Dataset.from_dict({"data": valid_data, "labels": valid_label})
    tokenized_valid_dataset = valid_dataset.map(lambda x: tokenizer(x["data"]), batched=True, num_proc=8)
    print("\n===================== Dataset Constructed =====================\n")

    print("\n===================== Constructing Trainer =====================\n")
    log_path = f"./checkpoint/train/{args.model_name}/"
    shutil.rmtree(log_path) if os.path.exists(log_path) else None

    train_args = TrainingArguments(
        output_dir=log_path,
        max_steps=args.train_step,

        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.train_accumulation_step,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_accumulation_steps=args.eval_accumulation_step,

        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=max(args.train_step // 100, 1),
        eval_on_start=True,
        save_strategy="steps",
        save_steps=max(args.train_step // 100, 1),
        save_total_limit=8,
        overwrite_output_dir=True,

        dataloader_num_workers=4,
        dataloader_prefetch_factor=4,
        dataloader_drop_last=True,
        dataloader_persistent_workers=True,

        learning_rate=args.learning_rate[0],
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": args.learning_rate[1]},
        weight_decay=args.weight_decay[0],
        warmup_steps=args.warmup_step,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,

        bf16=True,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_score,
        callbacks=[CustomCallback(args)],
    )
    print("Trainer ready for training.")
    print("\n===================== Trainer Constructed =====================\n")

    print("\n===================== Start Training =====================\n")
    trainer.train()
    print("\n===================== Training Finished =====================\n")

    print("\n===================== Saving Model =====================\n")
    tokenizer.save_pretrained(args.model_dir)
    model.save_pretrained(args.model_dir)
    shutil.copy("./utils/model.py", f"{args.model_dir}/model.py")

    swanlab.finish()
    print("\n===================== Model Saved =====================\n")
