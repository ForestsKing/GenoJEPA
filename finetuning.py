import argparse
import os
import random
import shutil
import warnings

import numpy as np
import pandas as pd
import swanlab
import torch
import yaml
from accelerate.test_utils.testing import get_backend
from datasets import Dataset, load_dataset
from lightly.utils.scheduler import cosine_schedule
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainerCallback
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

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


def process_data(train_set, valid_set, test_set, dna_tokenizer):
    train_data, train_label = train_set["sequence"], train_set["label"]
    valid_data, valid_label = valid_set["sequence"], valid_set["label"]
    test_data, test_label = test_set["sequence"], test_set["label"]

    train_set = Dataset.from_dict({"data": train_data, "labels": train_label})
    valid_set = Dataset.from_dict({"data": valid_data, "labels": valid_label})
    test_set = Dataset.from_dict({"data": test_data, "labels": test_label})

    tokenized_train_set = train_set.map(lambda x: dna_tokenizer(x["data"]), batched=True, num_proc=8)
    tokenized_valid_set = valid_set.map(lambda x: dna_tokenizer(x["data"]), batched=True, num_proc=8)
    tokenized_test_set = test_set.map(lambda x: dna_tokenizer(x["data"]), batched=True, num_proc=8)

    return tokenized_train_set, tokenized_valid_set, tokenized_test_set


def compute_score(output):
    pred = np.argmax(output.predictions, axis=-1)
    label = output.label_ids
    result = {
        "acc_score": accuracy_score(label, pred),
        "mcc_score": matthews_corrcoef(label, pred),
        "f1_score": f1_score(label, pred, average="macro"),
    }

    return result


class CustomCallback(TrainerCallback):
    def __init__(self, config):
        self.weight_decay = config.weight_decay

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
        if state.global_step % args.logging_steps == 0:
            optimizer = kwargs.get("optimizer")
            weight_decay = max(group["weight_decay"] for group in optimizer.param_groups)

            swanlab.log(
                data={
                    "train/weight_decay": weight_decay,
                },
                step=state.global_step,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, default="./config/finetuning_config.yaml")
    parser.add_argument("--params_path", type=str, default="./config/finetuning_params_tiny.csv")
    parser.add_argument("--eval_idx", type=int, default=54)

    args = parser.parse_args()
    args = update_args_from_yaml(args, args.config_path)
    args.model_name = args.model_dir.split("/")[-1]

    fix_seed(args.random_seed)

    df = pd.read_csv(f"{args.data_dir}/statistics.csv")
    benchmark_name = df["Benchmark"].values[args.eval_idx]
    task_name = df["Task"].values[args.eval_idx]
    label_num = df["Category"].values[args.eval_idx]

    params = pd.read_csv(args.params_path)
    params = params[(params["Benchmark"] == benchmark_name) & (params["Task"] == task_name)]
    args.train_batch_size = int(params["Batch Size"].values[0])
    args.learning_rate = [float(params["Learning Rate"].values[0]), float(params["Learning Rate"].values[0]) * 0.1]

    print("\n===================== Initializing SwanLab =====================\n")
    swanlab.init(
        project=f"Evaluate_{args.model_name}_Finetune",
        experiment_name=f"{args.eval_idx:02d}-{benchmark_name}-{task_name}",
        config=args,
        mode="online" if args.online_swanlab == 1 else "offline",
    )
    print("\n===================== SwanLab Initialized =====================\n")

    print("\n===================== Loading Model =====================\n")
    device, _, _ = get_backend()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, num_labels=label_num, trust_remote_code=True).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Parameters: {total_params:,}, Trainable Parameters: {trainable_params:,}\n")
    print(model)
    print("\n===================== Model Loaded =====================\n")

    print("\n===================== Constructing Dataset =====================\n")
    dataset = load_dataset("parquet", data_files={
        "train": f"{args.data_dir}/{benchmark_name}/{task_name}/test.parquet",
        "test": f"{args.data_dir}/{benchmark_name}/{task_name}/test.parquet",
    })
    split = dataset["train"].train_test_split(test_size=0.1, seed=args.random_seed)
    train_dataset, valid_dataset, test_dataset = split["train"], split["test"], dataset["test"]
    tokenized_train_dataset, tokenized_valid_dataset, tokenized_test_dataset = process_data(
        train_dataset, valid_dataset, test_dataset, tokenizer)
    print("\n===================== Dataset Constructed =====================\n")

    print("\n===================== Constructing Trainer =====================\n")
    log_path = f"./checkpoint/eval/{args.model_name}/{benchmark_name}/{task_name}/"
    shutil.rmtree(log_path) if os.path.exists(log_path) else None

    train_args = TrainingArguments(
        output_dir=log_path,
        num_train_epochs=args.train_epoch,

        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.train_accumulation_step,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_accumulation_steps=args.eval_accumulation_step,

        logging_strategy="epoch",
        eval_strategy="epoch",
        eval_on_start=True,
        save_strategy="epoch",
        save_total_limit=2,
        overwrite_output_dir=True,

        load_best_model_at_end=True,
        metric_for_best_model="mcc_score",

        dataloader_num_workers=4,
        dataloader_prefetch_factor=4,
        dataloader_drop_last=False,
        dataloader_persistent_workers=True,

        learning_rate=args.learning_rate[0],
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": args.learning_rate[1]},
        weight_decay=args.weight_decay[0],
        warmup_ratio=args.warmup_ratio,
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
        callbacks=[
            CustomCallback(args),
            EarlyStoppingCallback(early_stopping_patience=args.stop_patience),
        ]
    )
    print("Trainer ready for training.\n")
    print("===================== Trainer Constructed =====================\n")

    print("\n===================== Start Training =====================\n")
    trainer.train()
    print("\n===================== Training Finished =====================\n")

    print("\n===================== Evaluating Model =====================\n")
    res = trainer.predict(tokenized_test_dataset)
    acc, mcc, f1 = res.metrics["test_acc_score"], res.metrics["test_mcc_score"], res.metrics["test_f1_score"]
    print(f"\nTask {task_name} || ACC Score: {acc:.4f} || MCC Score: {mcc:.4f} || F1 Score: {f1:.4f}\n")

    swanlab.finish()
    print("\n===================== Task Finished =====================\n")
