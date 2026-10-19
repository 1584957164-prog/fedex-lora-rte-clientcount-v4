import csv
import torch
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from data_utils_clientcount_v4 import *
import data_utils_clientcount_v4 as data_utils_ccv4
from models import *
import argparse
import warnings
import os
from datetime import datetime
import numpy as np
import wandb
from train_eval import *
from fed_agg import (
    aggregate_models_fedawa,
    aggregate_models_fedawa_absolute_ba_global,
    aggregate_models_fedawa_effective,
    aggregate_models_fedawa_raw_global,
    aggregate_models_ffa,
    aggregate_models_normal,
    aggregate_models_ours,
)
import json
from utils import *
from diagnostics import (
    capture_lora_state,
    lora_delta_statistics,
    lora_state_statistics,
)

parser = argparse.ArgumentParser(description="Federated Learning with LoRA")

parser.add_argument(
    "--task", type=str, default="cola", help="GLUE task to fine-tune on"
)
parser.add_argument("--model", type=str, default="roberta-base", help="Model name")
parser.add_argument("--lora_r", type=int, default=4, help="LoRA R value")
parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha value")
parser.add_argument(
    "--lora_dropout", type=float, default=0.1, help="LoRA dropout value"
)
parser.add_argument("--rslora", action="store_true", help="Use RSLoRA")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument(
    "--agg_type", type=str, default="ours", help="Type of aggregation"
)
parser.add_argument(
    "--awa_temperature", type=float, default=1.0,
    help="FedEx-AWA softmax temperature for dynamic aggregation weights"
)
parser.add_argument(
    "--awa_strength", type=float, default=0.5,
    help="FedEx-AWA strength for mixing dynamic weights with the stable prior"
)
parser.add_argument(
    "--stable_prior",
    type=str,
    choices=["uniform", "sample_size"],
    default="uniform",
    help="Stable prior mixed with FedAWA dynamic weights",
)
parser.add_argument(
    "--classifier_prior",
    type=str,
    choices=["uniform", "sample_size"],
    default="uniform",
    help="Static prior used to aggregate classifier parameters",
)
parser.add_argument(
    "--fedex_prior",
    type=str,
    choices=["uniform", "sample_size"],
    default="uniform",
    help="Static client prior used by FedEx exact adapter aggregation",
)
parser.add_argument("--num_clients", type=int, default=3, help="Number of clients")
parser.add_argument("--rounds", type=int, default=50, help="Number of rounds")
parser.add_argument(
    "--local_epochs", type=int, default=3, help="Number of local epochs"
)
parser.add_argument("--warmup_ratio", type=float, default=0.06, help="Warmup ratio")
parser.add_argument(
    "--max_seq_length", type=int, default=512, help="Maximum sequence length"
)
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--partition",
    type=str,
    choices=["random", "dirichlet"],
    default="random",
    help="Training-data partition mode",
)
parser.add_argument(
    "--dirichlet_alpha",
    type=float,
    default=0.5,
    help="Dirichlet concentration for label-skew training-data partitioning",
)
parser.add_argument(
    "--partition_seed",
    type=int,
    default=None,
    help="Independent data-partition seed (defaults to --seed)",
)
parser.add_argument(
    "--partition_only",
    action="store_true",
    help="Create/print the client partition and exit before loading a model",
)
parser.add_argument(
    "--resume",
    action="store_true",
    default=False,
    help="Resume training from checkpoints/latest_checkpoint.pth",
)

args = parser.parse_args()
if args.partition_seed is None:
    args.partition_seed = args.seed

# Disable wandb
os.environ["WANDB_MODE"] = "disabled"

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def ensure_checkpoints_dir():
    os.makedirs("checkpoints", exist_ok=True)


def save_latest_checkpoint(
    round, global_model, max_metric_1, max_metric_2, args
):
    checkpoint = {
        "round": round,
        "model_state_dict": global_model.state_dict(),
        "max_metric_1": max_metric_1,
        "max_metric_2": max_metric_2,
        "agg_type": args.agg_type,
        "task": args.task,
    }
    torch.save(checkpoint, os.path.join("checkpoints", "latest_checkpoint.pth"))


def save_best_model(round, global_model, best_metric_1, max_metric_2, args):
    checkpoint = {
        "round": round,
        "model_state_dict": global_model.state_dict(),
        "best_accuracy": best_metric_1,
        "max_metric_2": max_metric_2,
        "agg_type": args.agg_type,
        "task": args.task,
    }
    torch.save(checkpoint, os.path.join("checkpoints", "best_model.pth"))


def append_training_log(round, task, agg_type, accuracy, metric_2):
    write_header = not os.path.exists("training_log.csv")
    with open("training_log.csv", mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["round", "task", "agg_type", "accuracy", "metric_2"])
        writer.writerow([
            round,
            task,
            agg_type,
            accuracy,
            metric_2 if metric_2 is not None else "",
        ])


def load_latest_checkpoint(global_model):
    checkpoint_path = os.path.join("checkpoints", "latest_checkpoint.pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    global_model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def federated_learning(task):
    train_data, val_data, test_data = load_and_preprocess_data(task)

    # Fixed label count
    labels = [int(x) for x in train_data["labels"]]
    num_labels = len(set(labels))

    if args.task == "stsb":
        num_labels = 1

    if args.partition == "dirichlet":
        client_dataloaders = data_utils_ccv4.create_dirichlet_client_dataloaders(
            train_data, args, task
        )
    else:
        # Preserve the historical np.random.randint assignment path exactly.
        client_dataloaders = create_client_dataloaders(train_data, args)

    if args.partition_only:
        if args.partition == "random":
            print("Partition mode: random")
            for client_index, dataloader in enumerate(client_dataloaders):
                print(f"Client {client_index}: total samples = {len(dataloader.dataset)}")
        print("Partition-only mode complete; no model was loaded or trained.")
        return None

    client_sample_counts = [
        len(client_dataloader.dataset) for client_dataloader in client_dataloaders
    ]
    total_client_samples = sum(client_sample_counts)
    sample_size_prior = [
        sample_count / total_client_samples for sample_count in client_sample_counts
    ]
    adapter_prior_mode = (
        args.stable_prior
        if args.agg_type in (
            "fedawa",
            "fedawa_effective",
            "fedawa_raw_global",
            "fedawa_absolute_ba_global",
        )
        else "not applicable"
    )
    print(f"Client sample counts: {client_sample_counts}")
    print(f"Adapter stable prior mode: {adapter_prior_mode}")
    print(f"Classifier prior mode: {args.classifier_prior}")
    if args.agg_type == "ours":
        print(f"FedEx prior mode: {args.fedex_prior}")
    print(f"Sample-size prior weights: {sample_size_prior}")

    supported_agg_types = {
        "normal",
        "ours",
        "fedawa",
        "fedawa_effective",
        "fedawa_raw_global",
        "fedawa_absolute_ba_global",
        "ffa",
    }
    if args.agg_type not in supported_agg_types:
        print(
            f"[ERROR] Unsupported --agg_type {args.agg_type!r}. No aggregation "
            "branch matches this value, so the global model will not be updated. "
            "The FedEx-LoRA baseline is currently named 'ours' in this codebase."
        )

    val_dataloader = create_dataloader(val_data, args)

    max_metric_1 = 0
    max_metric_2 = 0

    if args.agg_type == "ffa":
        base_model = RobertaForSequenceClassification.from_pretrained(
            args.model, num_labels=num_labels
        )
        global_model = create_peft_FFA_model(num_labels, args, base_model=base_model)
    else:
        base_model = RobertaForSequenceClassification.from_pretrained(
            args.model, num_labels=num_labels
        )
        global_model = create_peft_model(num_labels, args, base_model=base_model)

    client_models = []

    for i in range(args.num_clients):
        if args.agg_type == "ffa":
            client_model = create_peft_FFA_model(
                num_labels, args, base_model=base_model
            )
        else:
            client_model = create_peft_model(num_labels, args, base_model=base_model)
        client_models.append(client_model)

    ensure_checkpoints_dir()
    start_round = 0
    best_metric_1 = 0

    if args.resume:
        checkpoint_path = os.path.join("checkpoints", "latest_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = load_latest_checkpoint(global_model)
            start_round = checkpoint["round"] + 1
            max_metric_1 = checkpoint.get("max_metric_1", 0)
            max_metric_2 = checkpoint.get("max_metric_2", 0)
            best_metric_1 = max_metric_1
            print(
                f"Resuming from checkpoint round {checkpoint['round']} "
                f"(starting at round {start_round})"
            )
        else:
            print(
                "Resume flag detected but checkpoints/latest_checkpoint.pth not found. "
                "Starting from scratch."
            )
    else:
        print("Starting training from scratch.")

    for round in range(start_round, args.rounds):
        args.current_round = round + 1
        print(f"Round {round + 1}/{args.rounds}")

        global_lora_before = capture_lora_state(global_model)

        client_model_state_dicts = []
        for i in range(args.num_clients):
            client_model = client_models[i]
            client_model.load_state_dict(global_model.state_dict())
            client_model_state_dict = train_client(
                client_model,
                client_dataloaders[i],
                args,
                round_index=round + 1,
                client_index=i,
            )
            client_model_state_dicts.append(client_model_state_dict)
            client_lora_after = capture_lora_state(client_model)
            client_lora_stats = lora_state_statistics(client_lora_after)
            if i == 0:
                local_delta = lora_delta_statistics(
                    global_lora_before, client_lora_after
                )
                print(f"[DEBUG Client 0 Round {round + 1}]")
                print(f"Local LoRA delta L2 = {local_delta['total_l2']:.12e}")
                print(f"Local LoRA delta max = {local_delta['total_max']:.12e}")
                print(f"Local LoRA A delta L2 = {local_delta['A_l2']:.12e}")
                print(f"Local LoRA B delta L2 = {local_delta['B_l2']:.12e}")
                print(
                    "Local LoRA finite after training = "
                    f"{'YES' if not (client_lora_stats['nan_count'] or client_lora_stats['inf_count']) else 'NO'}"
                )
            if client_lora_stats["nan_count"] or client_lora_stats["inf_count"]:
                print(
                    f"[ERROR] NaN/Inf detected after local training for client "
                    f"{i} in round {round + 1}: "
                    f"nan={client_lora_stats['nan_count']}, "
                    f"inf={client_lora_stats['inf_count']}."
                )

        if args.agg_type == "normal":
            global_model = aggregate_models_normal(global_model, client_models)
        elif args.agg_type == "ours":
            global_model = aggregate_models_ours(
                global_model,
                client_models,
                args,
                client_sample_counts=client_sample_counts,
            )
        elif args.agg_type == "fedawa":
            global_model = aggregate_models_fedawa(
                global_model,
                client_models,
                args,
                client_sample_counts=client_sample_counts,
            )
        elif args.agg_type == "fedawa_effective":
            global_model = aggregate_models_fedawa_effective(
                global_model,
                client_models,
                args,
                client_sample_counts=client_sample_counts,
            )
        elif args.agg_type == "fedawa_raw_global":
            global_model = aggregate_models_fedawa_raw_global(
                global_model,
                client_models,
                args,
                client_sample_counts=client_sample_counts,
            )
        elif args.agg_type == "fedawa_absolute_ba_global":
            global_model = aggregate_models_fedawa_absolute_ba_global(
                global_model,
                client_models,
                args,
                client_sample_counts=client_sample_counts,
            )
        elif args.agg_type == "ffa":
            global_model = aggregate_models_ffa(global_model, client_models)

        aggregation_executed = args.agg_type in supported_agg_types
        print(
            f"[DEBUG Round {round + 1}] Aggregation executed = "
            f"{'YES' if aggregation_executed else 'NO'}"
        )
        global_lora_after = capture_lora_state(global_model)
        global_before_stats = lora_state_statistics(global_lora_before)
        global_after_stats = lora_state_statistics(global_lora_after)
        global_delta = lora_delta_statistics(global_lora_before, global_lora_after)
        print(f"[DEBUG Round {round + 1}]")
        print(f"LoRA parameter delta L2 = {global_delta['total_l2']:.12e}")
        print(f"LoRA parameter delta max = {global_delta['total_max']:.12e}")
        print(f"Global LoRA norm before = {global_before_stats['total_l2']:.12e}")
        print(f"Global LoRA norm after = {global_after_stats['total_l2']:.12e}")
        print(f"A delta L2 = {global_delta['A_l2']:.12e}")
        print(f"A delta max = {global_delta['A_max']:.12e}")
        print(f"B delta L2 = {global_delta['B_l2']:.12e}")
        print(f"B delta max = {global_delta['B_max']:.12e}")
        print(
            "Global LoRA finite after aggregation = "
            f"{'YES' if not (global_after_stats['nan_count'] or global_after_stats['inf_count']) else 'NO'}"
        )
        if global_after_stats["nan_count"] or global_after_stats["inf_count"]:
            print(
                f"[ERROR] NaN/Inf detected after server aggregation in round "
                f"{round + 1}: nan={global_after_stats['nan_count']}, "
                f"inf={global_after_stats['inf_count']}."
            )

        evaluation_fingerprint = global_after_stats["total_l2"]
        evaluation_model_id = id(global_model)
        print(
            f"[DEBUG Round {round + 1}] Global LoRA fingerprint after "
            f"aggregation = {evaluation_fingerprint:.12e}"
        )
        print(
            f"[DEBUG Round {round + 1}] Aggregated global model id = "
            f"{evaluation_model_id}"
        )

        metric_1, metric_2, max_metric_1, max_metric_2 = evaluate_global_model(
            global_model,
            val_dataloader,
            args,
            max_metric_1,
            max_metric_2,
            round_index=round + 1,
            expected_lora_fingerprint=evaluation_fingerprint,
            expected_model_id=evaluation_model_id,
        )

        append_training_log(round, args.task, args.agg_type, metric_1, metric_2)
        save_latest_checkpoint(round, global_model, max_metric_1, max_metric_2, args)

        if metric_1 > best_metric_1:
            best_metric_1 = metric_1
            save_best_model(round, global_model, best_metric_1, max_metric_2, args)


if __name__ == "__main__":
    task = args.task
    model = federated_learning(task)
