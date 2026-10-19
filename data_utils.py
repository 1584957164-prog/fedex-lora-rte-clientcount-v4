import torch
import hashlib
import json
import math
from pathlib import Path
from torch.utils.data import DataLoader
# 关键修改1：拆分导入，AdamW从torch.optim导入
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset, Dataset as HFDataset, DatasetDict
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup,
)
# 关键修改2：删掉这里重复的AdamW导入（避免冲突）
from tqdm import tqdm
import numpy as np
import pandas as pd
from peft import get_peft_model, LoraConfig, TaskType


def _load_local_glue(task):
    """Load GLUE from locally cached Arrow files."""
    cache_root = (
        Path(__file__).resolve().parent
        / "data"
        / "glue"
        / "glue"
        / task
        / "0.0.0"
    )

    train_candidates = sorted(cache_root.glob("*/glue-train.arrow"))

    if not train_candidates:
        return None

    base = train_candidates[0].parent

    split_files = {
        "train": "glue-train.arrow",
        "validation": "glue-validation.arrow",
        "test": "glue-test.arrow",
    }

    splits = {}

    for split, filename in split_files.items():
        file_path = base / filename
        if file_path.exists():
            splits[split] = HFDataset.from_file(str(file_path))

    if "train" not in splits:
        return None

    print(f"[Data] Loading local GLUE cache: {task}")
    for split, ds in splits.items():
        print(f"[Data] {split}: {len(ds)} examples")

    return DatasetDict(splits)


def load_and_preprocess_data(task):
    glue_task = "mnli" if "mnli" in task else task

    dataset = _load_local_glue(glue_task)

    if dataset is None:
        print(f"[Data] Local cache not found for {glue_task}; trying Hugging Face Hub.")
        dataset = load_dataset("glue", glue_task)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def tokenize_function(examples):

        # Handle different input formats
        if "premise" in examples and "hypothesis" in examples:
            # MNLI and similar tasks
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        elif "question" in examples and "sentence" in examples:
            # QNLI and similar tasks
            return tokenizer(
                examples["question"],
                examples["sentence"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        elif "sentence1" in examples and "sentence2" in examples:
            # MRPC, STS-B
            return tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        elif "question1" in examples and "question2" in examples:
            # QQP
            return tokenizer(
                examples["question1"],
                examples["question2"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        elif "sentence" in examples:
            # CoLA, SST-2
            return tokenizer(
                examples["sentence"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        else:
            raise ValueError(f"Unexpected format for task {task}")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    if task == "cola":
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    elif task == "sst2":
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    elif task == "mrpc":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
    elif task == "qqp":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["question1", "question2", "idx"]
        )
    elif task == "stsb":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
    elif task == "qnli":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["question", "sentence", "idx"]
        )
    elif task == "rte":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
    elif task == "wnli":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
    elif task == "mnli_matched" or task == "mnli_mismatched" or task == "mnli":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["premise", "hypothesis", "idx"]
        )
    else:
        raise ValueError(f"Unexpected task {task}")

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    if (
        task == "cola"
        or task == "sst2"
        or task == "mrpc"
        or task == "qqp"
        or task == "stsb"
        or task == "qnli"
        or task == "rte"
        or task == "wnli"
    ):
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation"]
        test_dataset = tokenized_datasets["test"]
    elif task == "mnli_matched":
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation_matched"]
        test_dataset = tokenized_datasets["test_matched"]
    elif task == "mnli_mismatched":
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation_mismatched"]
        test_dataset = tokenized_datasets["test_mismatched"]

    return train_dataset, val_dataset, test_dataset


def create_dataloader(dataset, args):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


def create_client_dataloaders_nlg(dataset, args):
    client_data = [[] for _ in range(args.num_clients)]
    for data in dataset:
        client_idx = np.random.randint(args.num_clients)
        client_data[client_idx].append(data)
    return client_data


def create_client_dataloaders(dataset, args):
    client_data = [[] for _ in range(args.num_clients)]
    for data in dataset:
        client_idx = np.random.randint(args.num_clients)
        client_data[client_idx].append(data)
    return [
        DataLoader(cd, batch_size=args.batch_size, shuffle=True) for cd in client_data
    ]


def extract_classification_labels(dataset):
    """Read scalar classification labels from a preprocessed dataset."""
    column_names = getattr(dataset, "column_names", None)
    if column_names is not None:
        label_key = "labels" if "labels" in column_names else "label" if "label" in column_names else None
    elif len(dataset) > 0 and isinstance(dataset[0], dict):
        label_key = "labels" if "labels" in dataset[0] else "label" if "label" in dataset[0] else None
    else:
        label_key = None

    if label_key is None:
        raise ValueError("The training dataset has neither a 'labels' nor a 'label' field.")

    try:
        raw_labels = dataset[label_key]
    except (TypeError, KeyError, IndexError):
        raw_labels = [dataset[index][label_key] for index in range(len(dataset))]

    labels = []
    for index, label in enumerate(raw_labels):
        if isinstance(label, torch.Tensor):
            if label.numel() != 1:
                raise ValueError(f"Label at index {index} is not scalar.")
            label = label.item()
        elif isinstance(label, np.generic):
            label = label.item()
        if isinstance(label, float) and (not math.isfinite(label) or not label.is_integer()):
            raise ValueError(
                "Dirichlet label-skew partitioning requires discrete classification "
                f"labels; found {label!r} at index {index}."
            )
        try:
            hash(label)
        except TypeError as error:
            raise ValueError(f"Label at index {index} is not a scalar value.") from error
        labels.append(label)

    if not labels:
        raise ValueError("Cannot partition an empty training dataset.")
    return labels


def dirichlet_label_partition(
    labels, num_clients, alpha, partition_seed, max_retries=1000,
    min_client_fraction=0.0, min_label_count=0
):
    """Assign every sample once using label-wise Dirichlet proportions."""
    if num_clients <= 0:
        raise ValueError("num_clients must be greater than zero.")
    if len(labels) < num_clients:
        raise ValueError(
            f"Cannot give {num_clients} clients at least one sample from a dataset "
            f"containing only {len(labels)} samples."
        )
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as error:
        raise ValueError("dirichlet_alpha must be a finite positive number.") from error
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("dirichlet_alpha must be a finite positive number.")
    if max_retries <= 0:
        raise ValueError("max_retries must be greater than zero.")

    label_indices = {}
    for sample_index, label in enumerate(labels):
        label_indices.setdefault(label, []).append(sample_index)

    # This RNG is private to data partitioning. Model construction, training,
    # aggregation selection, and any global NumPy calls cannot alter it.
    rng = np.random.default_rng(partition_seed)
    concentration = np.full(num_clients, alpha, dtype=np.float64)

    for retry_count in range(max_retries):
        client_indices = [[] for _ in range(num_clients)]
        for indices_for_label in label_indices.values():
            shuffled_indices = np.asarray(indices_for_label, dtype=np.int64).copy()
            rng.shuffle(shuffled_indices)
            proportions = rng.dirichlet(concentration)
            split_points = (
                np.cumsum(proportions)[:-1] * len(shuffled_indices)
            ).astype(np.int64)
            for client_index, split in enumerate(
                np.split(shuffled_indices, split_points)
            ):
                client_indices[client_index].extend(int(index) for index in split)

        for indices in client_indices:
            rng.shuffle(indices)
        min_client_size = math.ceil(len(labels) * min_client_fraction)
        size_ok = all(len(indices) >= min_client_size for indices in client_indices)

        label_ok = True
        if min_label_count > 0:
            label_values = list(label_indices.keys())
            for indices in client_indices:
                counts = {label: 0 for label in label_values}
                for sample_index in indices:
                    counts[labels[sample_index]] += 1
                total = len(indices)
                if any(
                    counts[label] < min_label_count
                    for label in label_values
                ):
                    label_ok = False
                    break

        if size_ok and label_ok:
            return client_indices, retry_count

    raise RuntimeError(
        "Dirichlet partitioning failed to give every client at least one sample "
        f"after {max_retries} attempts. Increase dirichlet_alpha, reduce "
        "num_clients, or use a larger training dataset."
    )


def validate_client_partition(client_indices, dataset_size):
    """Ensure a partition is non-empty, in range, exhaustive, and disjoint."""
    if not client_indices:
        raise ValueError("Partition contains no clients.")
    empty_clients = [index for index, indices in enumerate(client_indices) if not indices]
    if empty_clients:
        raise ValueError(f"Partition contains empty clients: {empty_clients}")
    flattened = [index for indices in client_indices for index in indices]
    if any(not isinstance(index, (int, np.integer)) for index in flattened):
        raise ValueError("Partition indices must be integers.")
    if any(index < 0 or index >= dataset_size for index in flattened):
        raise ValueError("Partition contains an out-of-range sample index.")
    if len(flattened) != dataset_size or len(set(flattened)) != dataset_size:
        raise ValueError(
            "Partition must contain every training sample exactly once, with no "
            "omissions or duplicates."
        )


def compute_partition_statistics(labels, client_indices):
    """Return per-client label distributions and Shannon entropy in nats."""
    label_values = list(dict.fromkeys(labels))
    statistics = []
    for indices in client_indices:
        counts = {label: 0 for label in label_values}
        for sample_index in indices:
            counts[labels[sample_index]] += 1
        total = len(indices)
        distribution = {label: count / total for label, count in counts.items()}
        entropy = -sum(
            probability * math.log(probability)
            for probability in distribution.values()
            if probability > 0
        )
        statistics.append(
            {
                "total": total,
                "counts": counts,
                "distribution": distribution,
                "entropy": entropy,
            }
        )
    return statistics


def _label_fingerprint(labels):
    serialized = json.dumps(
        labels, ensure_ascii=False, separators=(",", ":"), default=repr
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _manifest_path(task, alpha, num_clients, partition_seed, manifest_dir):
    safe_task = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(task)
    )
    alpha_text = format(float(alpha), ".12g")
    filename = (
        f"{safe_task}_dirichlet_alpha{alpha_text}_clients{num_clients}_"
        f"seed{partition_seed}.json"
    )
    return Path(manifest_dir) / filename


def get_or_create_dirichlet_partition(
    dataset, args, task, manifest_dir="partition_manifests", max_retries=1000
):
    """Load a matching manifest, or deterministically create and save one."""
    labels = extract_classification_labels(dataset)

    constrained_rte = str(task).lower() == "rte"
    if constrained_rte:
        # Client-count sensitivity protocol:
        # keep the same label-wise Dirichlet sampling while requiring only
        # non-empty client datasets for K in {3, 5, 10}.
        min_client_fraction = 1.0 / len(labels)
        min_label_count = 0
        manifest_dir = "partition_manifests_rte_clientcount_v1"
    else:
        min_client_fraction = 0.0
        min_label_count = 0

    alpha = float(args.dirichlet_alpha)
    partition_seed = int(args.partition_seed)
    manifest_path = _manifest_path(
        task, alpha, args.num_clients, partition_seed, manifest_dir
    )

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        expected_metadata = {
            "task": task,
            "partition": "dirichlet",
            "alpha": alpha,
            "num_clients": args.num_clients,
            "partition_seed": partition_seed,
            "dataset_size": len(labels),
            "label_fingerprint": _label_fingerprint(labels),
        }
        mismatches = [
            key
            for key, expected_value in expected_metadata.items()
            if manifest.get(key) != expected_value
        ]
        if mismatches:
            raise ValueError(
                f"Existing partition manifest {manifest_path} does not match the "
                f"current experiment for fields: {mismatches}."
            )
        try:
            client_indices = [
                manifest["client_indices"][f"client_{client_index}"]
                for client_index in range(args.num_clients)
            ]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"Existing partition manifest {manifest_path} is incomplete."
            ) from error
        validate_client_partition(client_indices, len(labels))
        retry_count = int(manifest.get("retry_count", 0))
        source = "loaded"
    else:
        client_indices, retry_count = dirichlet_label_partition(
            labels,
            args.num_clients,
            alpha,
            partition_seed,
            max_retries=max_retries,
            min_client_fraction=min_client_fraction,
            min_label_count=min_label_count,
        )
        validate_client_partition(client_indices, len(labels))
        manifest = {
            "task": task,
            "partition": "dirichlet",
            "alpha": alpha,
            "num_clients": args.num_clients,
            "partition_seed": partition_seed,
            "dataset_size": len(labels),
            "label_fingerprint": _label_fingerprint(labels),
            "retry_count": retry_count,
            "client_indices": {
                f"client_{client_index}": indices
                for client_index, indices in enumerate(client_indices)
            },
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
        temporary_path.replace(manifest_path)
        source = "created"

    return client_indices, labels, manifest_path, retry_count, source


def print_dirichlet_partition_summary(
    labels, client_indices, alpha, partition_seed, retry_count, manifest_path, source
):
    statistics = compute_partition_statistics(labels, client_indices)
    print("Partition mode: dirichlet")
    print(f"Dirichlet alpha: {float(alpha):g}")
    print(f"Partition seed: {partition_seed}")
    print(f"Partition manifest: {manifest_path} ({source})")
    print(f"Dirichlet retries: {retry_count}")
    for client_index, client_statistics in enumerate(statistics):
        print(f"Client {client_index}:")
        print(f"total samples = {client_statistics['total']}")
        for label, count in client_statistics["counts"].items():
            percentage = 100.0 * client_statistics["distribution"][label]
            print(f"label {label} = {count} ({percentage:.2f}%)")
        print(f"label entropy = {client_statistics['entropy']:.6f} nats")
    mean_entropy = sum(item["entropy"] for item in statistics) / len(statistics)
    print(f"Mean client label entropy = {mean_entropy:.6f} nats")


def create_dirichlet_client_dataloaders(dataset, args, task):
    client_indices, labels, manifest_path, retry_count, source = (
        get_or_create_dirichlet_partition(dataset, args, task)
    )
    print_dirichlet_partition_summary(
        labels,
        client_indices,
        args.dirichlet_alpha,
        args.partition_seed,
        retry_count,
        manifest_path,
        source,
    )
    return [
        DataLoader(
            Subset(dataset, indices), batch_size=args.batch_size, shuffle=True
        )
        for indices in client_indices
    ]


def create_e2e_data():
    def preprocess_function(examples):
        inputs = examples["meaning_representation"]
        targets = examples["human_reference"]

        # Combine the input-output pair into a single text
        model_inputs = [
            f"{input_} -> {target} <|endoftext|>"
            for input_, target in zip(inputs, targets)
        ]
        only_inputs = [f"{input_} ->" for input_, target in zip(inputs, targets)]

        # Tokenize the combined inputs
        tokenized_inputs = tokenizer(
            model_inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokenized_only_inputs = tokenizer(
            only_inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Labels are the same as input_ids but shift them for next-token prediction
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()

        # Set the labels to -100 where attention mask is 0 (this will ignore padding in loss computation)
        tokenized_inputs["labels"][tokenized_inputs["attention_mask"] == 0] = -100
        # set the labels to -100 where meaning representation input ids are present
        tokenized_inputs["labels"][tokenized_only_inputs["attention_mask"] == 1] = -100

        return tokenized_inputs

    dataset = load_dataset("tuetschek/e2e_nlg")
    from transformers import GPT2Tokenizer

    # Load the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = (
        tokenizer.eos_token
    )  # GPT-2 doesn't have a pad token, so we set it to the eos token
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    return (
        tokenized_datasets["train"],
        tokenized_datasets["validation"],
        tokenized_datasets["test"],
        tokenizer,
    )
