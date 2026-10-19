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
from data_utils import *
from models import *
import argparse
import warnings
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
from opacus import PrivacyEngine
from opacus.validators.module_validator import ModuleValidator
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.nist_score import corpus_nist
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
import torch
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup
from transformers import GPT2LMHeadModel
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments
from data_utils import *
import os
from copy import deepcopy
from diagnostics import capture_lora_state, lora_state_statistics

# Disable wandb completely
os.environ["WANDB_MODE"] = "disabled"

def train_client(model, dataloader, args, round_index=None, client_index=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(dataloader) * args.local_epochs
    num_warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    learning_rates_used = []
    optimizer_step_calls = 0
    model.train()
    for epoch in range(args.local_epochs):
        for step, data in enumerate(tqdm(dataloader)):
            data = {k: v.to(device) for k, v in data.items()}

            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(**data)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                learning_rates_used.append(float(optimizer.param_groups[0]["lr"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**data)
                loss = outputs.loss
                loss.backward()
                learning_rates_used.append(float(optimizer.param_groups[0]["lr"]))
                optimizer.step()

            optimizer_step_calls += 1

            scheduler.step()
            optimizer.zero_grad()

    if client_index == 0:
        round_label = round_index if round_index is not None else "unknown"
        print(f"[DEBUG Round {round_label}] Learning rate configured = {args.lr:.12e}")
        if learning_rates_used:
            print(
                f"[DEBUG Round {round_label}] Client 0 optimizer-step learning rate "
                f"first/min/max/last = {learning_rates_used[0]:.12e}/"
                f"{min(learning_rates_used):.12e}/{max(learning_rates_used):.12e}/"
                f"{learning_rates_used[-1]:.12e}"
            )
        else:
            print(f"[ERROR] No optimizer step executed for client 0 in round {round_label}.")
        print(
            f"[DEBUG Round {round_label}] Client 0 optimizer/scaler step calls = "
            f"{optimizer_step_calls}"
        )
        if not all(np.isfinite(value) for value in learning_rates_used):
            print(
                f"[ERROR] NaN/Inf learning rate detected for client 0 in "
                f"round {round_label}."
            )

    model.cpu()
    del optimizer
    del scheduler
    del scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model.state_dict()


def calculate_metrics(all_true_labels, all_predictions, task):
    if task == "cola":
        return accuracy_score(all_true_labels, all_predictions), matthews_corrcoef(
            all_true_labels, all_predictions
        )
    elif task in ["sst2", "qnli", "rte", "wnli"]:
        return accuracy_score(all_true_labels, all_predictions), None
    elif task == "mrpc":
        return f1_score(all_true_labels, all_predictions), accuracy_score(
            all_true_labels, all_predictions
        )
    elif task == "stsb":
        return (
            pearsonr(all_true_labels, all_predictions)[0],
            spearmanr(all_true_labels, all_predictions)[0],
        )
    elif task == "qqp":
        return accuracy_score(all_true_labels, all_predictions), f1_score(
            all_true_labels, all_predictions
        )
    elif task in ["mnli_matched", "mnli_mismatched"]:
        return accuracy_score(all_true_labels, all_predictions), None
    else:
        raise ValueError(f"Unknown task: {task}")


def evaluate_global_model(
    global_model,
    dataloader,
    args,
    max_metric1,
    max_metric2,
    round_index=None,
    expected_lora_fingerprint=None,
    expected_model_id=None,
):
    round_label = round_index if round_index is not None else "unknown"
    eval_lora_state = capture_lora_state(global_model)
    eval_lora_stats = lora_state_statistics(eval_lora_state)
    eval_lora_fingerprint = eval_lora_stats["total_l2"]
    print(
        f"[DEBUG Round {round_label}] Global LoRA fingerprint at evaluation start = "
        f"{eval_lora_fingerprint:.12e}"
    )
    print(
        f"[DEBUG Round {round_label}] Evaluation model id = {id(global_model)}"
    )
    if (
        expected_lora_fingerprint is not None
        and eval_lora_fingerprint != expected_lora_fingerprint
    ):
        print(
            f"[ERROR] Evaluation LoRA fingerprint mismatch in round {round_label}: "
            f"expected {expected_lora_fingerprint:.12e}, got "
            f"{eval_lora_fingerprint:.12e}."
        )
    if expected_model_id is not None and id(global_model) != expected_model_id:
        print(
            f"[ERROR] Evaluation received a different model object in round "
            f"{round_label}: expected id {expected_model_id}, got {id(global_model)}."
        )
    if eval_lora_stats["nan_count"] or eval_lora_stats["inf_count"]:
        print(
            f"[ERROR] NaN/Inf detected in global LoRA parameters at evaluation "
            f"start: nan={eval_lora_stats['nan_count']}, "
            f"inf={eval_lora_stats['inf_count']}."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)

    global_model.eval()
    eval_loss = 0
    all_predictions = []
    all_true_labels = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = global_model(**batch)
            eval_loss += outputs.loss.detach().cpu().numpy()

            if args.task == "stsb":
                predictions = outputs.logits.squeeze().cpu().numpy()
            else:
                predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_predictions.extend(predictions)
            all_true_labels.extend(batch["labels"].cpu().numpy())

    eval_loss /= len(dataloader)

    if args.task != "stsb":
        pred0 = sum(int(x) == 0 for x in all_predictions)
        pred1 = sum(int(x) == 1 for x in all_predictions)
        true0 = sum(int(x) == 0 for x in all_true_labels)
        true1 = sum(int(x) == 1 for x in all_true_labels)

        print(
            f"[PRED DEBUG Round {round_label}] "
            f"Pred0={pred0}, Pred1={pred1}, "
            f"True0={true0}, True1={true1}"
        )

    metric1, metric2 = calculate_metrics(all_true_labels, all_predictions, args.task)

    if metric1 > max_metric1:
        max_metric1 = metric1
    if metric2 is not None and metric2 > max_metric2:
        max_metric2 = metric2

    print(f"{args.task} - Eval Loss: {eval_loss:.8f}, Metric 1: {metric1:.8f}")
    if metric2 is not None:
        print(f"{args.task} - Metric 2: {metric2:.8f}")
    print(f"{args.task} - Max Metric 1: {max_metric1:.8f}")
    if max_metric2 is not None:
        print(f"{args.task} - Max Metric 2: {max_metric2:.8f}")

    global_model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metric1, metric2, max_metric1, max_metric2


def get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps):
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def train_client_e2e(model, train_dataset, val_dataset, tokenizer, args):
    num_epochs = args.local_epochs
    per_device_train_batch_size = args.batch_size
    num_training_steps = len(train_dataset) * num_epochs // per_device_train_batch_size
    num_warmup_steps = int(0.1 * num_training_steps)

    optimizer = torch.optim.AdamW(model.parameters())

    training_args = TrainingArguments(
        output_dir="./models_trained/gpt4/dump/models/gpt2-e2e-lora_gpt4",
        overwrite_output_dir=True,
        logging_dir="./models_trained/gpt4/dump/logs/gpt2-e2e-lora_gpt4",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=num_epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        report_to="none",
        run_name="fed-lora",
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        optimizers=(
            optimizer,
            get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps),
        ),
    )

    trainer.train()
    return model.state_dict()


def gen_and_save(model, dataloader, tokenizer, args):
    device = args.device
    model.to(device)
    model.eval()
    all_predictions = []
    all_inputs = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):
            inputs = {k: v.to(device) for k, v in batch.items()}
            generated = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=inputs["input_ids"].shape[1] + 50,
                num_return_sequences=1,
                no_repeat_ngram_size=4,
                do_sample=True,
                num_beams=10,
                penalty_alpha=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
            input_seq = tokenizer.batch_decode(
                inputs["input_ids"], skip_special_tokens=True
            )
            predictions = [
                tokenizer.decode(
                    generated[i][len(inputs["input_ids"][i]) :],
                    skip_special_tokens=True,
                )
                for i in range(generated.shape[0])
            ]
            all_inputs.extend(input_seq)
            all_predictions.extend(predictions)

    return all_predictions, all_inputs


def process_lists(input_list, second_list, third_list):
    result1 = []
    result2 = []
    result3 = []
    current_group = []
    current_item = None
    second_list_index = 0

    for item in input_list:
        if item != current_item:
            if current_group:
                result1.append(current_group)
                result2.append(current_item)
                result3.append(third_list[second_list_index - 1])
            current_item = item
            current_group = [second_list[second_list_index]]
            second_list_index += 1
        else:
            if second_list_index < len(second_list):
                current_group.append(second_list[second_list_index])
                second_list_index += 1

    if current_group:
        result1.append(current_group)

    return result1, result2, result3


def evaluate_e2e_save_text(model, test_data, tokenizer, args):
    def preprocess_function2(examples):
        inputs = examples["meaning_representation"]
        targets = examples["human_reference"]
        model_inputs = [f"{input_} ->" for input_, target in zip(inputs, targets)]
        tokenized_inputs = tokenizer(
            model_inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
        tokenized_inputs["labels"][tokenized_inputs["attention_mask"] == 0] = -100
        return tokenized_inputs

    tokenized_test_dataset = test_data.map(preprocess_function2, batched=True)
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(
        ["meaning_representation", "human_reference"]
    )
    tokenized_test_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    test_dataloader = create_dataloader(tokenized_test_dataset, args)
    all_predictions, all_inputs = gen_and_save(model, test_dataloader, tokenizer, args)
    all_references = test_data[0 : len(all_predictions)]["human_reference"]

    all_references_new, all_inputs_new, all_predictions_new = process_lists(
        all_inputs, all_references, all_predictions
    )

    path_pred = args.run_dir + "/predictions.txt"
    path_ref = args.run_dir + "/refs_exact.txt"

    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)

    with open(path_pred, "w", encoding="utf-8") as file:
        for item in all_predictions_new:
            file.write(item.strip() + "\n")

    with open(path_ref, "w", encoding="utf-8") as file:
        for str_list in all_references_new:
            for item in str_list:
                file.write(item.strip() + "\n")
            file.write("\n")
