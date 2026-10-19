import torch
from torch.utils.data import DataLoader
# 关键修改：拆分导入，AdamW从torch.optim导入（不再从transformers导入�?
from transformers import RobertaTokenizer, RobertaForSequenceClassification, GPT2LMHeadModel
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from peft import (
    get_peft_model,
    AdaLoraModel,
    AdaLoraConfig,
    TaskType,
    LoraConfig,
    prepare_model_for_kbit_training,
)
from data_utils import *
import argparse
from copy import deepcopy


def _build_lora_config(args):
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query", "value"],
    )


def create_peft_model(num_labels, args, base_model=None):
    if base_model is None:
        base_model = RobertaForSequenceClassification.from_pretrained(
            args.model, num_labels=num_labels
        )
    else:
        base_model = deepcopy(base_model)

    peft_config = _build_lora_config(args)
    model = get_peft_model(base_model, peft_config)

    return model


def create_peft_FFA_model(num_labels, args, base_model=None):
    if base_model is None:
        base_model = RobertaForSequenceClassification.from_pretrained(
            args.model, num_labels=num_labels
        )
    else:
        base_model = deepcopy(base_model)

    peft_config = _build_lora_config(args)
    model = get_peft_model(base_model, peft_config)

    # Make LoRA A matrices non-trainable
    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False

    return model


def create_peft_gpt2_model_e2e(args):
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Define LoRA configuration for language modeling task
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # For language modeling
        inference_mode=False,
        r=args.lora_r,  # The dimension of the low-rank update matrices
        lora_alpha=args.lora_alpha,  # The scaling factor for LoRA layers
        lora_dropout=args.lora_dropout,  # Dropout to apply to LoRA layers
        target_modules=["c_attn", "c_proj"],  # Modules to apply LoRA
    )

    # Apply LoRA to the GPT-2 model
    model = get_peft_model(model, lora_config)
    return model


def create_peft_gpt2_model_e2e_ffa(args):
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Define LoRA configuration for language modeling task
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # For language modeling
        inference_mode=False,
        r=args.lora_r,  # The dimension of the low-rank update matrices
        lora_alpha=args.lora_alpha,  # The scaling factor for LoRA layers
        lora_dropout=args.lora_dropout,  # Dropout to apply to LoRA layers
        target_modules=["c_attn", "c_proj"],  # Modules to apply LoRA
    )

    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False

    # Apply LoRA to the GPT-2 model
    model = get_peft_model(model, lora_config)
    return model