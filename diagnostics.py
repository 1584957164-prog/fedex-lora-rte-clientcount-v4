import math

import torch


def capture_lora_state(model):
    """Return a detached CPU snapshot of every LoRA A/B parameter."""
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }
    if not state:
        raise ValueError("No LoRA A/B parameters were found for diagnostics.")
    return state


def lora_state_statistics(state):
    """Summarize norms and non-finite values without flattening full tensors."""
    squared_norms = {"total": 0.0, "A": 0.0, "B": 0.0}
    nan_count = 0
    inf_count = 0
    element_count = 0

    for name, value in state.items():
        value = value.detach().to(dtype=torch.float64, device="cpu")
        group = "A" if "lora_A" in name else "B"
        squared = value.square().sum().item()
        squared_norms["total"] += squared
        squared_norms[group] += squared
        nan_count += torch.isnan(value).sum().item()
        inf_count += torch.isinf(value).sum().item()
        element_count += value.numel()

    return {
        "total_l2": math.sqrt(squared_norms["total"]),
        "A_l2": math.sqrt(squared_norms["A"]),
        "B_l2": math.sqrt(squared_norms["B"]),
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "tensor_count": len(state),
        "element_count": element_count,
    }


def lora_delta_statistics(before, after):
    """Summarize the elementwise change between two complete LoRA snapshots."""
    before_keys = set(before)
    after_keys = set(after)
    if before_keys != after_keys:
        missing = sorted(before_keys - after_keys)
        unexpected = sorted(after_keys - before_keys)
        raise ValueError(
            "LoRA diagnostic keys changed; "
            f"missing={missing}, unexpected={unexpected}"
        )

    squared_norms = {"total": 0.0, "A": 0.0, "B": 0.0}
    max_abs = {"total": 0.0, "A": 0.0, "B": 0.0}
    for name in sorted(before):
        before_value = before[name].detach().to(dtype=torch.float64, device="cpu")
        after_value = after[name].detach().to(dtype=torch.float64, device="cpu")
        if before_value.shape != after_value.shape:
            raise ValueError(
                f"LoRA diagnostic shape changed for {name}: "
                f"{tuple(before_value.shape)} -> {tuple(after_value.shape)}"
            )

        delta = after_value - before_value
        group = "A" if "lora_A" in name else "B"
        squared = delta.square().sum().item()
        squared_norms["total"] += squared
        squared_norms[group] += squared
        if delta.numel():
            tensor_max = torch.nan_to_num(
                delta.abs(), nan=float("inf"), posinf=float("inf"), neginf=float("inf")
            ).max().item()
            max_abs["total"] = max(max_abs["total"], tensor_max)
            max_abs[group] = max(max_abs[group], tensor_max)

    return {
        "total_l2": math.sqrt(squared_norms["total"]),
        "total_max": max_abs["total"],
        "A_l2": math.sqrt(squared_norms["A"]),
        "A_max": max_abs["A"],
        "B_l2": math.sqrt(squared_norms["B"]),
        "B_max": max_abs["B"],
    }
