import torch
import numpy as np
import re
import time
import warnings
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType


_LORA_FACTOR_KEY = re.compile(
    r"^(?P<prefix>.+)\.lora_(?P<factor>[AB])"
    r"(?:\.(?P<adapter>[^.]+))?\.weight$"
)


def effective_inner(A1, B1, A2, B2, scaling1=1.0, scaling2=1.0):
    """Return ``<scaling1 * B1 A1, scaling2 * B2 A2>_F``.

    Only rank-sized intermediate matrices are constructed.  In particular,
    neither ``B1 @ A1`` nor ``B2 @ A2`` is materialized.
    """
    matrices = {"A1": A1, "B1": B1, "A2": A2, "B2": B2}
    for matrix_name, matrix in matrices.items():
        if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2:
            raise ValueError(f"{matrix_name} must be a two-dimensional tensor.")

    if B1.shape[1] != A1.shape[0] or B2.shape[1] != A2.shape[0]:
        raise ValueError("Each LoRA B rank dimension must match its A rank dimension.")
    if B1.shape[0] != B2.shape[0] or A1.shape[1] != A2.shape[1]:
        raise ValueError("The two effective LoRA updates must have the same shape.")
    if len({A1.device, B1.device, A2.device, B2.device}) != 1:
        raise ValueError("All LoRA factors must be on the same device.")

    A1 = A1.float()
    B1 = B1.float()
    A2 = A2.float()
    B2 = B2.float()
    b_gram = B1.transpose(0, 1) @ B2
    a_gram = A2 @ A1.transpose(0, 1)
    inner = torch.sum(b_gram * a_gram.transpose(0, 1))
    return inner * float(scaling1) * float(scaling2)


def effective_delta_inner(
    A1,
    B1,
    A2,
    B2,
    Ag,
    Bg,
    scaling1=1.0,
    scaling2=1.0,
    global_scaling=1.0,
):
    """Return the inner product of two functional deltas from a global adapter."""
    return (
        effective_inner(A1, B1, A2, B2, scaling1, scaling2)
        - effective_inner(A1, B1, Ag, Bg, scaling1, global_scaling)
        - effective_inner(Ag, Bg, A2, B2, global_scaling, scaling2)
        + effective_inner(Ag, Bg, Ag, Bg, global_scaling, global_scaling)
    )


def effective_delta_cosine(
    A1,
    B1,
    A2,
    B2,
    Ag,
    Bg,
    scaling1=1.0,
    scaling2=1.0,
    global_scaling=1.0,
    eps=None,
):
    """Compute a stable single-layer cosine between two functional deltas."""
    numerator = effective_delta_inner(
        A1, B1, A2, B2, Ag, Bg, scaling1, scaling2, global_scaling
    )
    norm1_squared = effective_delta_inner(
        A1, B1, A1, B1, Ag, Bg, scaling1, scaling1, global_scaling
    ).clamp_min(0.0)
    norm2_squared = effective_delta_inner(
        A2, B2, A2, B2, Ag, Bg, scaling2, scaling2, global_scaling
    ).clamp_min(0.0)
    if eps is None:
        eps = torch.finfo(torch.float32).eps
    denominator = torch.sqrt(norm1_squared) * torch.sqrt(norm2_squared)
    if (
        not torch.isfinite(numerator)
        or not torch.isfinite(denominator)
        or denominator <= eps
    ):
        return torch.zeros((), dtype=torch.float32, device=numerator.device)
    return torch.clamp(numerator / (denominator + eps), min=-1.0, max=1.0)


def _pair_lora_state_dict_keys(state_dict):
    """Pair PEFT LoRA A/B keys by module prefix and adapter name."""
    factors = {}
    unsupported_keys = []
    for key in state_dict:
        if ".lora_A" not in key and ".lora_B" not in key:
            continue
        match = _LORA_FACTOR_KEY.match(key)
        if match is None:
            unsupported_keys.append(key)
            continue
        pair_id = (match.group("prefix"), match.group("adapter"))
        factor = match.group("factor")
        if factor in factors.setdefault(pair_id, {}):
            raise ValueError(f"Duplicate LoRA {factor} factor for {pair_id}: {key}")
        factors[pair_id][factor] = key

    if unsupported_keys:
        raise ValueError(
            "Unsupported LoRA factor key format: " + ", ".join(unsupported_keys)
        )

    missing = [
        f"{prefix} (adapter={adapter!r}, missing {sorted({'A', 'B'} - set(pair))})"
        for (prefix, adapter), pair in factors.items()
        if set(pair) != {"A", "B"}
    ]
    if missing:
        raise ValueError("Unpaired LoRA state_dict keys: " + "; ".join(missing))
    if not factors:
        raise ValueError("No paired LoRA A/B weights were found in the model state_dict.")

    return [
        {
            "prefix": prefix,
            "adapter": adapter,
            "A_key": factors[(prefix, adapter)]["A"],
            "B_key": factors[(prefix, adapter)]["B"],
        }
        for prefix, adapter in sorted(factors, key=lambda item: (item[0], item[1] or ""))
    ]


def _get_lora_scaling(model, prefix, adapter):
    modules = dict(model.named_modules())
    if prefix not in modules:
        raise ValueError(f"LoRA module {prefix!r} was not found in named_modules().")
    scaling = getattr(modules[prefix], "scaling", None)
    if isinstance(scaling, dict):
        if adapter not in scaling:
            raise ValueError(
                f"LoRA scaling for adapter {adapter!r} was not found in module {prefix!r}."
            )
        scaling = scaling[adapter]
    scaling = _safe_float(scaling, np.nan)
    if not np.isfinite(scaling):
        raise ValueError(f"Invalid LoRA scaling for module {prefix!r}: {scaling!r}")
    return scaling


def _collect_lora_factors(global_model, client_models, device):
    global_state = global_model.state_dict()
    client_states = [model.state_dict() for model in client_models]
    pairs = _pair_lora_state_dict_keys(global_state)
    expected_keys = {(pair["A_key"], pair["B_key"]) for pair in pairs}

    for client_index, client_state in enumerate(client_states):
        client_pairs = _pair_lora_state_dict_keys(client_state)
        client_keys = {(pair["A_key"], pair["B_key"]) for pair in client_pairs}
        if client_keys != expected_keys:
            raise ValueError(
                f"Client {client_index + 1} LoRA A/B keys do not match the global model."
            )

    layers = []
    for pair in pairs:
        prefix = pair["prefix"]
        adapter = pair["adapter"]
        scaling = _get_lora_scaling(global_model, prefix, adapter)
        for client_index, client_model in enumerate(client_models):
            client_scaling = _get_lora_scaling(client_model, prefix, adapter)
            if not np.isclose(client_scaling, scaling):
                raise ValueError(
                    f"Client {client_index + 1} scaling for {prefix!r} "
                    f"({client_scaling}) differs from global scaling ({scaling})."
                )

        A_key = pair["A_key"]
        B_key = pair["B_key"]
        global_A = global_state[A_key].detach().to(device=device, dtype=torch.float32)
        global_B = global_state[B_key].detach().to(device=device, dtype=torch.float32)
        client_A = torch.stack(
            [state[A_key].detach().to(device=device, dtype=torch.float32) for state in client_states]
        )
        client_B = torch.stack(
            [state[B_key].detach().to(device=device, dtype=torch.float32) for state in client_states]
        )
        layers.append(
            {
                **pair,
                "scaling": scaling,
                "global_A": global_A,
                "global_B": global_B,
                "client_A": client_A,
                "client_B": client_B,
            }
        )
    return layers


def _stable_pairwise_cosines(norms_squared, inner_product, num_clients):
    device = norms_squared.device
    eps = torch.finfo(torch.float32).eps
    norms_squared = torch.nan_to_num(
        norms_squared.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    norms = torch.sqrt(norms_squared)
    similarities = torch.zeros(
        (num_clients, num_clients), dtype=torch.float32, device=device
    )
    similarities.diagonal().copy_((norms > eps).float())

    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            numerator = inner_product(i, j)
            denominator = norms[i] * norms[j]
            if (
                not torch.isfinite(numerator)
                or not torch.isfinite(denominator)
                or denominator <= eps
            ):
                cosine = torch.zeros((), dtype=torch.float32, device=device)
            else:
                cosine = torch.clamp(
                    numerator.float() / (denominator + eps), min=-1.0, max=1.0
                )
            similarities[i, j] = cosine
            similarities[j, i] = cosine

    if num_clients > 1:
        scores = (similarities.sum(dim=1) - similarities.diagonal()) / (
            num_clients - 1
        )
        pair_indices = torch.triu_indices(num_clients, num_clients, offset=1, device=device)
        mean_similarity = similarities[pair_indices[0], pair_indices[1]].mean()
    else:
        scores = torch.zeros((1,), dtype=torch.float32, device=device)
        mean_similarity = torch.zeros((), dtype=torch.float32, device=device)
    return similarities, scores, mean_similarity


def compute_effective_update_pairwise_cosines(layers):
    """Compute one global cosine over all low-rank functional-delta layers."""
    if not layers:
        raise ValueError("At least one LoRA layer is required.")
    num_clients = layers[0]["client_A"].shape[0]
    device = layers[0]["client_A"].device
    norms_squared = torch.zeros(num_clients, dtype=torch.float32, device=device)

    for client_index in range(num_clients):
        for layer in layers:
            norms_squared[client_index] += effective_delta_inner(
                layer["client_A"][client_index],
                layer["client_B"][client_index],
                layer["client_A"][client_index],
                layer["client_B"][client_index],
                layer["global_A"],
                layer["global_B"],
                layer["scaling"],
                layer["scaling"],
                layer["scaling"],
            )
    norms_squared.clamp_min_(0.0)

    def model_inner(i, j):
        result = torch.zeros((), dtype=torch.float32, device=device)
        for layer in layers:
            result += effective_delta_inner(
                layer["client_A"][i],
                layer["client_B"][i],
                layer["client_A"][j],
                layer["client_B"][j],
                layer["global_A"],
                layer["global_B"],
                layer["scaling"],
                layer["scaling"],
                layer["scaling"],
            )
        return result

    return _stable_pairwise_cosines(norms_squared, model_inner, num_clients)


def compute_absolute_ba_pairwise_cosines(layers):
    """Compute one global cosine over all scaled absolute LoRA updates."""
    if not layers:
        raise ValueError("At least one LoRA layer is required.")
    num_clients = layers[0]["client_A"].shape[0]
    device = layers[0]["client_A"].device
    norms_squared = torch.zeros(num_clients, dtype=torch.float32, device=device)

    for client_index in range(num_clients):
        for layer in layers:
            norms_squared[client_index] += effective_inner(
                layer["client_A"][client_index],
                layer["client_B"][client_index],
                layer["client_A"][client_index],
                layer["client_B"][client_index],
                layer["scaling"],
                layer["scaling"],
            )
    norms_squared.clamp_min_(0.0)

    def model_inner(i, j):
        result = torch.zeros((), dtype=torch.float32, device=device)
        for layer in layers:
            result += effective_inner(
                layer["client_A"][i],
                layer["client_B"][i],
                layer["client_A"][j],
                layer["client_B"][j],
                layer["scaling"],
                layer["scaling"],
            )
        return result

    return _stable_pairwise_cosines(norms_squared, model_inner, num_clients)


def compute_raw_factor_pairwise_cosines(layers):
    """Diagnostic cosine of concatenated raw ``A-A_g`` and ``B-B_g`` factors."""
    if not layers:
        raise ValueError("At least one LoRA layer is required.")
    num_clients = layers[0]["client_A"].shape[0]
    device = layers[0]["client_A"].device
    norms_squared = torch.zeros(num_clients, dtype=torch.float32, device=device)
    for client_index in range(num_clients):
        for layer in layers:
            delta_A = layer["client_A"][client_index] - layer["global_A"]
            delta_B = layer["client_B"][client_index] - layer["global_B"]
            norms_squared[client_index] += torch.sum(delta_A * delta_A)
            norms_squared[client_index] += torch.sum(delta_B * delta_B)
    norms_squared.clamp_min_(0.0)

    def model_inner(i, j):
        result = torch.zeros((), dtype=torch.float32, device=device)
        for layer in layers:
            delta_A_i = layer["client_A"][i] - layer["global_A"]
            delta_B_i = layer["client_B"][i] - layer["global_B"]
            delta_A_j = layer["client_A"][j] - layer["global_A"]
            delta_B_j = layer["client_B"][j] - layer["global_B"]
            result += torch.sum(delta_A_i * delta_A_j)
            result += torch.sum(delta_B_i * delta_B_j)
        return result

    return _stable_pairwise_cosines(norms_squared, model_inner, num_clients)


def _dynamic_weights_from_similarity_scores(similarity_scores, temperature=1.0):
    num_clients = similarity_scores.shape[0]
    if num_clients == 0:
        raise ValueError("At least one client is required for aggregation.")
    similarity_scores = similarity_scores.float()
    uniform_weights = torch.full_like(similarity_scores, 1.0 / num_clients)
    if not torch.isfinite(similarity_scores).all():
        return uniform_weights
    temperature = _safe_float(temperature, 1.0)
    if temperature <= 0:
        temperature = 1.0
    weights = torch.softmax(similarity_scores / temperature, dim=0)
    if not torch.isfinite(weights).all():
        return uniform_weights
    return weights


def _all_factors_finite(layers):
    return all(
        torch.isfinite(layer[key]).all()
        for layer in layers
        for key in ("global_A", "global_B", "client_A", "client_B")
    )


def _format_weights(weights):
    return "[" + ", ".join(f"{value:.8f}" for value in weights.detach().cpu().tolist()) + "]"

def aggregate_models_normal(global_model, client_models):
    global_dict = global_model.state_dict()
    client_state_dicts = [model.state_dict() for model in client_models]
    
    for k in global_dict.keys():
        if "lora" in k or "classifier" in k:
            global_dict[k] = torch.stack(
                [client_state_dicts[i][k].float() for i in range(len(client_state_dicts))], 0
            ).mean(0)

    global_model.load_state_dict(global_dict)
    return global_model

def aggregate_models_ffa(global_model, client_models):
    global_dict = global_model.state_dict()
    client_state_dicts = [model.state_dict() for model in client_models]
    
    for k in global_dict.keys():
        if "lora_B" in k or "classifier" in k:
            global_dict[k] = torch.stack(
                [client_state_dicts[i][k].float() for i in range(len(client_state_dicts))], 0
            ).mean(0)

    global_model.load_state_dict(global_dict)
    return global_model

def compute_dynamic_weights_from_lora(lora_A_weights, lora_B_weights, temperature=1.0):
    """Compute FedEx-AWA client weights from LoRA update similarity."""
    num_clients = lora_A_weights.shape[0]
    device = lora_A_weights.device

    if num_clients == 0:
        raise ValueError("At least one client is required for aggregation.")

    # Work in float32 to make the cosine and softmax calculations stable even
    # when the model weights use a lower-precision dtype.
    lora_A_weights = lora_A_weights.float()
    lora_B_weights = lora_B_weights.float()
    uniform_weights = torch.full(
        (num_clients,), 1.0 / num_clients, device=device, dtype=lora_A_weights.dtype
    )

    if not torch.isfinite(lora_A_weights).all() or not torch.isfinite(lora_B_weights).all():
        return uniform_weights

    client_vectors = torch.matmul(lora_B_weights, lora_A_weights).flatten(start_dim=1)
    global_vector = client_vectors.mean(dim=0)
    eps = torch.finfo(client_vectors.dtype).eps
    similarities = torch.nn.functional.cosine_similarity(
        client_vectors, global_vector.unsqueeze(0), dim=1, eps=eps
    )

    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        temperature = 1.0
    if not np.isfinite(temperature) or temperature <= 0:
        temperature = 1.0

    weights = torch.softmax(similarities / temperature, dim=0)
    if not torch.isfinite(weights).all():
        return uniform_weights
    return weights

def _safe_float(value, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return value


def compute_stable_prior_weights(
    reference_weights, stable_prior="uniform", client_sample_counts=None
):
    """Build a normalized uniform or sample-size stable aggregation prior."""
    if not isinstance(reference_weights, torch.Tensor) or reference_weights.ndim != 1:
        raise ValueError("reference_weights must be a one-dimensional tensor.")
    num_clients = reference_weights.numel()
    if num_clients == 0:
        raise ValueError("At least one client is required for aggregation.")

    if stable_prior == "uniform":
        return torch.ones_like(reference_weights) / num_clients
    if stable_prior != "sample_size":
        raise ValueError(
            f"Unknown stable_prior {stable_prior!r}; expected 'uniform' or "
            "'sample_size'."
        )
    if client_sample_counts is None:
        raise ValueError(
            "client_sample_counts are required when stable_prior='sample_size'."
        )

    sample_counts = torch.as_tensor(
        client_sample_counts,
        dtype=reference_weights.dtype,
        device=reference_weights.device,
    )
    if sample_counts.ndim != 1 or sample_counts.numel() != num_clients:
        raise ValueError(
            "client_sample_counts must contain exactly one count per client."
        )
    if not torch.isfinite(sample_counts).all() or (sample_counts < 0).any():
        raise ValueError("Client sample counts must be finite and non-negative.")
    total_samples = sample_counts.sum()
    if total_samples <= 0:
        raise ValueError("The total number of client training samples must be positive.")
    return sample_counts / total_samples



def save_mechanism_data(name, data, round_id):
    import os, csv
    os.makedirs(os.path.join("mechanism_logs", name), exist_ok=True)

    with open(
        os.path.join("mechanism_logs", name, f"round_{round_id}.csv"),
        "w",
        newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["item","value"])
        for k,v in data.items():
            writer.writerow([k,v])

def mix_stable_weights(dynamic_weights, prior_weights, awa_strength):
    """Mix dynamic FedAWA weights with a stable prior and normalize the result."""
    if dynamic_weights.shape != prior_weights.shape:
        raise ValueError("Dynamic weights and prior weights must have the same shape.")
    awa_strength = _safe_float(awa_strength, 0.5)
    final_weights = (
        awa_strength * dynamic_weights + (1 - awa_strength) * prior_weights
    )
    return final_weights / final_weights.sum()


def _format_sample_counts(client_sample_counts):
    if client_sample_counts is None:
        return "not provided"
    return "[" + ", ".join(str(int(count)) for count in client_sample_counts) + "]"


def aggregate_classifier_parameters(
    global_dict,
    client_state_dicts,
    classifier_prior="uniform",
    client_sample_counts=None,
):
    """Aggregate classifier state using only the selected static prior."""
    num_clients = len(client_state_dicts)
    reference_weights = torch.empty(num_clients, dtype=torch.float32)
    classifier_weights = compute_stable_prior_weights(
        reference_weights, classifier_prior, client_sample_counts
    )

    for key in global_dict.keys():
        if "classifier" not in key:
            continue
        client_values = torch.stack(
            [client_state_dicts[i][key].float() for i in range(num_clients)], 0
        )
        if classifier_prior == "uniform":
            # Preserve the legacy arithmetic-mean operation exactly.
            global_dict[key] = client_values.mean(0)
        else:
            weight_shape = (num_clients,) + (1,) * (client_values.ndim - 1)
            weights = classifier_weights.to(client_values.device).reshape(weight_shape)
            global_dict[key] = torch.sum(weights * client_values, dim=0)
    return classifier_weights


def compute_fedex_exact_components(
    lora_A_weights, lora_B_weights, client_weights=None
):
    """Return weighted FedEx target, averaged factors, and exact residual."""
    if lora_A_weights.ndim != 3 or lora_B_weights.ndim != 3:
        raise ValueError("Stacked LoRA A/B weights must be three-dimensional.")
    num_clients = lora_A_weights.shape[0]
    if num_clients == 0 or lora_B_weights.shape[0] != num_clients:
        raise ValueError("LoRA A/B weights must contain the same non-zero clients.")

    if client_weights is None:
        # Preserve the legacy uniform FedEx operations exactly.
        M = sum(
            lora_B_weights[i] @ lora_A_weights[i] for i in range(num_clients)
        ) / num_clients
        lora_A_avg = lora_A_weights.mean(0)
        lora_B_avg = lora_B_weights.mean(0)
    else:
        client_weights = client_weights.to(
            device=lora_A_weights.device, dtype=lora_A_weights.dtype
        )
        if client_weights.ndim != 1 or client_weights.numel() != num_clients:
            raise ValueError("client_weights must contain one value per LoRA client.")
        if not torch.isfinite(client_weights).all() or (client_weights < 0).any():
            raise ValueError("FedEx client weights must be finite and non-negative.")
        if not torch.isclose(
            client_weights.sum(),
            torch.ones((), device=client_weights.device, dtype=client_weights.dtype),
        ):
            raise ValueError("FedEx client weights must sum to one.")
        M = sum(
            client_weights[i] * (lora_B_weights[i] @ lora_A_weights[i])
            for i in range(num_clients)
        )
        lora_A_avg = sum(
            client_weights[i] * lora_A_weights[i] for i in range(num_clients)
        )
        lora_B_avg = sum(
            client_weights[i] * lora_B_weights[i] for i in range(num_clients)
        )

    residue = M - lora_B_avg @ lora_A_avg
    return M, lora_A_avg, lora_B_avg, residue

def aggregate_models_ours(
    global_model, client_models, args, client_sample_counts=None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_model = global_model.to(device)
    global_dict = global_model.state_dict()
    client_state_dicts = [model.state_dict() for model in client_models]

    classifier_prior_mode = getattr(args, "classifier_prior", "uniform")
    classifier_weights = aggregate_classifier_parameters(
        global_dict,
        client_state_dicts,
        classifier_prior_mode,
        client_sample_counts,
    )

    for client_model, state_dict in zip(client_models, client_state_dicts):
        for k in global_dict.keys():
            if "classifier" in k:
                state_dict[k] = global_dict[k]
        client_model.load_state_dict(state_dict)

    fedex_prior_mode = getattr(args, "fedex_prior", "uniform")
    fedex_reference = torch.empty(
        len(client_models), dtype=torch.float32, device=device
    )
    fedex_weights = compute_stable_prior_weights(
        fedex_reference, fedex_prior_mode, client_sample_counts
    )
    print(f"Classifier weights: {_format_weights(classifier_weights)}")
    print(f"FedEx adapter weights: {_format_weights(fedex_weights)}")

    try:
        save_mechanism_data(
            "fedex_weights",
            {
                f"client_{i}": float(w)
                for i,w in enumerate(fedex_weights.detach().cpu())
            },
            getattr(args,"current_round",0)
        )
    except Exception:
        pass

    for name, module in global_model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            lora_A_keys = f"{name}.lora_A.default.weight"
            lora_B_keys = f"{name}.lora_B.default.weight"
            base_layer_keys = f"{name}.base_layer.weight"

            lora_A_weights = torch.stack(
                [state_dict[lora_A_keys].detach().to(device) for state_dict in client_state_dicts]
            )
            lora_B_weights = torch.stack(
                [state_dict[lora_B_keys].detach().to(device) for state_dict in client_state_dicts]
            )

            fedex_component_weights = (
                None if fedex_prior_mode == "uniform" else fedex_weights
            )
            M, lora_A_avg, lora_B_avg, residue = compute_fedex_exact_components(
                lora_A_weights, lora_B_weights, fedex_component_weights
            )

            if args.rslora:
                scaling_factor = args.lora_alpha / np.sqrt(args.lora_r)
            else:
                scaling_factor = args.lora_alpha / args.lora_r

            global_dict[lora_A_keys] = lora_A_avg
            global_dict[lora_B_keys] = lora_B_avg
            global_dict[base_layer_keys] = global_dict[base_layer_keys].to(device)
            global_dict[base_layer_keys] += torch.transpose(
                residue * scaling_factor, 1, 0
            ).to(device)

    global_model.load_state_dict(global_dict)
    return global_model

def aggregate_models_fedawa(
    global_model, client_models, args, client_sample_counts=None
):
    """Aggregate LoRA adapters with FedEx-AWA dynamic client weights."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_model = global_model.to(device)
    global_dict = global_model.state_dict()
    client_state_dicts = [model.state_dict() for model in client_models]

    classifier_prior_mode = getattr(args, "classifier_prior", "uniform")
    classifier_weights = aggregate_classifier_parameters(
        global_dict,
        client_state_dicts,
        classifier_prior_mode,
        client_sample_counts,
    )

    for client_model, state_dict in zip(client_models, client_state_dicts):
        for k in global_dict.keys():
            if "classifier" in k:
                state_dict[k] = global_dict[k]
        client_model.load_state_dict(state_dict)

    temperature = getattr(args, "awa_temperature", 1.0)
    stable_prior_mode = getattr(args, "stable_prior", "uniform")
    prior_reference = torch.empty(
        len(client_models), dtype=torch.float32, device=device
    )
    prior_weights = compute_stable_prior_weights(
        prior_reference, stable_prior_mode, client_sample_counts
    )
    print(f"Adapter stable prior mode: {stable_prior_mode}")
    print(f"Client sample counts: {_format_sample_counts(client_sample_counts)}")
    print(f"Prior weights: {_format_weights(prior_weights)}")
    print(f"Classifier weights: {_format_weights(classifier_weights)}")

    for name, module in global_model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            lora_A_keys = f"{name}.lora_A.default.weight"
            lora_B_keys = f"{name}.lora_B.default.weight"
            base_layer_keys = f"{name}.base_layer.weight"

            lora_A_weights = torch.stack(
                [state_dict[lora_A_keys].detach().to(device) for state_dict in client_state_dicts]
            )
            lora_B_weights = torch.stack(
                [state_dict[lora_B_keys].detach().to(device) for state_dict in client_state_dicts]
            )
            dynamic_weights = compute_dynamic_weights_from_lora(
                lora_A_weights, lora_B_weights, temperature=temperature
            )

            awa_strength = _safe_float(getattr(args, "awa_strength", 0.5), 0.5)
            final_weights = mix_stable_weights(
                dynamic_weights, prior_weights, awa_strength
            )

            print(f"LoRA layer: {name}")
            print(f"Dynamic adapter weights: {_format_weights(dynamic_weights)}")
            print(f"Final adapter weights: {_format_weights(final_weights)}")

            num_clients = len(final_weights)

            M = sum(
                final_weights[i] * (lora_B_weights[i] @ lora_A_weights[i])
                for i in range(num_clients)
            )
            lora_A_avg = sum(
                final_weights[i] * lora_A_weights[i] for i in range(num_clients)
            )
            lora_B_avg = sum(
                final_weights[i] * lora_B_weights[i] for i in range(num_clients)
            )

            if args.rslora:
                scaling_factor = args.lora_alpha / np.sqrt(args.lora_r)
            else:
                scaling_factor = args.lora_alpha / args.lora_r

            residue = M - lora_B_avg @ lora_A_avg
            global_dict[lora_A_keys] = lora_A_avg
            global_dict[lora_B_keys] = lora_B_avg
            global_dict[base_layer_keys] = global_dict[base_layer_keys].to(device)
            global_dict[base_layer_keys] += torch.transpose(
                residue * scaling_factor, 1, 0
            ).to(device)

    global_model.load_state_dict(global_dict)
    return global_model


def aggregate_models_fedawa_effective(
    global_model, client_models, args, client_sample_counts=None
):
    """Run FedEx-AWA using global functional-delta agreement for its weights."""
    if not client_models:
        raise ValueError("At least one client is required for aggregation.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = global_model.to(device)
    global_dict = global_model.state_dict()
    client_state_dicts = [model.state_dict() for model in client_models]

    classifier_prior_mode = getattr(args, "classifier_prior", "uniform")
    classifier_weights = aggregate_classifier_parameters(
        global_dict,
        client_state_dicts,
        classifier_prior_mode,
        client_sample_counts,
    )

    for client_model, state_dict in zip(client_models, client_state_dicts):
        for key in global_dict.keys():
            if "classifier" in key:
                state_dict[key] = global_dict[key]
        client_model.load_state_dict(state_dict)

    print("Aggregation mode: fedawa_effective")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    similarity_start = time.perf_counter()
    layers = _collect_lora_factors(global_model, client_models, device)
    effective_similarities, effective_scores, mean_effective_similarity = (
        compute_effective_update_pairwise_cosines(layers)
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    similarity_time = time.perf_counter() - similarity_start

    _, _, mean_raw_similarity = compute_raw_factor_pairwise_cosines(layers)
    all_finite = _all_factors_finite(layers)
    temperature = getattr(args, "awa_temperature", 1.0)
    if all_finite:
        dynamic_weights = _dynamic_weights_from_similarity_scores(
            effective_scores, temperature=temperature
        )
    else:
        warnings.warn(
            "Non-finite LoRA factors detected; using uniform dynamic weights for "
            "fedawa_effective.",
            RuntimeWarning,
        )
        dynamic_weights = torch.full(
            (len(client_models),),
            1.0 / len(client_models),
            dtype=torch.float32,
            device=device,
        )

    num_clients = len(dynamic_weights)
    stable_prior_mode = getattr(args, "stable_prior", "uniform")
    prior_weights = compute_stable_prior_weights(
        dynamic_weights, stable_prior_mode, client_sample_counts
    )
    awa_strength = _safe_float(getattr(args, "awa_strength", 0.5), 0.5)
    final_weights = mix_stable_weights(
        dynamic_weights, prior_weights, awa_strength
    )

    print(f"Adapter stable prior mode: {stable_prior_mode}")
    print(f"Client sample counts: {_format_sample_counts(client_sample_counts)}")
    print(f"Prior weights: {_format_weights(prior_weights)}")
    print(f"Classifier weights: {_format_weights(classifier_weights)}")
    print("Effective-update pairwise cosine similarities:")
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            print(
                f"client {i + 1} vs client {j + 1} = "
                f"{effective_similarities[i, j].item():.8f}"
            )
    print(f"Mean raw-factor cosine = {mean_raw_similarity.item():.8f}")
    print(f"Mean effective-update cosine = {mean_effective_similarity.item():.8f}")
    print(f"Dynamic adapter weights: {_format_weights(dynamic_weights)}")
    print(f"Final adapter weights: {_format_weights(final_weights)}")
    print(f"Effective similarity computation time = {similarity_time:.6f} s")

    # From this point on, preserve the stable FedAWA/FedEx exact aggregation:
    # the only difference is that one full-model weight vector is reused for
    # every LoRA layer.
    for layer in layers:
        lora_A_key = layer["A_key"]
        lora_B_key = layer["B_key"]
        base_layer_key = f"{layer['prefix']}.base_layer.weight"
        if base_layer_key not in global_dict:
            raise ValueError(
                f"Base-layer weight for LoRA module {layer['prefix']!r} was not found."
            )

        lora_A_weights = layer["client_A"]
        lora_B_weights = layer["client_B"]
        M = sum(
            final_weights[i] * (lora_B_weights[i] @ lora_A_weights[i])
            for i in range(num_clients)
        )
        lora_A_avg = sum(
            final_weights[i] * lora_A_weights[i] for i in range(num_clients)
        )
        lora_B_avg = sum(
            final_weights[i] * lora_B_weights[i] for i in range(num_clients)
        )

        if args.rslora:
            scaling_factor = args.lora_alpha / np.sqrt(args.lora_r)
        else:
            scaling_factor = args.lora_alpha / args.lora_r

        residue = M - lora_B_avg @ lora_A_avg
        global_dict[lora_A_key] = lora_A_avg
        global_dict[lora_B_key] = lora_B_avg
        global_dict[base_layer_key] = global_dict[base_layer_key].to(device)
        global_dict[base_layer_key] += torch.transpose(
            residue * scaling_factor, 1, 0
        ).to(device)

    global_model.load_state_dict(global_dict)
    return global_model


def aggregate_models_fedawa_raw_global(
    global_model, client_models, args, client_sample_counts=None
):
    """Run FedEx-AWA with one global raw-factor-delta client weight vector."""
    if not client_models:
        raise ValueError("At least one client is required for aggregation.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = global_model.to(device)
    global_dict = global_model.state_dict()
    client_state_dicts = [model.state_dict() for model in client_models]

    classifier_prior_mode = getattr(args, "classifier_prior", "uniform")
    classifier_weights = aggregate_classifier_parameters(
        global_dict,
        client_state_dicts,
        classifier_prior_mode,
        client_sample_counts,
    )

    for client_model, state_dict in zip(client_models, client_state_dicts):
        for key in global_dict.keys():
            if "classifier" in key:
                state_dict[key] = global_dict[key]
        client_model.load_state_dict(state_dict)

    print("Aggregation mode: fedawa_raw_global")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    similarity_start = time.perf_counter()
    layers = _collect_lora_factors(global_model, client_models, device)
    raw_similarities, raw_scores, mean_raw_similarity = (
        compute_raw_factor_pairwise_cosines(layers)
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    similarity_time = time.perf_counter() - similarity_start

    all_finite = _all_factors_finite(layers)
    temperature = getattr(args, "awa_temperature", 1.0)
    if all_finite:
        dynamic_weights = _dynamic_weights_from_similarity_scores(
            raw_scores, temperature=temperature
        )
    else:
        warnings.warn(
            "Non-finite LoRA factors detected; using uniform dynamic weights for "
            "fedawa_raw_global.",
            RuntimeWarning,
        )
        dynamic_weights = torch.full(
            (len(client_models),),
            1.0 / len(client_models),
            dtype=torch.float32,
            device=device,
        )

    num_clients = len(dynamic_weights)
    stable_prior_mode = getattr(args, "stable_prior", "uniform")
    prior_weights = compute_stable_prior_weights(
        dynamic_weights, stable_prior_mode, client_sample_counts
    )
    awa_strength = _safe_float(getattr(args, "awa_strength", 0.5), 0.5)
    final_weights = mix_stable_weights(
        dynamic_weights, prior_weights, awa_strength
    )

    print(f"Adapter stable prior mode: {stable_prior_mode}")
    print(f"Client sample counts: {_format_sample_counts(client_sample_counts)}")
    print(f"Prior weights: {_format_weights(prior_weights)}")
    print(f"Classifier weights: {_format_weights(classifier_weights)}")
    print("Global raw-factor pairwise cosine similarities:")
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            print(
                f"client {i + 1} vs client {j + 1} = "
                f"{raw_similarities[i, j].item():.8f}"
            )
    print(f"Mean global raw-factor cosine = {mean_raw_similarity.item():.8f}")
    print(f"Dynamic adapter weights: {_format_weights(dynamic_weights)}")
    print(f"Final adapter weights: {_format_weights(final_weights)}")
    print(f"Raw-global similarity computation time = {similarity_time:.6f} s")

    # Reuse the one global weight vector for every LoRA layer, exactly as in
    # fedawa_effective. Only the similarity representation above differs.
    for layer in layers:
        lora_A_key = layer["A_key"]
        lora_B_key = layer["B_key"]
        base_layer_key = f"{layer['prefix']}.base_layer.weight"
        if base_layer_key not in global_dict:
            raise ValueError(
                f"Base-layer weight for LoRA module {layer['prefix']!r} was not found."
            )

        lora_A_weights = layer["client_A"]
        lora_B_weights = layer["client_B"]
        M, lora_A_avg, lora_B_avg, residue = compute_fedex_exact_components(
            lora_A_weights, lora_B_weights, final_weights
        )

        if args.rslora:
            scaling_factor = args.lora_alpha / np.sqrt(args.lora_r)
        else:
            scaling_factor = args.lora_alpha / args.lora_r

        global_dict[lora_A_key] = lora_A_avg
        global_dict[lora_B_key] = lora_B_avg
        global_dict[base_layer_key] = global_dict[base_layer_key].to(device)
        global_dict[base_layer_key] += torch.transpose(
            residue * scaling_factor, 1, 0
        ).to(device)

    global_model.load_state_dict(global_dict)
    return global_model


def aggregate_models_fedawa_absolute_ba_global(
    global_model, client_models, args, client_sample_counts=None
):
    """Run FedEx-AWA with one global scaled absolute-BA weight vector."""
    if not client_models:
        raise ValueError("At least one client is required for aggregation.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = global_model.to(device)
    global_dict = global_model.state_dict()
    client_state_dicts = [model.state_dict() for model in client_models]

    classifier_prior_mode = getattr(args, "classifier_prior", "uniform")
    classifier_weights = aggregate_classifier_parameters(
        global_dict,
        client_state_dicts,
        classifier_prior_mode,
        client_sample_counts,
    )

    for client_model, state_dict in zip(client_models, client_state_dicts):
        for key in global_dict.keys():
            if "classifier" in key:
                state_dict[key] = global_dict[key]
        client_model.load_state_dict(state_dict)

    print("Aggregation mode: fedawa_absolute_ba_global")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    similarity_start = time.perf_counter()
    layers = _collect_lora_factors(global_model, client_models, device)
    absolute_similarities, absolute_scores, mean_absolute_similarity = (
        compute_absolute_ba_pairwise_cosines(layers)
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    similarity_time = time.perf_counter() - similarity_start

    all_finite = _all_factors_finite(layers)
    temperature = getattr(args, "awa_temperature", 1.0)
    if all_finite:
        dynamic_weights = _dynamic_weights_from_similarity_scores(
            absolute_scores, temperature=temperature
        )
    else:
        warnings.warn(
            "Non-finite LoRA factors detected; using uniform dynamic weights for "
            "fedawa_absolute_ba_global.",
            RuntimeWarning,
        )
        dynamic_weights = torch.full(
            (len(client_models),),
            1.0 / len(client_models),
            dtype=torch.float32,
            device=device,
        )

    num_clients = len(dynamic_weights)
    stable_prior_mode = getattr(args, "stable_prior", "uniform")
    prior_weights = compute_stable_prior_weights(
        dynamic_weights, stable_prior_mode, client_sample_counts
    )
    awa_strength = _safe_float(getattr(args, "awa_strength", 0.5), 0.5)
    final_weights = mix_stable_weights(
        dynamic_weights, prior_weights, awa_strength
    )

    print(f"Client sample counts: {_format_sample_counts(client_sample_counts)}")
    print(f"Prior weights: {_format_weights(prior_weights)}")
    print(f"Classifier weights: {_format_weights(classifier_weights)}")
    print("Absolute-BA global pairwise cosine similarities:")
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            print(
                f"client {i + 1} vs client {j + 1} = "
                f"{absolute_similarities[i, j].item():.8f}"
            )
    print(f"Mean absolute-BA global cosine = {mean_absolute_similarity.item():.8f}")
    print(f"Dynamic adapter weights: {_format_weights(dynamic_weights)}")
    print(f"Final adapter weights: {_format_weights(final_weights)}")
    print(f"Absolute-BA similarity computation time = {similarity_time:.6f} s")

    # Keep the weighted FedEx exact aggregation identical to fedawa_effective;
    # only the global similarity representation above is different.
    for layer in layers:
        lora_A_key = layer["A_key"]
        lora_B_key = layer["B_key"]
        base_layer_key = f"{layer['prefix']}.base_layer.weight"
        if base_layer_key not in global_dict:
            raise ValueError(
                f"Base-layer weight for LoRA module {layer['prefix']!r} was not found."
            )

        lora_A_weights = layer["client_A"]
        lora_B_weights = layer["client_B"]
        M = sum(
            final_weights[i] * (lora_B_weights[i] @ lora_A_weights[i])
            for i in range(num_clients)
        )
        lora_A_avg = sum(
            final_weights[i] * lora_A_weights[i] for i in range(num_clients)
        )
        lora_B_avg = sum(
            final_weights[i] * lora_B_weights[i] for i in range(num_clients)
        )

        if args.rslora:
            scaling_factor = args.lora_alpha / np.sqrt(args.lora_r)
        else:
            scaling_factor = args.lora_alpha / args.lora_r

        residue = M - lora_B_avg @ lora_A_avg
        global_dict[lora_A_key] = lora_A_avg
        global_dict[lora_B_key] = lora_B_avg
        global_dict[base_layer_key] = global_dict[base_layer_key].to(device)
        global_dict[base_layer_key] += torch.transpose(
            residue * scaling_factor, 1, 0
        ).to(device)

    global_model.load_state_dict(global_dict)
    return global_model
