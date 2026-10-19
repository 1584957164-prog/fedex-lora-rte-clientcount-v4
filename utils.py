import torch
import os
from datetime import datetime
import json


def tensor_to_list(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_list(v) for v in obj]
    else:
        return obj


def save_dict_to_json(data_dict, args, base_path):
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"compare_dict_rounds_{timestamp}.json"
    file_path = os.path.join(base_path, filename)

    # Ensure the directory exists
    os.makedirs(base_path, exist_ok=True)

    # Combine data_dict and args
    combined_dict = {"args": vars(args), "data": data_dict}

    # Convert tensors to lists
    json_serializable_dict = tensor_to_list(combined_dict)

    # Write JSON data to the file
    with open(file_path, "w") as json_file:
        json.dump(json_serializable_dict, json_file, indent=2)

    print(f"Data and args saved to {file_path}")
