import json
import os

from transformers import BertTokenizer, BertModel
import torch

def get_embedding_bert(text):
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")  # Supports multiple languages
    model = BertModel.from_pretrained("bert-base-multilingual-cased")
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1)

def store_predicted(predicted, lang, dataset_name, points_per_language, model_name, adversarial_attack):
    output_data = {
        "predicted": predicted,
        "language": lang,
        "dataset_name": dataset_name,
        "points_per_language": points_per_language,
        "model_name": model_name,
        "adversarial_attack": adversarial_attack
    }

    # Ensure output directory exists
    output_dir = "output/predicted"
    os.makedirs(output_dir, exist_ok=True)

    # Find next available filename
    base_filename = "predicted"
    existing_files = [f for f in os.listdir(output_dir) if f.startswith(base_filename) and f.endswith(".json")]

    # Extract existing indices
    indices = []
    for fname in existing_files:
        parts = fname.replace(".json", "").split("_")
        if len(parts) == 1:
            indices.append(0)
        elif parts[1].isdigit():
            indices.append(int(parts[1]))

    # Get next index
    next_index = max(indices) + 1 if indices else 0
    filename = f"{base_filename}_{next_index}.json" if next_index > 0 else f"{base_filename}.json"

    # Write to file
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"Saved predictions to: {filename}")

def store_attack(before_attack, after_attack, lang, dataset_name, points_per_language, model_name, adversarial_attack):
    file_name = "attack.txt"
    file_path = f"output/attacks/{file_name}"

    # Make sure output/ directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Load existing data if the file exists
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    # Create a new result entry with all metadata
    result_entry = {
        "before_attack": before_attack,
        "after_attack": after_attack,
        "language": lang,
        "dataset_name": dataset_name,
        "points_per_language": points_per_language,
        "model_name": model_name,
        "adversarial_attack": adversarial_attack
    }

    existing_data.append(result_entry)

    # Write back to file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

    print(f"Attack entry stored in: {file_path}")
