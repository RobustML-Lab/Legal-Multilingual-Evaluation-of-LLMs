import json
import os

from transformers import BertTokenizer, BertModel
import torch

lang_map = {
    # EU Languages
    "en": "English", "eng": "English",
    "fr": "French", "fra": "French",
    "de": "German", "deu": "German",
    "es": "Spanish", "spa": "Spanish",
    "it": "Italian", "ita": "Italian",
    "pt": "Portuguese", "por": "Portuguese",
    "nl": "Dutch", "nld": "Dutch",
    "sv": "Swedish", "swe": "Swedish",
    "da": "Danish", "dan": "Danish",
    "fi": "Finnish", "fin": "Finnish",
    "no": "Norwegian", "nor": "Norwegian",
    "is": "Icelandic", "isl": "Icelandic",
    "pl": "Polish", "pol": "Polish",
    "cs": "Czech", "ces": "Czech",
    "sk": "Slovak", "slk": "Slovak",
    "hu": "Hungarian", "hun": "Hungarian",
    "ro": "Romanian", "ron": "Romanian",
    "bg": "Bulgarian", "bul": "Bulgarian",
    "hr": "Croatian", "hrv": "Croatian",
    "sr": "Serbian", "srp": "Serbian",
    "sl": "Slovenian", "slv": "Slovenian",
    "et": "Estonian", "est": "Estonian",
    "lv": "Latvian", "lav": "Latvian",
    "lt": "Lithuanian", "lit": "Lithuanian",
    "mt": "Maltese", "mlt": "Maltese",
    "el": "Greek", "ell": "Greek",
    "ga": "Irish", "gle": "Irish",
    "cy": "Welsh", "cym": "Welsh",

    # Major non-EU Languages
    "ar": "Arabic", "ara": "Arabic",
    "tr": "Turkish", "tur": "Turkish",
    "zh": "Chinese", "zho": "Chinese",
    "zh-cn": "Chinese (Simplified)", "zhs": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)", "zht": "Chinese (Traditional)",
    "ja": "Japanese", "jpn": "Japanese",
    "ko": "Korean", "kor": "Korean",
    "hi": "Hindi", "hin": "Hindi",
    "th": "Thai", "tha": "Thai",
    "vi": "Vietnamese", "vie": "Vietnamese",
    "ru": "Russian", "rus": "Russian",
    "uk": "Ukrainian", "ukr": "Ukrainian",
    "he": "Hebrew", "heb": "Hebrew",
    "fa": "Persian", "fas": "Persian",
    "id": "Indonesian", "ind": "Indonesian",
    "ms": "Malay", "msa": "Malay",
}

def get_embedding_bert(text):
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")  # Supports multiple languages
    model = BertModel.from_pretrained("bert-base-multilingual-cased")
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1)

def store_predicted(predicted, true, lang, dataset_name, points_per_language, model_name, adversarial_attack):
    output_data = {
        "predicted": predicted,
        "true": true,
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


def store_judge(scores, numeric_scores, lan):
    # Ensure output directory exists
    os.makedirs("output/llm_judge", exist_ok=True)

    with open("output/llm_judge/judge_responses.txt", "a", encoding="utf-8") as f:
        f.write(f"==============Language: {lan}================\n")

        # Write raw responses
        f.write("Raw LLM Responses:\n")
        for i, response in enumerate(scores, 1):
            f.write(f"{response.strip()}, ")

        f.write("\nParsed Numeric Scores:\n")
        for i, score in enumerate(numeric_scores, 1):
            f.write(f"{score}, ")

        f.write("\n\n")


def get_language_from_code(lang):
    return lang_map.get(lang.lower(), "Unknown")