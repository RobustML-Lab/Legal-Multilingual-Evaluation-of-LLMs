import json
from deep_translator import GoogleTranslator
from tqdm import tqdm  # Optional for progress bar

# Supported language codes for translation
language_fields = ['en', 'fr', 'es', 'ar', 'ru', 'zh']

language_code_map = {
    "en": "en",
    "es": "es",
    "fr": "fr",
    "ar": "ar",
    "zh": "zh-CN",  # Simplified Chinese
    "ru": "ru"
}

def round_trip_translate(text, source_lang_code, pivot_lang_code='el'):
    try:
        # Translate to Greek
        to_pivot = GoogleTranslator(source=source_lang_code, target=pivot_lang_code).translate(text)
        # Translate back to original language
        back = GoogleTranslator(source=pivot_lang_code, target=source_lang_code).translate(to_pivot)
        return back
    except Exception as e:
        print(f"[{source_lang_code}] Translation error: {e}")
        return text
# Read your .jsonl file
input_path = "sampled_sdgs_one_label.jsonl"
output_path = "attacked_data_one_label.jsonl"

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in tqdm(infile):
        try:
            obj = json.loads(line)

            # For each language, create an attacked version
            for lang in language_fields:
                if lang in obj and isinstance(obj[lang], str) and obj[lang].strip():
                    obj[f"{lang}_attacked"] = round_trip_translate(obj[lang], language_code_map.get(lang))
                else:
                    obj[f"{lang}_attacked"] = obj.get(lang, "")

            # Write updated object
            outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")

        except json.JSONDecodeError:
            print("Skipping invalid JSON line.")
