import json
import random

# Define the languages to process
languages = ['en', 'ru', 'ar', 'es', 'fr', 'zh']

# Percentage of characters to modify
CHAR_MOD_PERCENT = 0.2  # 10% of the characters

# Modify text at the character level
def character_level_attack(text, percent=0.1):
    chars = list(text)
    num_mods = max(1, int(len(chars) * percent))
    indices = list(range(len(chars)))
    random.shuffle(indices)
    modified = 0

    for idx in indices:
        if modified >= num_mods or idx >= len(chars):
            break

        action = random.choice(['swap', 'delete', 'insert'])
        if action == 'swap' and idx < len(chars) - 1:
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            modified += 1
        elif action == 'delete':
            chars.pop(idx)
            modified += 1
        elif action == 'insert':
            insert_char = random.choice(chars)
            chars.insert(idx, insert_char)
            modified += 1

    return ''.join(chars)

# Input/output files
input_path = 'sampled_sdgs_one_label.jsonl'
output_path = 'sampled_sdgs_attack_character_level_02.jsonl'

# Process the file
with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        record = json.loads(line)
        for lang in languages:
            if lang in record and isinstance(record[lang], str):
                try:
                    record[f'{lang}_char_attack'] = character_level_attack(record[lang], percent=CHAR_MOD_PERCENT)
                except Exception as e:
                    print(f"Error modifying '{lang}' text: {e}")
        json.dump(record, outfile, ensure_ascii=False)
        outfile.write('\n')
