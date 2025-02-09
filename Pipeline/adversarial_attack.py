import random
import fasttext.util
from transformers import pipeline

# # Load multilingual masked language model (for context-aware attack)
# mask_filler = pipeline("fill-mask", model="xlm-roberta-base")
#
# # Cache for FastText models
# fasttext_models = {}

# def load_fasttext_model(lang):
#     """Load and cache FastText models for different languages."""
#     if lang not in fasttext_models:
#         fasttext.util.download_model(lang, if_exists='ignore')  # Download if needed
#         fasttext_models[lang] = fasttext.load_model(f'cc.{lang}.300.bin')
#     return fasttext_models[lang]

# Language name to FastText mapping
language_map = {
    'english': 'en', 'danish': 'da', 'german': 'de', 'dutch': 'nl', 'swedish': 'sv',
    'spanish': 'es', 'french': 'fr', 'italian': 'it', 'portuguese': 'pt', 'romanian': 'ro',
    'bulgarian': 'bg', 'czech': 'cs', 'croatian': 'hr', 'polish': 'pl', 'slovenian': 'sl',
    'estonian': 'et', 'finnish': 'fi', 'hungarian': 'hu', 'lithuanian': 'lt', 'latvian': 'lv',
    'greek': 'el', 'irish': 'ga', 'maltese': 'mt', 'slovak': 'sk',
}

# typos = {
#     "en": {"e": "3", "o": "0", "l": "1", "a": "@", "s": "$"},
#     "fr": {"e": "é", "a": "à", "c": "ç", "u": "ù", "o": "ô"},
#     "de": {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "e": "3"},
#     "es": {"n": "ñ", "e": "é", "o": "ó", "u": "ú", "i": "í"},
#     "pt": {"a": "ã", "e": "é", "o": "ô", "c": "ç", "u": "ú"},
#     "it": {"e": "è", "o": "ò", "i": "ì", "a": "à", "u": "ù"},
#     "nl": {"e": "ë", "i": "ï", "o": "ö", "u": "ü", "a": "á"},
#     "sv": {"a": "å", "o": "ö", "e": "é", "u": "ü", "d": "ð"},
#     "da": {"a": "å", "o": "ø", "e": "é", "u": "ü", "d": "ð"},
#     "fi": {"a": "ä", "o": "ö", "e": "ë", "u": "ü", "y": "ÿ"},
#     "no": {"a": "å", "o": "ø", "e": "é", "u": "ü", "y": "ÿ"},
#     "pl": {"o": "ó", "l": "ł", "z": "ż", "s": "ś", "c": "ć"},
#     "cs": {"c": "č", "e": "ě", "s": "š", "r": "ř", "z": "ž"},
#     "sk": {"c": "č", "e": "ě", "s": "š", "r": "ř", "z": "ž"},
#     "sl": {"c": "č", "s": "š", "z": "ž", "d": "đ", "n": "ñ"},
#     "hr": {"c": "č", "s": "š", "z": "ž", "d": "đ", "e": "é"},
#     "hu": {"o": "ó", "u": "ú", "e": "é", "a": "á", "i": "í"},
#     "ro": {"a": "ă", "t": "ț", "s": "ș", "i": "î", "e": "é"},
#     "bg": {"и": "й", "е": "ё", "г": "ґ", "д": "ѳ", "т": "ѳ"},
#     "el": {"α": "ά", "ε": "έ", "η": "ή", "ι": "ί", "ο": "ό"},
#     "et": {"o": "õ", "a": "ä", "u": "ü", "s": "š", "z": "ž"},
#     "lv": {"a": "ā", "e": "ē", "i": "ī", "u": "ū", "s": "š"},
#     "lt": {"a": "ą", "e": "ė", "i": "į", "u": "ų", "s": "š"},
#     "ga": {"a": "á", "o": "ó", "e": "é", "i": "í", "u": "ú"},
#     "mt": {"h": "ħ", "z": "ż", "c": "ċ", "g": "ġ", "e": "é"},
# }

typos = {
    "en": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
    "fr": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
    "de": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
    "es": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
    "pt": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
    "it": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
    "nl": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
    "sv": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
    "da": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
    "fi": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
    "no": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
    "pl": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
    "cs": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
    "sk": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
    "sl": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
    "hr": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
    "hu": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
    "ro": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
    "bg": {"е": "3", "о": "0", "а": "@", "с": "$", "и": "1"},
    "el": {"ε": "3", "ο": "0", "α": "@", "ς": "$", "ι": "1"},
    "et": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
    "lv": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
    "lt": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
    "ga": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
    "mt": {"e": "3", "o": "0", "a": "@", "s": "$", "i": "1"},
}



def attack(data, attack_type, lang):
    """Apply adversarial attacks to a dataset containing the 24 EU languages."""
    if lang in language_map:
        lang = language_map[lang]  # Convert language name to FastText format

    total_words = 0
    changed_words = 0

    for entry in data:
        if "text" in entry:
            original_text = entry["text"]
            modified_text, _ = adversarial_attack(entry["text"], attack_type, lang)

            # Count changes
            total_words += len(original_text.split())
            changed_words += count_changes(original_text, modified_text)

            entry["text"] = modified_text

    # Calculate percentage of changes
    change_percentage = (changed_words / total_words) * 100 if total_words > 0 else 0

    # Save results to file
    save_results(lang, change_percentage)

    return data

def adversarial_attack(text, attack_type, lang):
    # if attack_type == 1: # Synonym replacement
    #     return replace_synonyms(text, lang)
    if attack_type == 2: # Character typo
        return introduce_typos(text, lang)
    else:
        raise ValueError(f"Unsupported attack_type: {attack_type}")

def replace_synonyms(text, lang):
    words = text.split()
    modified_count = 0
    for i, word in enumerate(words):
        if random.random() < 0.2:
            synonym = get_synonym(word, lang)
            if synonym:
                words[i] = synonym
                modified_count += 1
    return " ".join(words)

# def get_synonym(word, lang):
#     """Try FastText first, then XLM-R for synonym replacement."""
#     ft_model = load_fasttext_model(lang)
#     nearest_neighbors = ft_model.get_nearest_neighbors(word)
#     if nearest_neighbors:
#         return nearest_neighbors[0][1]
#
#     masked_text = f"The word <mask> means {word}."
#     predictions = mask_filler(masked_text)
#     if predictions:
#         return predictions[0]["token_str"]
#
#     return word

def introduce_typos(text, lang):
    """Introduce typos randomly in words for all 24 EU languages."""
    # Get language-specific typo rules
    typo_rules = typos.get(lang)

    # Convert text to list for modification
    new_text = list(text)
    changed = 0

    for i in range(len(new_text)):
        if new_text[i] in typo_rules and random.random() < 0.1:  # 10% chance per character
            new_text[i] = typo_rules[new_text[i]]
            changed += 1

    return "".join(new_text), changed  # Return modified text and count of changes


def count_changes(original_text, modified_text):
    """Count the number of words that were modified."""
    original_words = original_text.split()
    modified_words = modified_text.split()

    changes = sum(1 for o, m in zip(original_words, modified_words) if o != m)
    return changes

def save_results(lang, percentage):
    """Save attack results to a file."""
    with open("output/attack_results.txt", "a", encoding="utf-8") as f:
        f.write(f"{lang}: {percentage:.2f}% words modified\n")

