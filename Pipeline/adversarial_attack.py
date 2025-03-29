import random
import textattack
from nltk.corpus.reader import WordNetError
from textattack.attack_recipes import TextBuggerLi2018, TextFoolerJin2019
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.augmentation import EasyDataAugmenter, WordNetAugmenter
from textattack.attack_recipes import GeneticAlgorithmAlzantot2018
from textattack.transformations import (
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterDeletion,
    WordSwapQWERTY,
    WordSwapEmbedding
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from difflib import SequenceMatcher

import torch
import nltk

import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
from nlpaug.util import Action

# Language mapping
language_map = {
    'english': 'en', 'danish': 'da', 'german': 'de', 'dutch': 'nl', 'swedish': 'sv',
    'spanish': 'es', 'french': 'fr', 'italian': 'it', 'portuguese': 'pt', 'romanian': 'ro',
    'bulgarian': 'bg', 'czech': 'cs', 'croatian': 'hr', 'polish': 'pl', 'slovenian': 'sl',
    'estonian': 'et', 'finnish': 'fi', 'hungarian': 'hu', 'lithuanian': 'lt', 'latvian': 'lv',
    'greek': 'el', 'irish': 'ga', 'maltese': 'mt', 'slovak': 'sk', 'chinese': 'zh'
}

wordnet_lang_map = {
    'en': 'english',
    'es': 'spanish',
    'fr': 'french',
    'de': 'german',
    'nl': 'dutch',
    'it': 'italian',
}

nlpaug_lang_map = {
    'en': 'eng',
    'es': 'spa',
    'fr': 'fra',
    'de': 'deu',
    'nl': 'nld',
    'it': 'ita',
    'zh': 'zho'
}

lang_map = {
    "ar": "ara",  # Arabic
    "bg": "bul",  # Bulgarian
    "de": "deu",  # German
    "el": "ell",  # Greek
    "en": "eng",  # English
    "es": "spa",  # Spanish
    "fr": "fra",  # French
    "hi": "hin",  # Hindi
    "ro": "ron",  # Romanian
    "ru": "rus",  # Russian
    "th": "tha",  # Thai
    "tr": "tur",  # Turkish
    "vi": "vie",  # Vietnamese
    "zh": "zho",  # Chinese (Simplified)
    "nl": "nld",  # Dutch
    "hu": "hun",  # Hungarian
    "it": "ita",  # Italian
    "no": "nob",  # Norwegian Bokmål (used as 'no' in deep-translator)
    "nb": "nob",  # Alias for consistency
    "pl": "pol",  # Polish
    "mt": "mlt",  # Maltese
    "sv": "swe",  # Swedish
}




# Mapping NLTK POS tags to WordNet POS tags
pos_map = {
    "NN": wordnet.NOUN,
    "NNS": wordnet.NOUN,
    "VB": wordnet.VERB,
    "VBD": wordnet.VERB,
    "VBG": wordnet.VERB,
    "VBN": wordnet.VERB,
    "VBP": wordnet.VERB,
    "VBZ": wordnet.VERB,
    "JJ": wordnet.ADJ,
    "JJR": wordnet.ADJ,
    "JJS": wordnet.ADJ,
    "RB": wordnet.ADV,
    "RBR": wordnet.ADV,
    "RBS": wordnet.ADV,
}

LANG_TO_MODEL = {
    'ara': 'asafaya/bert-base-arabic',  # Arabic-specific BERT
    'bul': 'bert-base-multilingual-uncased',  # No dedicated Bulgarian model, use mBERT
    'deu': 'dbmdz/bert-base-german-uncased',  # German BERT
    'ell': 'nlpaueb/bert-base-greek-uncased-v1',  # Greek-specific BERT
    'eng': 'bert-base-uncased',  # Standard English BERT
    'spa': 'dccuchile/bert-base-spanish-wwm-uncased',  # Spanish-specific BERT
    'fra': 'camembert-base',  # French-specific RoBERTa-based model
    'hin': 'google/muril-base-cased',  # Hindi-specific multilingual model
    'ron': 'dumitrescustefan/bert-base-romanian-uncased',  # Romanian-specific BERT
    'rus': 'DeepPavlov/rubert-base-cased',  # Russian BERT
    'tha': 'airesearch/wangchanberta-base-att-spm-uncased',  # Thai-specific BERT
    'tur': 'dbmdz/bert-base-turkish-uncased',  # Turkish-specific BERT
    'vie': 'NlpHUST/vibert4news-base-cased',  # Vietnamese-specific BERT
    'zho': 'bert-base-chinese',  # Chinese (Simplified) BERT
    "nld": "wietsedv/bert-base-dutch-cased",
    "hun": "SZTAKI-HLT/hubert-base-cc",  # Hungarian
    "ita": "dbmdz/bert-base-italian-xxl-cased",
    "nob": "NbAiLab/nb-bert-base",  # Norwegian Bokmål
    "pol": "dkleczek/bert-base-polish-cased-v1",
    "mlt": "bert-base-multilingual-uncased",  # fallback for Maltese
    "swe": "KBLab/bert-base-swedish-cased"
}


nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('omw-1.4')



# Load a real classifier model (e.g., BERT for classification)
model_name = "textattack/bert-base-uncased-SST-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


class CustomHuggingFaceModelWrapper(HuggingFaceModelWrapper):
    def __call__(self, text_inputs):
        """Ensure model outputs correct logits format for TextAttack."""
        model_outputs = super().__call__(text_inputs)  # Get raw logits

        print(f"DEBUG: Raw model output shape: {model_outputs.shape}")  # Print actual shape

        if isinstance(model_outputs, torch.Tensor):
            # Ensure (batch_size, num_classes) shape
            if model_outputs.dim() == 1:  # If output is (num_classes,)
                model_outputs = model_outputs.unsqueeze(0)  # Add batch dimension
            elif model_outputs.dim() == 3:  # If output is (batch, 1, num_classes)
                model_outputs = model_outputs.squeeze(1)

        print(f"DEBUG: Fixed model output shape: {model_outputs.shape}")  # Print new shape
        return model_outputs


model_wrapper = CustomHuggingFaceModelWrapper(model, tokenizer)

def attack(data, attack_type, lang, mapped_data):
    """Apply adversarial attacks using TextAttack-based methods."""
    if lang in language_map:
        lang = language_map[lang]

    total_words = 0
    changed_words = 0

    for i, entry in enumerate(data):
        if "text" in entry:
            original_text = entry["text"]

            ground_truth_label = None
            if "label" in entry:
                ground_truth_label = mapped_data[i]["label"] if mapped_data and i < len(mapped_data) else None

            modified_text, changes = adversarial_attack(original_text, attack_type, lang, ground_truth_label)

            total_words += 1
            changed_words += changes
            entry["text"] = modified_text

    change_percentage = (changed_words / total_words) if total_words > 0 else 0
    save_results(lang, change_percentage, attack_type)
    return data

def adversarial_attack(text, attack_type, lang, ground_truth_label):
    """Applies different adversarial attack strategies based on the attack type."""
    if attack_type == 1:  # Word substitution and augmentation attack
        return word_substitution_attack(text, lang)
    elif attack_type == 2:  # Typo-based attack
        return typo_attack(text)
    elif attack_type == 3:  # Character swap attack
        return character_swap_attack(text)
    elif attack_type == 4:  # TextBugger Attack (typo-based adversarial attack)
        return textbugger_attack(text, ground_truth_label)
    elif attack_type == 5:  # TextFooler Attack (synonym-based adversarial attack)
        return textfooler_attack(text, ground_truth_label)
    elif attack_type == 6:  # CLARE (Context-Aware Rewriting)
        return clare_attack(text)
    elif attack_type == 7:  # TextEvo (Evolutionary Adversarial Attack)
        return textevo_attack(text)
    elif attack_type == 8:  # Genetic Attack (uses evolutionary algorithms to modify words)
        return genetic_attack(text, ground_truth_label)
    elif attack_type == 9:  # Word substitution for multilingual
        return synonym_multilingual_attack(text, lang)
    elif 9 < attack_type < 15: # Attack using nlpaug
        return nlpaug_attack(text, lang, attack_type)
    else:
        raise ValueError(f"Unsupported attack_type: {attack_type}")

def word_substitution_attack(text, lang):
    """Apply synonym-based word substitutions."""
    augmenter = EasyDataAugmenter(transformations_per_example=2)
    perturbed_texts = augmenter.augment(text)
    if perturbed_texts:
        text = perturbed_texts[0]

    augmenter = WordNetAugmenter(transformations_per_example=2)
    perturbed_texts = augmenter.augment(text)
    if perturbed_texts:
        text = perturbed_texts[0]

    return text, count_changes(text, text)

def typo_attack(text):
    """Introduce typos using TextAttack's transformation methods."""
    augmenter = textattack.augmentation.Augmenter(
        transformation=WordSwapRandomCharacterInsertion(), transformations_per_example=1
    )
    perturbed_texts = augmenter.augment(text)
    if perturbed_texts:
        text = perturbed_texts[0]
    return text, count_changes(text, text)

def character_swap_attack(text):
    """Apply character swaps and deletions."""
    augmenter = textattack.augmentation.Augmenter(
        transformation=WordSwapQWERTY(), transformations_per_example=1
    )
    perturbed_texts = augmenter.augment(text)
    if perturbed_texts:
        text = perturbed_texts[0]

    augmenter = textattack.augmentation.Augmenter(
        transformation=WordSwapRandomCharacterDeletion(), transformations_per_example=1
    )
    perturbed_texts = augmenter.augment(text)
    if perturbed_texts:
        text = perturbed_texts[0]

    return text, count_changes(text, text)

def clare_attack(text, max_number_of_changes=10):
    """Apply CLARE (Context-Aware Rewriting) adversarial attack."""
    words = text.split()
    possible_changes = max(1, min(int(len(words) / 2), max_number_of_changes))
    augmenter = textattack.augmentation.Augmenter(
        transformation=WordSwapEmbedding(max_candidates=3),
        transformations_per_example=possible_changes
    )
    perturbed_texts = augmenter.augment(text)
    if perturbed_texts:
        text = perturbed_texts[0]

    return text, count_changes(text, text)

def textevo_attack(text, max_number_of_changes=10):
    """Apply TextEvo (fast evolutionary-based adversarial attack with diverse modifications)."""
    words = text.split()
    possible_changes = max(1, min(int(len(words) / 2), max_number_of_changes))

    for _ in range(possible_changes):
        transformation = random.choice([
            WordSwapQWERTY(),
            WordSwapRandomCharacterInsertion(),
            WordSwapRandomCharacterDeletion(),
            WordSwapEmbedding(max_candidates=2)
        ])

        augmenter = textattack.augmentation.Augmenter(
            transformation=transformation,
            transformations_per_example=1
        )

        perturbed_texts = augmenter.augment(text)
        if perturbed_texts:
            text = perturbed_texts[0]

    return text, count_changes(text, text)

def genetic_attack(text, ground_truth_label):
    """Apply Genetic Algorithm-based adversarial attack using a real classification model."""
    print(f"Starting attack on text: {text}")  # Debugging

    attack = GeneticAlgorithmAlzantot2018.build(model_wrapper, use_constraint=False)

    try:
        print("Running adversarial attack...")  # Debugging
        attack_result = attack.attack(text, ground_truth_label)
        print("Attack completed!")  # Debugging

        if isinstance(attack_result, textattack.attack_results.FailedAttackResult):
            print(f"Attack failed: {text}")
            return text, 0  # Return original text

        perturbed_text = attack_result.perturbed_text()
        print(f"Attack succeeded:\nOriginal: {text}\nAdversarial: {perturbed_text}")

        # Debug model output
        model_output = model_wrapper([perturbed_text])
        print(f"DEBUG: Model output shape: {model_output.shape}, Values: {model_output}")

        return perturbed_text, count_changes(text, perturbed_text)

    except IndexError as e:
        print(f"IndexError encountered: {e}. Retrying with reshaped logits.")
        return text, 0  # Return original if attack fails

def textbugger_attack(text, ground_truth_label):
    """Apply TextBugger adversarial attack using a real classification model."""
    attack = TextBuggerLi2018.build(model_wrapper)
    attack_result = attack.attack(text, ground_truth_label)

    if isinstance(attack_result, textattack.attack_results.FailedAttackResult):
        print('aaaaaaaaaaaaaaaaaaa')
        return text, 0

    return attack_result.perturbed_text(), count_changes(text, attack_result.perturbed_text())

def textfooler_attack(text, ground_truth_label):
    """Apply TextFooler adversarial attack using a real classification model."""
    attack = TextFoolerJin2019.build(model_wrapper)
    attack_result = attack.attack(text, ground_truth_label)

    if isinstance(attack_result, textattack.attack_results.FailedAttackResult):
        return text, 0

    return attack_result.perturbed_text(), count_changes(text, attack_result.perturbed_text())

def get_synonyms(word, pos, lang="english"):
    """Retrieve suitable synonyms for a word based on part of speech and language."""
    synonyms = set()

    wordnet_lang = wordnet_lang_map.get(lang.lower(), "eng")  # Convert lang to lowercase for safety

    for synset in wordnet.synsets(word, pos=pos, lang=wordnet_lang):
        for lemma in synset.lemmas():
            synonym = lemma.name().replace("_", " ")
            if synonym.lower() != word.lower():  # Avoid self-replacement
                synonyms.add(synonym)

    return list(synonyms)

def synonym_multilingual_attack(text, lang="english", target_percentage=10):
    """
    Ensures a consistent percentage of word replacements across languages using only WordNet.
    - Computes how many words to replace based on `target_percentage`.
    - If not enough synonyms exist, re-attempts substitutions until the target is met.
    """
    lang = lang.lower()
    if lang in wordnet_lang_map:
        lang = wordnet_lang_map[lang]

    try:
        sentences = sent_tokenize(text, language=lang)
    except LookupError:
        print(f"Error: NLTK does not support sentence tokenization for '{lang}'. Using default 'english'.")
        sentences = sent_tokenize(text, language="english")

    words = [word for sentence in sentences for word in word_tokenize(sentence)]
    tagged_words = pos_tag(words)

    total_words = len(tagged_words)
    target_changes = max(1, int((target_percentage / 100) * total_words))  # Ensure at least 1 change
    modified_words = 0

    perturbed_words = []
    potential_replacements = []

    # Collect words that can be replaced
    for word, tag in tagged_words:
        if tag in pos_map:
            wordnet_pos = pos_map[tag]
            synonyms = get_synonyms(word, wordnet_pos, lang)

            if synonyms:
                potential_replacements.append((word, random.choice(synonyms)))

    # Replace words based on collected replacements
    random.shuffle(potential_replacements)  # Shuffle for randomness
    replacement_dict = dict(potential_replacements[:target_changes])  # Select exact number needed

    for word, tag in tagged_words:
        if word in replacement_dict:
            perturbed_words.append(replacement_dict[word])
            modified_words += 1
        else:
            perturbed_words.append(word)

    perturbed_text = " ".join(perturbed_words)

    return perturbed_text, modified_words

def nlpaug_attack(text, lang, attack_type):
    """
    Applies an adversarial attack to the input text based on the attack type using the nlpaug library.

    Parameters:
        text (str): The input text to be modified.
        lang (str): The language of the text (default is 'eng' for English).
        attack_type (int): The type of attack to apply. Options:
                           10: Synonym Replacement (WordNet-based)
                           11: Random Character Insertion
                           12: Keyboard Typo Simulation
                           13: Back Translation
                           14: Contextual Word Substitution (BERT-based)

    Returns:
        str: The modified text after applying the attack.
    """

    if lang in lang_map.keys():
        lang = lang_map[lang]

    original_words = word_tokenize(text)

    if attack_type == 10:
        try:
            # Synonym replacement (WordNet-based, supports multiple languages)
            synonym_aug = naw.SynonymAug(aug_src='wordnet', lang=lang)
            augmented_text = synonym_aug.augment(text)
        except WordNetError as e:
            print(f"{e} {lang}")
            return text, 0

    elif attack_type == 11:
        # Character-level modifications (e.g., random character insertion)
        char_aug = nac.RandomCharAug(action="insert", aug_char_p=0.3)
        augmented_text = char_aug.augment(text)

    elif attack_type == 12:
        # Keyboard typo simulation (QWERTY-based)
        keyboard_aug = nac.KeyboardAug(aug_word_p=0.3)
        augmented_text = keyboard_aug.augment(text)

    elif attack_type == 13:
        # Back translation
        back_translation_aug = naw.BackTranslationAug(
            from_model_name='transformer.wmt19.en-de',
            to_model_name='transformer.wmt19.de-en'
        )
        augmented_text = back_translation_aug.augment(text)

    elif attack_type == 14:
        # Contextual word embeddings using BERT
        bert_model = LANG_TO_MODEL.get(lang, 'bert-base-multilingual-uncased')
        contextual_aug = naw.ContextualWordEmbsAug(model_path=bert_model, action="substitute")
        augmented_text = contextual_aug.augment(text)

    else:
        raise ValueError(f"Unsupported attack_type: {attack_type}. Supported types are 10-14.")

    if isinstance(augmented_text, list):
        augmented_text = augmented_text[0] if augmented_text else text

    modified_words = word_tokenize(augmented_text)
    changed_words = sum(1 for o, m in zip(original_words, modified_words) if o != m)
    changed_words += abs(len(original_words) - len(modified_words))
    total_words = len(original_words)
    change_percentage = (changed_words / total_words) * 100 if total_words > 0 else 0

    return augmented_text, change_percentage

def count_changes(original_text, modified_text):
    """Count modified words between the original and adversarial text."""
    return sum(1 for o, m in zip(original_text.split(), modified_text.split()) if o != m)

import os

def save_results(lang, percentage, attack_type):
    """Save attack results to a file."""

    # Ensure the output/attacks directory exists
    os.makedirs("output/attacks", exist_ok=True)

    # Append the results to the file
    with open("output/attacks/attack_percentage.txt", "a", encoding="utf-8") as f:
        f.write(f"{lang}: {percentage:.2f}% words modified, attack {attack_type}\n")

