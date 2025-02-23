import random
import textattack
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
import torch

# Language mapping
language_map = {
    'english': 'en', 'danish': 'da', 'german': 'de', 'dutch': 'nl', 'swedish': 'sv',
    'spanish': 'es', 'french': 'fr', 'italian': 'it', 'portuguese': 'pt', 'romanian': 'ro',
    'bulgarian': 'bg', 'czech': 'cs', 'croatian': 'hr', 'polish': 'pl', 'slovenian': 'sl',
    'estonian': 'et', 'finnish': 'fi', 'hungarian': 'hu', 'lithuanian': 'lt', 'latvian': 'lv',
    'greek': 'el', 'irish': 'ga', 'maltese': 'mt', 'slovak': 'sk',
}

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
            if model_outputs.dim() == 3:
                model_outputs = model_outputs.squeeze(1)  # Converts (1, 1, 2) → (1, 2)
            elif model_outputs.dim() == 1:
                model_outputs = model_outputs.unsqueeze(0)  # Ensure batch dimension exists

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
        if "text" in entry and "label" in entry:
            original_text = entry["text"]
            ground_truth_label = mapped_data[i]["label"]

            modified_text, _ = adversarial_attack(original_text, attack_type, lang, ground_truth_label)

            entry["text"] = modified_text

    change_percentage = (changed_words / total_words) * 100 if total_words > 0 else 0
    save_results(lang, change_percentage)
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

    # ✅ Disable GoogleLanguageModel to prevent the hang
    attack = GeneticAlgorithmAlzantot2018.build(model_wrapper, use_constraint=False)

    try:
        print("Running adversarial attack...")  # Debugging
        attack_result = attack.attack(text, ground_truth_label)
        print("Attack completed!")  # Debugging

        if isinstance(attack_result, textattack.attack_results.FailedAttackResult):
            print(f"❌ Attack failed: {text}")
            return text, 0  # Return original text

        # Get perturbed sentence
        perturbed_text = attack_result.perturbed_text()
        print(f"✅ Attack succeeded:\nOriginal: {text}\nAdversarial: {perturbed_text}")

        # Debug model output
        model_output = model_wrapper([perturbed_text])
        print(f"DEBUG: Model output shape: {model_output.shape}, Values: {model_output}")

        return perturbed_text, count_changes(text, perturbed_text)

    except IndexError as e:
        print(f"❌ IndexError encountered: {e}. Retrying with reshaped logits.")
        return text, 0  # Return original if attack fails






def textbugger_attack(text, ground_truth_label):
    """Apply TextBugger adversarial attack using a real classification model."""
    attack = TextBuggerLi2018.build(model_wrapper)
    attack_result = attack.attack(text, ground_truth_label)

    if isinstance(attack_result, textattack.attack_results.FailedAttackResult):
        return text, 0

    return attack_result.perturbed_text(), count_changes(text, attack_result.perturbed_text())

def textfooler_attack(text, ground_truth_label):
    """Apply TextFooler adversarial attack using a real classification model."""
    attack = TextFoolerJin2019.build(model_wrapper)
    attack_result = attack.attack(text, ground_truth_label)

    if isinstance(attack_result, textattack.attack_results.FailedAttackResult):
        return text, 0

    return attack_result.perturbed_text(), count_changes(text, attack_result.perturbed_text())

def count_changes(original_text, modified_text):
    """Count modified words between the original and adversarial text."""
    return sum(1 for o, m in zip(original_text.split(), modified_text.split()) if o != m)

def save_results(lang, percentage):
    """Save attack results to a file."""
    with open("output/attack_results.txt", "a", encoding="utf-8") as f:
        f.write(f"{lang}: {percentage:.2f}% words modified\n")
