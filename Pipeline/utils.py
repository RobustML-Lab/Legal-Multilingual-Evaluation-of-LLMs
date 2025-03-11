import random
import string

from transformers import BertTokenizer, BertModel
import torch

def preturb_prompt(prompt):
    """
    Perturb the prompt to increase the difficulty of the task.
    :param prompt: the original prompt
    :return: the perturbed prompt
    """
    characters = list(set(prompt))
    random_character = random.choice(characters)
    available_characters = list(set(string.printable)-{random_character})
    replacement = random.choice(available_characters)
    perturbed_prompt = prompt.replace(random_character, replacement)
    return perturbed_prompt

def get_embedding_bert(text):
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")  # Supports multiple languages
    model = BertModel.from_pretrained("bert-base-multilingual-cased")
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1)
