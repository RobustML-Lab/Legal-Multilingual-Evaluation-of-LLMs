import json
import random
import re
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

# Download NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')

# Language-specific BERT models (for fallback/contextual support - not used here directly)
LANG_MODELS = {
    'en': 'bert-base-uncased',
    'es': 'dccuchile/bert-base-spanish-wwm-cased',
    'fr': 'camembert-base',
    'ru': 'DeepPavlov/rubert-base-cased',
    'ar': 'aubmindlab/bert-base-arabertv2',
    'zh': 'bert-base-chinese'
}

# Multilingual WordNet language mapping
wordnet_lang_map = {
    'en': 'eng', 'es': 'spa', 'fr': 'fra', 'ru': 'rus', 'ar': 'arb', 'zh': 'cmn'
}

# POS mapping for better synonym matching
pos_map = {
    "NN": wordnet.NOUN, "NNS": wordnet.NOUN,
    "VB": wordnet.VERB, "VBD": wordnet.VERB, "VBG": wordnet.VERB,
    "VBN": wordnet.VERB, "VBP": wordnet.VERB, "VBZ": wordnet.VERB,
    "JJ": wordnet.ADJ, "JJR": wordnet.ADJ, "JJS": wordnet.ADJ,
    "RB": wordnet.ADV, "RBR": wordnet.ADV, "RBS": wordnet.ADV,
}


# Get synonyms using multilingual WordNet
def get_synonyms(word, pos, lang_code):
    synonyms = set()
    try:
        synsets = wordnet.synsets(word, pos=pos, lang=lang_code)
        for synset in synsets:
            for lemma in synset.lemmas(lang=lang_code):
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
    except:
        pass
    return list(synonyms)


# Language-aware synonym substitution using WordNet
def synonym_replace(text, lang='en', percent=0.25):
    lang_code = wordnet_lang_map.get(lang, 'eng')

    # Tokenize sentences using default (English) punkt tokenizer
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = [text]

    # Tokenize words — crude fallback for non-space-separated langs like zh
    if lang == 'zh':
        words = list(text)  # crude char-level tokenization
    else:
        words = [w for sent in sentences for w in word_tokenize(sent)]

    tagged_words = pos_tag(words, lang='eng')  # POS tagging only works for English

    num_to_replace = max(1, int(len(tagged_words) * percent))
    random.shuffle(tagged_words)
    replaced = 0
    replaced_words = {}

    for word, tag in tagged_words:
        if replaced >= num_to_replace:
            break
        if tag in pos_map:
            pos = pos_map[tag]
            synonyms = get_synonyms(word, pos, lang_code)
            if synonyms:
                replaced_words[word] = random.choice(synonyms)
                replaced += 1

    # Reconstruct sentence with replacements
    new_words = [replaced_words.get(word, word) for word, _ in tagged_words]
    return ' '.join(new_words)


# Load and augment JSONL data
input_path = 'sampled_sdgs_one_label.jsonl'
output_path = 'sampled_sdgs_attack_synonym_wordnet.jsonl'
languages = ['en', 'ru', 'ar', 'es', 'fr', 'zh']

with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        record = json.loads(line)
        for lang in languages:
            if lang in record and isinstance(record[lang], str):
                try:
                    record[f'{lang}_augmented'] = synonym_replace(record[lang], lang=lang, percent=0.25)
                except Exception as e:
                    print(f"Error processing language '{lang}' for record: {e}")
        json.dump(record, outfile, ensure_ascii=False)
        outfile.write('\n')
