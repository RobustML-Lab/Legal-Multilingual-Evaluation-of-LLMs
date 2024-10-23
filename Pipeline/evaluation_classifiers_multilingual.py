from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import load_dataset
from transformers import pipeline
from deep_translator import GoogleTranslator
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor  # For parallelism

device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, otherwise CPU

# Load the Multi-EURLEX dataset
dataset = load_dataset('multi_eurlex', 'all_languages', split='test', trust_remote_code=True)

# Define label options (21 categories) in English
label_options = [
    "POLITICS", "INTERNATIONAL RELATIONS", "EUROPEAN UNION", "LAW", "ECONOMICS",
    "TRADE", "FINANCE", "SOCIAL QUESTIONS", "EDUCATION AND COMMUNICATIONS", "SCIENCE",
    "BUSINESS AND COMPETITION", "EMPLOYMENT AND WORKING CONDITIONS", "TRANSPORT",
    "ENVIRONMENT", "AGRICULTURE, FORESTRY AND FISHERIES", "AGRI-FOODSTUFFS",
    "PRODUCTION, TECHNOLOGY AND RESEARCH", "ENERGY", "INDUSTRY", "GEOGRAPHY",
    "INTERNATIONAL ORGANISATIONS"
]

# Function to translate label options to a given language using GoogleTranslator
def translate_labels(labels, target_language):
    translator = GoogleTranslator(source='en', target=target_language)
    translated_labels = [translator.translate(label) for label in labels]
    return translated_labels

# Preprocess the dataset and map true labels to category numbers
def preprocess_dataset(dataset, language, label_options):
    preprocessed_data = []
    for item in dataset:
        text = item['text'][language]  # Extract text in the specific language
        labels = item['labels']        # True label numbers
        preprocessed_data.append({"text": text, "labels": labels})
    return preprocessed_data[:1000]  # Limiting to the first 1000 samples

# Load a zero-shot classification pipeline using Hugging Face
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

# Classify text using zero-shot classification with BART-MNLI
def classify_text_with_zero_shot(text, label_options, threshold=0.5):
    classification = classifier(text, candidate_labels=label_options, multi_label=True)
    predicted_labels = [label for label, score in zip(classification['labels'], classification['scores']) if score > threshold]
    return predicted_labels

# Evaluate the classifier on the dataset
def evaluate_classifier_on_dataset(dataset, label_options):
    all_true_labels = []
    all_predicted_labels = []

    for entry in dataset:
        text = entry['text']
        true_labels = entry['labels']

        # Get predicted labels from the zero-shot classifier
        predicted_labels = classify_text_with_zero_shot(text, label_options)
        predicted_label_indices = [label_options.index(label) for label in predicted_labels if label in label_options]

        # Store true and predicted labels for comparison
        all_true_labels.append(true_labels)
        all_predicted_labels.append(predicted_label_indices)

    return all_true_labels, all_predicted_labels

# Function to process a single language (parallelized)
def process_language(lang):
    print(f"Evaluating for language: {lang}")

    # Translate the labels to the current language
    if lang != 'en':  # If not English, translate the labels
        translated_labels = translate_labels(label_options, lang)
    else:
        translated_labels = label_options  # Keep labels in English for English texts

    # Preprocess the dataset for the current language
    preprocessed_data = preprocess_dataset(dataset, lang, translated_labels)

    # Run the evaluation on the preprocessed dataset
    true_labels, predicted_labels = evaluate_classifier_on_dataset(preprocessed_data, translated_labels)

    # Convert true and predicted labels to binary format using MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=list(range(len(translated_labels))))

    # Binarize the true and predicted labels
    binary_true = mlb.fit_transform(true_labels)
    binary_pred = mlb.transform(predicted_labels)

    # Get indices of labels with non-zero true or predicted samples
    relevant_labels = np.where((binary_true.sum(axis=0) + binary_pred.sum(axis=0)) > 0)[0]

    # Filter binary_true and binary_pred to only include relevant labels
    filtered_binary_true = binary_true[:, relevant_labels]
    filtered_binary_pred = binary_pred[:, relevant_labels]

    # Calculate precision, recall, F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        filtered_binary_true, filtered_binary_pred, average='macro', zero_division=0
    )

    # Return the results for the current language
    return lang, {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

# Languages available in the dataset
languages = ['en', 'de', 'fr', 'it', 'es', 'pl', 'ro', 'nl', 'el', 'hu', 'pt', 'cs',
             'sv', 'bg', 'da', 'fi', 'sk', 'lt', 'hr', 'sl', 'et', 'lv', 'mt']

# Dictionary to store results for each language
results = {}

# Use ThreadPoolExecutor to parallelize the processing of multiple languages
with ThreadPoolExecutor(max_workers=23) as executor:  # Adjust max_workers to the number of CPUs available
    future_to_lang = {executor.submit(process_language, lang): lang for lang in languages}

    for future in future_to_lang:
        lang, metrics = future.result()
        results[lang] = metrics

# Print the results for each language
for lang, metrics in results.items():
    print(f"Results for {lang}:")
    print(f"Precision: {metrics['Precision']}")
    print(f"Recall: {metrics['Recall']}")
    print(f"F1 Score: {metrics['F1 Score']}")
    print("------------------------------------")
