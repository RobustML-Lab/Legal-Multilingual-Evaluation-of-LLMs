from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import load_dataset
from transformers import pipeline
import numpy as np

# Load the Multi-EURLEX dataset
dataset = load_dataset('multi_eurlex', 'all_languages', split='test')

# Define label options (21 categories)
label_options = [
    "POLITICS", "INTERNATIONAL RELATIONS", "EUROPEAN UNION", "LAW", "ECONOMICS",
    "TRADE", "FINANCE", "SOCIAL QUESTIONS", "EDUCATION AND COMMUNICATIONS", "SCIENCE",
    "BUSINESS AND COMPETITION", "EMPLOYMENT AND WORKING CONDITIONS", "TRANSPORT",
    "ENVIRONMENT", "AGRICULTURE, FORESTRY AND FISHERIES", "AGRI-FOODSTUFFS",
    "PRODUCTION, TECHNOLOGY AND RESEARCH", "ENERGY", "INDUSTRY", "GEOGRAPHY",
    "INTERNATIONAL ORGANISATIONS"
]

# Preprocess the dataset and map true labels to category numbers
def preprocess_dataset(dataset, label_options):
    preprocessed_data = []
    for item in dataset:
        text = item['text']['en']  # Extract English text
        labels = item['labels']    # True label numbers
        preprocessed_data.append({"text": text, "labels": labels})
    return preprocessed_data[:150]

# Load a zero-shot classification pipeline using Hugging Face
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Classify text using zero-shot classification with BART-MNLI
def classify_text_with_zero_shot(text, label_options, threshold=0.5):
    classification = classifier(text, candidate_labels=label_options, multi_label=True)
    predicted_labels = [label for label, score in zip(classification['labels'], classification['scores']) if score > threshold]
    print(classification['labels'], classification['scores'])
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
        print(predicted_label_indices)

        # Store true and predicted labels for comparison
        all_true_labels.append(true_labels)
        all_predicted_labels.append(predicted_label_indices)

    return all_true_labels, all_predicted_labels

# Preprocess the dataset
preprocessed_data = preprocess_dataset(dataset, label_options)

# Run the evaluation on the preprocessed dataset
true_labels, predicted_labels = evaluate_classifier_on_dataset(preprocessed_data, label_options)

# Convert true and predicted labels to binary format using MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=list(range(len(label_options))))

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

# Print the results
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
