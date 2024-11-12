# Imports
from models import *
from data import *

# Get the dataset
dataset = Dataset.get_dataset('xnli')

# languages = dataset.languages
languages = ["ar", "el", "fr", "ru", "en", "th"]

for language in languages:
    data, label_options = dataset.get_data(language)
    print(f"Length of the data: {len(data)}")
    prompt = dataset.prompt
    model = Model.get_model('ollama', dataset, multi_class=True)
    with open("responses.txt", "a") as file:
        file.write(f"---------------------------------------{language}--------------------------------\n")
    # Get the predicted labels
    predicted_labels = model.predict(data, prompt, language)
    true_labels = dataset.get_true_labels(data)
    with open("predicted_labels.txt", "a") as file:
        file.write(f"---------------------------------------{language}--------------------------------\n")
        file.write(f"P labels: {predicted_labels}\n")
        file.write(f"T labels: {true_labels}\n")
    print(f"Predicted labels: {predicted_labels}")
    print(f"True labels: {true_labels}")
    print(f"Statistics for {language}: ")

    # Evaluate the performance
    dataset.evaluate(true_labels, predicted_labels)
