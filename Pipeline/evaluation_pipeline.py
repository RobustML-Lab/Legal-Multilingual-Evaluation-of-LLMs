# Imports
from models import *
from data import *
import os

proxy_address = "http://192.168.1.100:8080"
os.environ["http_proxy"] = proxy_address
os.environ["https_proxy"] = proxy_address

# Get the dataset
dataset, task = Dataset.get_dataset('eurlex_sum')

# languages = dataset.languages
languages = ["english", "german", "french", "spanish"]

for language in languages:
    data = dataset.get_data(language)
    print(f"Length of the data: {len(data)}")
    prompt = dataset.prompt
    model = Model.get_model('ollama', dataset, multi_class=True, task=task)
    with open("../Results/Multi_Eurlex/Run2/responses.txt", "a") as file:
        file.write(f"---------------------------------------{language}--------------------------------\n")
    # Get the predicted labels
    predicted_labels = model.predict(data, prompt, language)
    true_labels = dataset.get_true_labels(data)
    with open("../Results/Multi_Eurlex/Run2/predicted_labels.txt", "a") as file:
        file.write(f"---------------------------------------{language}--------------------------------\n")
        file.write(f"P labels: {predicted_labels}\n")
        file.write(f"T labels: {true_labels}\n")
    print(f"Predicted labels: {predicted_labels}")
    print(f"True labels: {true_labels}")
    print(f"Statistics for {language}: ")

    # Evaluate the performance
    dataset.evaluate(true_labels, predicted_labels)
