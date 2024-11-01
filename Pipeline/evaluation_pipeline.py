# Imports
from models import *
from data import *

# Get the dataset
dataset = Dataset.get_dataset('xnli')

languages = dataset.languages

for language in languages:
    print(f"-----------------------------------------------{language}----------------------------------------------")
    # Initialize the model
    data, label_options = dataset.get_data(language)
    print(f"Length of the data: {len(data)}")
    prompt = dataset.prompt
    model = Model.get_model('ollama', dataset, multi_class=True)

    # Get the predicted labels
    predicted_labels = model.predict(data, prompt, language)
    true_labels = dataset.get_true_labels(data)
    print(f"Predicted labels: {predicted_labels}")
    print(f"True labels: {true_labels}")
    print(f"Statistics for {language}: ")

    # Evaluate the performance
    dataset.evaluate(true_labels, predicted_labels)
