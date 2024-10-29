# Imports
import sys
import ast

from models import *
from data import *

arguments = sys.argv[1:]

dataset_name = arguments[0]
languages = ast.literal_eval(arguments[1])
points_per_language = int(arguments[2])

model_name = arguments[3]
api_key = None
if model_name == 'google':
    api_key = arguments[4]

# Get the dataset
dataset = Dataset.get_dataset(dataset_name)

results = {}
all_true = {}
all_predicted = {}

for lang in languages:
    data, label_options, prompt, txt = dataset.get_data(lang, dataset_name, points_per_language)
    model = Model.get_model(model_name, label_options, multi_class=True, api_key=api_key)

    # Get the predicted labels
    predicted_labels, first_ten_answers = model.predict(data, prompt, txt)
    true_labels = dataset.get_true_labels(data)

    results[lang] = dataset.evaluate(true_labels, predicted_labels)
    all_true[lang] = true_labels
    all_predicted[lang] = predicted_labels

    if first_ten_answers is not None:
        dataset.save_first_10_results_to_file_by_language(first_ten_answers, true_labels, predicted_labels, label_options, lang)

dataset.evaluate_results(results, all_true, all_predicted)