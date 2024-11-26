# Imports
import sys
import ast

from models import *
from data import *

arguments = sys.argv[1:]

dataset_name = arguments[0]
languages = ast.literal_eval(arguments[1])
points_per_language = int(arguments[2])
generation = bool(int(arguments[3]))

model_name = arguments[4]
api_key = None
if model_name == 'google':
    api_key = arguments[5]

# Get the dataset
dataset = Dataset.get_dataset(dataset_name)

results = {}
all_true = {}
all_predicted = {}


for lang in languages:
    if generation:
        data, prompt = dataset.get_data(lang, dataset_name, points_per_language)
        label_options = None
    else:
        data, label_options, prompt = dataset.get_data(lang, dataset_name, points_per_language)
    model = Model.get_model(model_name, label_options, multi_class=True, api_key=api_key, generation=generation)

    # Get the predicted labels
    predicted, first_ten_answers = model.predict(data, prompt)

    if not generation:
        predicted = dataset.extract_labels_from_generated_text(predicted)

    true = dataset.get_true(data)

    filtered_true = [x for x in true if x is not None]
    filtered_predicted = [predicted[i] for i in range(len(true)) if true[i] is not None]

    # Count missing values in filtered_true and filtered_predicted
    missing_in_true = sum(1 for ref in filtered_true if ref is None)
    missing_in_predicted = sum(1 for pred in filtered_predicted if pred is None)

    # Replace None with empty strings
    filtered_true = [ref if ref is not None else "" for ref in filtered_true]
    filtered_predicted = [pred if pred is not None else "" for pred in filtered_predicted]

    # Print the counts
    if missing_in_true or missing_in_predicted:
        print(f"Number of missing values in 'filtered_true': {missing_in_true}")
        print(f"Number of missing values in 'filtered_predicted': {missing_in_predicted}")

    results[lang] = dataset.evaluate(filtered_true, filtered_predicted)
    all_true[lang] = true
    all_predicted[lang] = predicted

    if not generation:
        dataset.save_first_10_results_to_file_by_language(first_ten_answers, true, predicted, label_options, lang)

dataset.evaluate_results(results, all_true, all_predicted)






