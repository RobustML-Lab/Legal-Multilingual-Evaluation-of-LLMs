#%%
# Imports
import sys
import ast
import os

from models import *
from data import *
from adversarial_attack import attack
from huggingface_hub import login

dataset_name = "xnli"
languages = ["bg", "el", "en", "es", "fr", "th"]
points_per_language = 300
generation = False
model_name = "ollama"
api_key = None
adversarial_attack = 0
p = 0.1
# arguments = sys.argv[1:]
# dataset_name = arguments[0]
# languages = ast.literal_eval(arguments[1])
# points_per_language = int(arguments[2])
# generation = bool(int(arguments[3]))
# model_name = arguments[4]
# api_key = None
# adversarial_attack = int(arguments[5])
# if model_name == 'google':
#     api_key = arguments[6]
#%%
# Get the dataset
dataset = Dataset.get_dataset(dataset_name)

results = {}
all_true = {}
all_predicted = {}
final_results = {}
data_points = []
predicted_points = {}
true_points = {}
for lang in languages:
    if generation:
        data, prompt = dataset.get_data(lang, dataset_name, points_per_language)
        label_options = None
    else:
        data, label_options, prompt = dataset.get_data(lang, dataset_name, points_per_language)
    model = Model.get_model(model_name, label_options, multi_class=True, api_key=api_key, generation=generation)

    os.makedirs(os.path.dirname("output/results.json"), exist_ok=True)

    if os.path.exists("output/results.json"):
        with open("output/results.json", "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    before_attack = [entry["text"] for entry in data[:5]]
    after_attack = []

    if adversarial_attack:
        # mapped_data = dataset.get_mapped_data(data)
        mapped_data = None
        data = attack(data, adversarial_attack, lang, mapped_data, p)
        after_attack = [entry["text"] for entry in data[:5]]

    result_entry = {
        "before_attack": before_attack,
        "after_attack": after_attack
    }

    existing_data.append(result_entry)

    with open("output/results.json", "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)


    # Get the predicted labels
    predicted, first_ten_answers = model.predict(data, prompt, lang)

    # if not generation:
    #     predicted = dataset.extract_labels_from_generated_text(predicted)

    true = dataset.get_true(data)

    predicted = np.concatenate([np.array(sublist) for sublist in predicted])

    filtered_true = [true[i] for i in range(len(true)) if true[i] is not None and predicted[i] is not None]
    filtered_predicted = [predicted[i] for i in range(len(true)) if true[i] is not None and predicted[i] is not None]


    # Count missing values in filtered_true and filtered_predicted
    missing_in_true = sum(1 for ref in filtered_true if ref is None)
    missing_in_predicted = sum(1 for pred in filtered_predicted if pred is None)
    print("Missing predicted:", missing_in_predicted)
    print("Filtered true:", filtered_true)
    print("Filtered predicted:", filtered_predicted)
    if len(filtered_predicted) == 0:
        print("All predicted values are missing. Skipping this language.")
        continue

    # Convert back to lists
    filtered_true = list(filtered_true)
    filtered_predicted = list(filtered_predicted)
    # filtered_predicted = np.concatenate([np.array(sublist) for sublist in filtered_predicted])

    # Filter both lists together
    filtered_true, filtered_predicted = zip(*[
        (ref, pred) for ref, pred in zip(filtered_true, filtered_predicted) if ref is not None and pred is not None
    ])

    # # Convert back to lists
    filtered_true = list(filtered_true)
    filtered_predicted = list(filtered_predicted)
    # filtered_predicted = np.concatenate([np.array(sublist) for sublist in filtered_predicted])

    print("True values: ", true)
    print("Fitered true values: ", filtered_true)
    print("Predicted values: ", predicted)
    print("Filtered predicted values: ", filtered_predicted)
    print("Data points: ", len(filtered_true))

    data_points.append(len(filtered_true))
    predicted_points[lang] = predicted
    true_points[lang] = true

    # Print the counts
    if missing_in_true or missing_in_predicted:
        print(f"Number of missing values in 'filtered_true': {missing_in_true}")
        print(f"Number of missing values in 'filtered_predicted': {missing_in_predicted}")

    results[lang] = dataset.evaluate(filtered_true, filtered_predicted)
    all_true[lang] = filtered_true
    all_predicted[lang] = filtered_predicted

    if model_name.lower() == 'multi_eurlex':
        dataset.save_first_10_results_to_file_by_language(first_ten_answers, true, predicted, label_options, lang)
    print(results)
    min_accuracy = results[lang]
    l = final_results.get(lang, [])
    l.append(min_accuracy)
    final_results[lang] = l

print("Final results: ", final_results)
print("Data points: ", data_points)
print("Predicted points: ", predicted_points)
print("True points: ", true_points)

try:
    dataset.evaluate_results(results, all_true, all_predicted)
except AttributeError as e:
    print(e)





