#%%
# Imports
import sys
import ast
import os

from models import *
from data import *
from llm_judge import *
from adversarial_attack import attack
from utils import store_predicted, store_attack

dataset_name = "xnli"
languages = ["th"]
points_per_language = 100
generation = 0
model_name = "ollama"
api_key = None
llm_judge_key = None
adversarial_attack = 0
iterations = 15

# Redirect print traffic to both the file and terminal
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for s in self.streams:
            s.write(message)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

log_file = open("output/log.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

# Arguments
# arguments = sys.argv[1:]
# dataset_name = arguments[0]
# languages = ast.literal_eval(arguments[1])
# points_per_language = int(arguments[2])
# generation = bool(int(arguments[3]))
# model_name = arguments[4]
# api_key = None
# adversarial_attack = int(arguments[5])
# llm_judge_key = arguments
# if llm_judge_key == 'None':
#     llm_judge_key = None
# if model_name == 'google':
#     api_key = arguments[7]

llm_judge = None
if llm_judge_key:
    llm_judge = JudgeEvaluator(llm_judge_key)

# Get the dataset
dataset = Dataset.get_dataset(dataset_name, llm_judge)
complete_predictions = []
for i in range(iterations):
    print(f"Iteration {i}")
    print("--------------------------------------------------------------------------------------")
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
        model = Model.get_model(model_name, label_options, multi_class=False, api_key=api_key, generation=generation)

        if adversarial_attack:
            # mapped_data = dataset.get_mapped_data(data)
            mapped_data = None
            before_attack = [entry["text"] for entry in data[:5]]
            data = attack(data, adversarial_attack, lang, mapped_data)
            after_attack = [entry["text"] for entry in data[:5]]
            store_attack(before_attack, after_attack, lang, dataset_name, points_per_language, model_name, adversarial_attack)

        # Get the predicted labels
        predicted, first_ten_answers = model.predict(data, prompt, lang)

        # Extract the predicted labels from the generated text
        # TODO put the extract labels method in the dataset file
        # if not generation:
        #     predicted = dataset.extract_labels_from_generated_text(predicted)

        # Get the true labels/text
        true = dataset.get_true(data)

        # Create a file with the answers
        store_predicted(predicted, true, lang, dataset_name, points_per_language, model_name, adversarial_attack)

        # Extract questions from data if available
        questions = [item.get("question") for item in data] if "question" in data[0] else None

        filtered_true = []
        filtered_predicted = []
        filtered_questions = []

        # Remove any inconsistencies
        for i in range(len(true)):
            if true[i] is not None and predicted[i] is not None:
                filtered_true.append(true[i])
                filtered_predicted.append(predicted[i])
                if questions:
                    filtered_questions.append(questions[i])

        # Print missing counts
        missing_in_true = sum(1 for ref in true if ref is None)
        missing_in_predicted = sum(1 for pred in predicted if pred is None)

        if missing_in_true or missing_in_predicted:
            print(f"Number of missing values in 'true': {missing_in_true}")
            print(f"Number of missing values in 'predicted': {missing_in_predicted}")

        if questions:
            results[lang] = dataset.evaluate(filtered_true, filtered_predicted, questions)
        else:
            print(filtered_true)
            print(filtered_predicted)
            results[lang] = dataset.evaluate(filtered_true, filtered_predicted)
        all_true[lang] = filtered_true
        all_predicted[lang] = filtered_predicted
        predicted_points[lang] = predicted
        true_points[lang] = true

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
    print("Unfiltered predicted points: ", predicted_points)
    print("Unfiltered true points: ", true_points)
    complete_predictions.append(predicted_points)
    try:
        dataset.evaluate_results(results, all_true, all_predicted)
    except AttributeError as e:
        print(e)
print("Complete predictions: ", complete_predictions)




