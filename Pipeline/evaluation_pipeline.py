import sys
import ast

from models import *
from data import *
from llm_judge import *
from adversarial_attack import attack
from utils import store_predicted, store_attack

# dataset_name = "eur_lex_sum"
# languages = ["english"]
# points_per_language = 1
# generation = True
# model_name = "llama"
# api_key = None
# llm_judge_key = None
# adversarial_attack = 0

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

log_file = open("output/log.txt", "w")
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

# Arguments
arguments = sys.argv[1:]
dataset_name = arguments[0]
languages = ast.literal_eval(arguments[1])
points_per_language = int(arguments[2])
generation = bool(int(arguments[3]))
model_name = arguments[4]
api_key = None
adversarial_attack = int(arguments[5])
llm_judge_key = arguments[6]
if llm_judge_key == 'None':
    llm_judge_key = None
if model_name == 'google':
    api_key = arguments[7]

llm_judge = None
if llm_judge_key:
    llm_judge = JudgeEvaluator(llm_judge_key)

# Get the dataset
dataset = Dataset.get_dataset(dataset_name, llm_judge)

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

    if adversarial_attack:
        # mapped_data = dataset.get_mapped_data(data)
        mapped_data = None
        before_attack = [entry["text"] for entry in data[:5]]
        data = attack(data, adversarial_attack, lang, mapped_data)
        after_attack = [entry["text"] for entry in data[:5]]
        store_attack(before_attack, after_attack, lang, dataset_name, points_per_language, model_name, adversarial_attack)

    # Get the predicted labels
    predicted, first_ten_answers = model.predict(data, prompt)

    # Extract the predicted labels from the generated text
    if not generation:
        predicted = dataset.extract_labels_from_generated_text(predicted)

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
        results[lang] = dataset.evaluate(filtered_true, filtered_predicted)
    all_true[lang] = filtered_true
    all_predicted[lang] = filtered_predicted

    if dataset_name.lower() == 'multi_eurlex':
        dataset.save_first_10_results_to_file_by_language(first_ten_answers, lang)

try:
    dataset.evaluate_results(results)
except AttributeError as e:
    print(e)
