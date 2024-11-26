import csv
import json
import os
import re

from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from translator import translate
import numpy as np
import evaluate
import textwrap


class Dataset:
    """
    Base Dataset class with a.py factory method to return the appropriate dataset object.
    """

    def get_data(self, language, dataset_name, points_per_language):
        """
        Abstract method to get data in a.py specific language.
        This should be implemented by child classes.
        """
        raise NotImplementedError("Child class must implement this method")

    def get_true(self, data):
        """
        Abstract method to get the true labels/text for the dataset.
        This should be implemented by child classes.
        """
        raise NotImplementedError("Child class must implement this method")

    @staticmethod
    def get_dataset(name):
        """
        :param name: name of the dataset
        :return: the dataset object
        """
        if name.lower() == 'multi_eurlex':
            return Multi_Eurlex()
        elif name.lower() == 'eur_lex_sum':
            return Eur_Lex_Sum()
        else:
            raise ValueError(f"Dataset '{name}' is not available")


class Multi_Eurlex(Dataset):
    """
    Child class of Dataset representing the Multi-EUR-Lex dataset.
    """

    label_options = None

    def __init__(self):
        self.prompt = ("<|endoftext|>Question: Which of the following labels apply? Only answer with the numbers of "
                       "the labels that are relevant and no"
                       "further explanation! (You can select more than one): ")

    def load_label_options(self, lang_code):
        with open("output/eurovoc_categories.json", "r", encoding="utf-8") as file:
            # Load the JSON data
            eurovoc_data = json.load(file)

            # Retrieve categories for the specified language
            categories = eurovoc_data.get(lang_code, [])

            # Format as a.py lowercase list for label_options
            label_options = [option.lower() for option in categories]
            return label_options

    def get_data(self, language, dataset_name, points_per_language):
        """
        :param language: the language for which data should be retrieved
        :return: the data corresponding to the language parameter
        """
        self.label_options = self.load_label_options(language)
        dataset = load_dataset(dataset_name, language, split='test', trust_remote_code=True)
        if language == 'all_languages':
            data = self.extract_text_all_languages(dataset)
        else:
            data = self.extract_text(dataset)
        inst = translate(language, self.prompt)
        return data[:points_per_language], self.label_options, inst

    def extract_text_all_languages(self, dataset):
        """
        :param dataset: the dataset containing the text data
        :return: a.py list of text data from all languages
        """
        data = []
        for item in dataset:
            documents = item['text']
            texts = documents.keys()
            data.append({"text:": text, "labels": item['labels']} for text in texts)
        return data

    def extract_text(self, dataset):
        """
        :param dataset: the dataset containing the text data
        :return: a.py list of text data in the specified language
        """
        preprocessed_data = []
        for item in dataset:
            text = item['text']  # Extract text
            labels = item['labels']  # True label numbers
            preprocessed_data.append({"text": text, "labels": labels})
        return preprocessed_data

    def get_true(self, data):
        """
        :return: a.py list of true labels for the dataset
        """
        true_labels = [entry['labels'] for entry in data]
        return true_labels

    def extract_labels_from_generated_text(self, generated_texts):
        """
        :param generated_text: the generated text
        :param label_options: the list of label options
        :return: a list of predicted labels for the generated text
        """
        all_labels = []
        for text in generated_texts:
            labels = []
            for i in range(21):
                # Use regex to match only whole words for each index, avoiding partial matches
                if re.search(rf'\b{i}\b', text):
                    labels.append(i)
            all_labels.append(labels)

        return all_labels

    def evaluate(self, true_labels, predicted_labels):
        mlb = MultiLabelBinarizer(classes=list(range(len(self.label_options))))

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

        return {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Length": len(true_labels)
        }

    def evaluate_results(self, results, all_true, all_predicted):
        # Print out the results for each language
        for lang, metrics in results.items():
            print(f"Results for {lang}:")
            print(f"Precision: {metrics['Precision']}")
            print(f"Recall: {metrics['Recall']}")
            print(f"F1 Score: {metrics['F1 Score']}")
            print(f"Length: {metrics['Length']}")
            print("ENDMETRICS")
            true_labels = all_true[lang]
            predicted_labels = all_predicted[lang]
            for idx, label in enumerate(self.label_options):
                tp = sum([1 for true, pred in zip(true_labels, predicted_labels) if idx in pred and idx in true])
                fp = sum([1 for true, pred in zip(true_labels, predicted_labels) if idx in pred and idx not in true])
                fn = sum([1 for true, pred in zip(true_labels, predicted_labels) if idx not in pred and idx in true])
                true_num = sum([1 for true in true_labels if idx in true])
                predicted_num = sum([1 for true in predicted_labels if idx in true])

                # Precision and Recall calculations
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                print(f"{label} Precision: {precision}")
                print(f"{label} Recall: {recall}")
                print(f"{label} F1 Score: {f1}")
                print(f"True Num: {true_num}")
                print(f"Predicted Num: {predicted_num}")
                print("ENDCLASS")
            print("ENDLANGUAGE")

    def save_first_10_results_to_file_by_language(self, first_ten_answers, true_labels, predicted_labels, label_options,
                                                  language):
        # Define the output folder path
        output_folder = "output/10_first"

        # Create the directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Create a.py filename specific to the language within the output folder
        filename = os.path.join(output_folder, f"gemini_results_{language}.txt")

        # Check if the file exists; if not, create it and write headers
        if not os.path.exists(filename):
            with open(filename, 'w', encoding='utf-8') as file:
                file.write("Text\tTrue Labels\tPredicted Labels\n")

        # Write the first 10 samples' text, true labels, and predicted labels to the file
        with open(filename, 'a', encoding='utf-8') as file:
            for i in range(min(10, len(first_ten_answers))):  # Ensure we don't go out of bounds
                text = first_ten_answers[i]
                true_label_names = [label_options[idx] for idx in true_labels[i]]
                predicted_label_names = [label_options[idx] for idx in predicted_labels[i]]

                # Format the data to write
                file.write(f"{text}\t{', '.join(true_label_names)}\t{', '.join(predicted_label_names)}\n\n\n")


class Eur_Lex_Sum(Dataset):
    """
    Child class of Dataset representing the Eur-Lex-sum dataset.
    """

    def __init__(self):
        self.prompt = "\n<|endoftext|>\nTask: Summarize the text above. Include all the important information."

    def get_data(self, language, dataset_name, points_per_language):
        """
        :param language: the language for which data should be retrieved
        :return: the data corresponding to the language parameter
        """
        dataset = load_dataset('dennlinger/eur-lex-sum', language, streaming=True, split='train', trust_remote_code=True)
        self.language = language
        data = self.extract_text(dataset, points_per_language)
        inst = translate(language, self.prompt)
        print(len(data))
        return data, inst

    def extract_text(self, dataset, points_per_language):
        """
        :param dataset: the dataset containing the text data
        :return: a list of text data in the specified language
        """
        data = []
        count = 0
        for item in dataset:
            if count == points_per_language:
                break
            data.append({"text": item['reference'], "summary": item['summary']})
            count += 1
        return data

    def get_true(self, data):
        """
        :return: the true summary of the data
        """
        summary = [entry['summary'] for entry in data]
        return summary

    def format_text_to_width(self, text, width):
        """
        Splits a text into lines of a given width.
        """
        return "<br>".join(textwrap.wrap(text, width))

    def evaluate(self, references, predictions):
        rouge = evaluate.load("rouge")

        results = rouge.compute(predictions=predictions, references=references)

        file_path = "output/Eur_Lex_Sum_evaluation.md"
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', encoding='utf-8') as f:
            if not file_exists:
                f.write("| Language | Reference Summary                          | Predicted Summary                           |\n")
                f.write("|----------|------------------------------------------|--------------------------------------------|\n")
            count = 0
            for reference, prediction in zip(references, predictions):
                # Wrap text to fit within 50 characters
                formatted_reference = self.format_text_to_width(reference, 50)
                formatted_prediction = self.format_text_to_width(prediction, 50)
                # Write formatted text into md table
                f.write(f"| {self.language} | {formatted_reference} | {formatted_prediction} |\n")
                count += 1
                if count == 3:
                    break

        return results

    def evaluate_results(self, results, all_true, all_predicted):
        # Print out the results for each language
        for lang, metrics in results.items():
            print(f"Results for {lang}:")
            print(f"Rouge1: {metrics['rouge1']}")
            print(f"Rouge2: {metrics['rouge2']}")
            print(f"RougeL: {metrics['rougeL']}")
            print(f"RougeL sum: {metrics['rougeLsum']}")
            print("-------------------------------------------------------------")
