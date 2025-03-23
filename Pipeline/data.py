import copy
import json

import os
import csv

import unicodedata
from datasets import load_dataset
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer
from translator import translate
import numpy as np
import textwrap

import re
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from deep_translator import GoogleTranslator
import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rapidfuzz import fuzz

from utils import get_embedding_bert, get_language_from_code, store_judge
from sklearn.metrics.pairwise import cosine_similarity



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
    def get_dataset(name, llm_judge):
        """
        :param name: name of the dataset
        :return: the dataset object
        """
        if name.lower() == 'multi_eurlex':
            return Multi_Eurlex()
        elif name.lower() == 'go_emotions':
            return Go_Emotions()
        elif name.lower() == 'casehold':
            return CaseHOLD()
        elif name.lower() == 'xnli':
            return XNLI()
        elif name.lower() == 'eur_lex_sum':
            return Eur_Lex_Sum()
        # elif name.lower() == 'multi_legal_pile':
        #     return Multi_Legal_Pile()
        elif name.lower() == 'europa_random_split':
            return Europa_Random_Split()
        elif name.lower() == 'xquad':
            return XQuAD(llm_judge)
        elif name.lower() == 'sst2':
            return SST2()
        elif name.lower() == 'qqp':
            return QQP()
        elif name.lower() == 'mnli':
            return MNLI()
        elif name.lower() == 'qnli':
            return QNLI()
        else:
            raise ValueError(f"Dataset '{name}' is not available")

    def normalize_text(self, text):
        # Convert to lowercase and remove accents
        text = text.lower()
        return ''.join(
            c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn'
        )

    def extract_labels_from_generated_text(self, generated_text, label_options):
        cleaned_text = self.normalize_text(generated_text.replace("\u200B", ""))
        relevant_labels = []
        for i, label in enumerate(label_options):
            cleaned_label = self.normalize_text(label.replace("\u200B", ""))
            # Use \b to ensure the label is a.py standalone word or phrase
            pattern = r'\b' + re.escape(cleaned_label) + r'\b'
            if re.search(pattern, cleaned_text, re.IGNORECASE):
                relevant_labels.append(i)
        return relevant_labels


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
        count = 0
        for item in dataset:
            if count == 5:
                break
            documents = item['text']
            texts = documents.keys()
            data.append({"text:": text, "labels": item['labels']} for text in texts)
            count += 1
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
        # self.prompt = "\n<|endoftext|>\nTask: Write something about the text above."

    def get_data(self, language, dataset_name, points_per_language):
        """
        :param language: the language for which data should be retrieved
        :return: the data corresponding to the language parameter
        """
        print("Reached get_data")
        dataset = load_dataset('dennlinger/eur-lex-sum', language, streaming=True, split='train', trust_remote_code=True)
        self.language = language
        data = self.extract_text(dataset, points_per_language)
        inst = translate(language, self.prompt)
        return data, inst[0]

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
        rouge = evaluate.load("rouge", cache_dir=f"/tmp/huggingface_cache/{os.getpid()}")

        results_rouge = rouge.compute(predictions=predictions, references=references)
        embedded_references = [get_embedding_bert(reference) for reference in references]
        embedded_predictions = [get_embedding_bert(prediction) for prediction in predictions]
        cosine_similarities = [cosine_similarity(embedded_reference, embedded_prediction) for (embedded_reference, embedded_prediction) in zip(embedded_references, embedded_predictions)]
        avg_cosine_similarity = np.mean(cosine_similarities)
        results_cosine = {"cosine_similarity": avg_cosine_similarity}
        results = results_rouge | results_cosine

        # Store the first 3 reference and predicted for checking
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
            print(f"Cosine Similarity: {metrics['cosine_similarity']}")
            print("-------------------------------------------------------------")


    def extract_labels_from_generated_text(self, generated_text, label_options):
        """
        :param generated_text: the generated text
        :param label_options: the list of label options
        :return: a list of predicted labels for the generated text
        """
        labels = []
        for i in range(len(label_options)):
            # Use regex to match only whole words for each index, avoiding partial matches
            if re.search(rf'\b{i}\b', generated_text):
                labels.append(i)

        return labels


class Go_Emotions(Dataset):
    """
    Child class of Dataset representing the GoEmotions dataset.
    """

    def __init__(self):
        self.label_options = [
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "realization", "relief", "remorse", "sadness", "surprise"
        ]
        self.prompt = "<|endoftext|>" + (
                "Question: Which of the following emotions apply to this text? (You can select more than one): "
                + ', '.join(self.label_options) + " "
                                                  "Answer:"
        )

    def get_data(self, language=None):
        """
        Loads the GoEmotions dataset.
        :return: the data and label options
        """
        dataset = load_dataset('go_emotions', split='test')
        return self.extract_text(dataset)

    def extract_text(self, dataset):
        """
        Extracts and formats the data from the GoEmotions dataset.
        :param dataset: the dataset containing the text data
        :return: a list of text data and labels
        """
        data = []
        count = 0
        for item in dataset:
            if count == 50:
                break
            count += 1
            data.append({"text": item['text'], "labels": item['labels']})
        return data

    def get_true_labels(self, data):
        """
        :param data: list of data entries
        :return: list of true labels for the dataset
        """
        true_labels = [entry['labels'] for entry in data]
        return true_labels

    def evaluate(self, true_labels, predicted_labels):
        """
        Evaluates the model using precision, recall, and F1 score.
        :param true_labels: list of true labels
        :param predicted_labels: list of predicted labels
        """
        mlb = MultiLabelBinarizer(classes=list(range(len(self.label_options))))

        binary_true = mlb.fit_transform(true_labels)
        binary_pred = mlb.transform(predicted_labels)

        relevant_labels = np.where((binary_true.sum(axis=0) + binary_pred.sum(axis=0)) > 0)[0]
        filtered_binary_true = binary_true[:, relevant_labels]
        filtered_binary_pred = binary_pred[:, relevant_labels]

        precision, recall, f1, _ = precision_recall_fscore_support(
            filtered_binary_true, filtered_binary_pred, average='macro', zero_division=0
        )

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")


class CaseHOLD(Dataset):
    """
    Child class of Dataset representing the CaseHOLD dataset.
    """

    def __init__(self):
        self.label_options = ["A", "B", "C", "D", "E"]
        self.prompt = (
            "<|endoftext|> Question: Based on the case description, select the most appropriate legal answer by only "
            "stating the appropriate character:\n"
        )
        self.languages = ['en']

    def get_data(self, language=None):
        """
        Loads the CaseHOLD dataset.
        :return: the data and label options
        """
        dataset = load_dataset('lex_glue', 'case_hold', split='test')
        return self.extract_text(dataset)

    def extract_text(self, dataset):
        """
        Extracts and formats the data from the CaseHOLD dataset.
        :param dataset: the dataset containing the text data
        :return: a list of text data and labels
        """
        data = []
        count = 0
        print("Length of the dataset: ", len(dataset))
        for item in dataset:
            if count == 200:
                break
            count += 1

            # Create choices formatted with corresponding letters
            choices = "\n".join([f"{letter}) {ending}" for letter, ending in zip(self.label_options, item['endings'])])
            # Combine context and choices into the text
            text_with_choices = f"{item['context']}\n\n{choices}"

            data.append({
                "text": text_with_choices,  # Choices are now included in the text
                "label": item['label']  # Keep the label for evaluation
            })
        return data

    def get_true_labels(self, data):
        """
        :param data: list of data entries
        :return: list of true labels for the dataset
        """
        true_labels = [entry['label'] for entry in data]
        return true_labels

    def evaluate(self, true_labels, predicted_labels):
        """
        Evaluates the model using precision, recall, F1 score, and accuracy.
        :param true_labels: list of true labels
        :param predicted_labels: list of predicted labels
        """
        flat_predicted_labels = [item for sublist in predicted_labels for item in sublist]
        accuracy = accuracy_score(true_labels, flat_predicted_labels)

        print(f"Accuracy: {accuracy}")

    def extract_labels_from_generated_text(self, generated_text, label_options):
        """
        Extracts the first predicted label from the model's response.
        :param response: The model's output as a string
        :return: The first valid label (A, B, C, D, E) found in the response, or None if not found
        """
        # Find the first capital letter in the response within the range A-E
        print("Reached extract_labels in CaseHOLD class")
        match = re.search(r'\b([A-E])\b', generated_text)
        if match:
            print("Mathced response: ")
            print(match)
            return match.group(1)  # Return the first matched capital letter
        return ["F"]


class XNLI(Dataset):
    """
    Child class of Dataset representing the XNLI dataset.
    """
    def __init__(self):
        self.label_options = ["0", "1", "2"]
        self.languages = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
        self.prompt = ("<|endoftext|>\nTask: Please identify whether the premise entails or contradicts "
                           "the hypothesis, or neither. The answer should be '0' for entailment, "
                           "'1' for neither, or '2' for contradiction. The answer should be exactly '0', '1', or '2'."
                           )

    def get_data(self, language, dataset_name, points):
        """
        Loads the XNLI dataset for the specified language.
        :param language: the language of the dataset
        :return: the data and label options
        """
        dataset = load_dataset('xnli', language, split='test', trust_remote_code=True)
        self.language = language
        if language == 'all_languages':
            data = self.extract_text_all_languages(dataset)
        else:
            data = self.extract_text(dataset, points)
        return data, self.label_options, translate(language, self.prompt)[0]

    def extract_text_all_languages(self, dataset):
        """
        :param dataset: the dataset containing the text data
        :return: a list of text data from all languages
        """
        data = []
        count = 0
        for item in dataset:
            if count == 5:
                break
            documents = item['text']
            texts = documents.keys()
            data.append({"text:": text, "labels": item['labels']} for text in texts)
            count += 1

    def extract_text(self, dataset, points):
        """
        :param dataset: the dataset containing the text data
        :return: a list of text data in the specified language
        """
        data = []
        count = 0
        for item in dataset:
            if count == points:
                break
            translator = GoogleTranslator(source="en", target=self.language)
            if self.language == "ar":
                text = item["hypothesis"] + translator.translate("Hypothesis: ") + item["premise"] + translator.translate("Premise: ")
            else:
                text = translator.translate("Premise: ") + item["premise"] + translator.translate(" Hypothesis: ") + item["hypothesis"]
            data.append({"text": text, "label": item['label']})
            count += 1
        return data

    def evaluate(self, true_labels, predicted_labels):
        """
        Evaluates the model using precision, recall, F1 score, and accuracy.
        :param true_labels: list of true labels
        :param predicted_labels: list of predicted labels
        """
        accuracy = accuracy_score(true_labels, predicted_labels)
        file_path = "XNLI_evaluation.csv"
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Language", "Accuracy"])
            writer.writerow([self.language, accuracy])

        print(f"Accuracy {self.language}: {accuracy}")

    import re

    def extract_labels_from_generated_text(self, generated_texts):
        """
        Extracts the first predicted label (0, 1, or 2) from the model's response.
        :param generated_texts: List of generated model outputs.
        :return: List of extracted labels (0, 1, 2), or None if no valid label is found.
        """
        word_to_digit = {"zero": 0, "one": 1, "two": 2}  # Handle word numbers
        all_labels = []

        print(generated_texts)

        for text in generated_texts:
            if text is not None:
                text_lower = text.lower()

                # Remove punctuation for easier matching
                text_lower = re.sub(r"[^\w\s]", "", text_lower)

                # Try to find exact numbers first
                match = re.findall(r"\b(0|1|2)\b", text_lower)

                if match:
                    all_labels.append(int(match[0]))  # Extract first match
                    continue  # Skip to next iteration

                # Try matching word numbers ("zero", "one", "two")
                for word, digit in word_to_digit.items():
                    if re.search(rf"\b{word}\b", text_lower):
                        all_labels.append(digit)
                        break  # Stop after first valid match
                else:
                    all_labels.append(None)  # No valid label found
            else:
                all_labels.append(None)

        return all_labels



    def get_true(self, data):
        """
        :return: A list of true labels for the dataset
        """
        return [entry['label'] for entry in data]

# class Multi_Legal_Pile(Dataset):
#     """
#     Child class of Dataset representing the Eur-Lex-sum dataset.
#     """
#
#     def __init__(self):
#         self.prompt = "\n<|endoftext|>\nTask: Summarize the text above. Include all the important information."
#
#     def get_data(self, language, dataset_name, points_per_language):
#         """
#         :param language: the language for which data should be retrieved
#         :return: the data corresponding to the language parameter
#         """
#         config = f"{language}_legal-mc4"
#         dataset = load_dataset('joelniklaus/Multi_Legal_Pile', config, streaming=True, split='train', trust_remote_code=True)
#         limited_data = list(islice(dataset, points_per_language))
#         print(limited_data[0])
#         self.language = language
#         data = self.extract_text(limited_data, points_per_language)
#         inst = translate(language, self.prompt)
#         return data, inst[0]
#
#     def extract_text(self, dataset, points_per_language):
#         """
#         :param dataset: the dataset containing the text data
#         :return: a list of text data in the specified language
#         """
#         data = []
#         count = 0
#         for item in dataset:
#             if count == points_per_language:
#                 break
#             data.append({"text": item['reference'], "summary": item['summary']})
#             count += 1
#         return data
#
#     def get_true(self, data):
#         """
#         :return: the true summary of the data
#         """
#         summary = [entry['summary'] for entry in data]
#         return summary
#
#     def format_text_to_width(self, text, width):
#         """
#         Splits a text into lines of a given width.
#         """
#         return "<br>".join(textwrap.wrap(text, width))
#
#     def evaluate(self, references, predictions):
#         rouge = evaluate.load("rouge", cache_dir=f"/tmp/huggingface_cache/{os.getpid()}")
#
#         results = rouge.compute(predictions=predictions, references=references)
#
#         file_path = "output/Eur_Lex_Sum_evaluation.md"
#         file_exists = os.path.isfile(file_path)
#         with open(file_path, mode='a', encoding='utf-8') as f:
#             if not file_exists:
#                 f.write("| Language | Reference Summary                          | Predicted Summary                           |\n")
#                 f.write("|----------|------------------------------------------|--------------------------------------------|\n")
#             count = 0
#             for reference, prediction in zip(references, predictions):
#                 # Wrap text to fit within 50 characters
#                 formatted_reference = self.format_text_to_width(reference, 50)
#                 formatted_prediction = self.format_text_to_width(prediction, 50)
#                 # Write formatted text into md table
#                 f.write(f"| {self.language} | {formatted_reference} | {formatted_prediction} |\n")
#                 count += 1
#                 if count == 3:
#                     break
#
#         return results
#
#     def evaluate_results(self, results, all_true, all_predicted):
#         # Print out the results for each language
#         for lang, metrics in results.items():
#             print(f"Results for {lang}:")
#             print(f"Rouge1: {metrics['rouge1']}")
#             print(f"Rouge2: {metrics['rouge2']}")
#             print(f"RougeL: {metrics['rougeL']}")
#             print("-------------------------------------------------------------")


class Europa_Random_Split(Dataset):
    """
    Child class of Dataset representing the Eur-Lex-sum dataset.
    """

    def __init__(self):
        self.prompt = "\n<|endoftext|>\nTask: Give me a list of keyphrases for the text above. Only give me the keyphrases separated by a new line. Give the most important keyphrases first. Include all the important information."

    def get_data(self, language, dataset_name, points_per_language):
        """
        :param language: the language for which data should be retrieved
        :return: the data corresponding to the language parameter
        """

        print("Reached get_data")
        dataset = load_dataset('NCube/europa-random-split', streaming=True, split='train', trust_remote_code=True)
        filtered_dataset = (example for example in dataset if example["lang"] == language)
        self.language = language
        data = self.extract_text(filtered_dataset, points_per_language)
        inst = translate(language, self.prompt)
        return data, inst[0]

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
            data.append({"text": item['input_text'], "keyphrases": item['keyphrases']})
            count += 1
        return data

    def get_true(self, data):
        """
        :return: the true summary of the data
        """
        summary = [entry['keyphrases'] for entry in data]
        return summary

    def format_text_to_width(self, text, width):
        """
        Splits a text into lines of a given width.
        """
        return "<br>".join(textwrap.wrap(text, width))

    def calculate_f1(self, true_set, pred_list, k=None, threshold=80):
        """
        Calculate F1 score using fuzzy matching to account for order insensitivity.
        :param true_set: Set of true keyphrases.
        :param pred_list: List of predicted keyphrases.
        :param k: If specified, use only the top-k predictions.
        :param threshold: Fuzzy matching similarity threshold (0-100).
        :return: Precision, Recall, and F1 score.
        """
        if k:
            pred_list = pred_list[:k]

        matched_true = set()
        matched_pred = set()

        # Iterate over predicted keyphrases
        for pred in pred_list:
            # Find the best match in the true set
            for true in true_set:
                if true not in matched_true and fuzz.ratio(pred, true) >= threshold:
                    matched_true.add(true)
                    matched_pred.add(pred)
                    break

        # Calculate true positives
        true_positives = len(matched_true)

        # Calculate precision and recall
        precision = true_positives / len(pred_list) if len(pred_list) > 0 else 0.0
        recall = true_positives / len(true_set) if len(true_set) > 0 else 0.0

        # Calculate F1 score
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def calculate_map(self, true_set, pred_list, k=50, threshold=80):
        """
        Calculate Mean Average Precision (MAP) at k using fuzzy matching.
        :param true_set: Set of true keyphrases.
        :param pred_list: List of predicted keyphrases.
        :param k: Use only top-k predictions.
        :param threshold: Fuzzy matching similarity threshold (0-100).
        :return: MAP score.
        """
        if k:
            pred_list = pred_list[:k]

        matched_true = set()
        binary_relevance = []

        # Iterate over predictions to calculate binary relevance
        for pred in pred_list:
            match_found = False
            for true in true_set:
                if true not in matched_true and fuzz.ratio(pred, true) >= threshold:
                    matched_true.add(true)
                    match_found = True
                    break
            binary_relevance.append(1 if match_found else 0)

        if not binary_relevance:
            return 0.0

        # Calculate precision at each relevant index
        relevant_indices = [i + 1 for i, rel in enumerate(binary_relevance) if rel == 1]
        precisions = [sum(binary_relevance[:i]) / i for i in relevant_indices]

        # Calculate MAP
        return sum(precisions) / len(true_set) if len(true_set) > 0 else 0.0

    def evaluate(self, references, predictions):
        """
        Evaluate predictions using F1@k, F1@M, and MAP@50 for present and absent keyphrases.
        :param references: List of lists, where each sublist contains true keyphrases for an instance.
        :param predictions: List of strings, where each string contains predicted keyphrases separated by newlines.
        :return: Dictionary of evaluation metrics.
        """
        metrics = {
            "F1@5": [],
            "F1@10": [],
            "F1@M": [],
            "MAP@50": []
        }

        for ref, pred_str in zip(references, predictions):
            # Convert predictions to a list of keyphrases
            pred_list = [phrase.strip() for phrase in pred_str.split('\n') if phrase.strip()]

            # Convert references to a set for comparison
            true_set = set(ref)

            # Calculate F1@5, F1@10, and F1@M
            _, _, f1_5 = self.calculate_f1(true_set, pred_list, k=5)
            _, _, f1_10 = self.calculate_f1(true_set, pred_list, k=10)
            _, _, f1_m = self.calculate_f1(true_set, pred_list)

            # Calculate MAP@50
            map_50 = self.calculate_map(true_set, pred_list, k=50)

            # Append metrics for this instance
            metrics["F1@5"].append(f1_5)
            metrics["F1@10"].append(f1_10)
            metrics["F1@M"].append(f1_m)
            metrics["MAP@50"].append(map_50)

        # Aggregate metrics across all instances
        aggregated_metrics = {metric: sum(scores) / len(scores) if scores else 0.0 for metric, scores in metrics.items()}

        return aggregated_metrics

    def evaluate_results(self, results, all_true, all_predicted):
        """
        Display aggregated F1@k, F1@M, and MAP@50 results.

        :param results: Dictionary where each key is a language (e.g., en, el)
                        and the value is a map of metrics and scores.
        """
        print("\nAggregated Keyphrase Generation Metrics:\n")
        for language, scores in results.items():
            print(f"{language}: {scores}")
        print("-" * 40)

class XQuAD(Dataset):
    """
    Child class of Dataset representing the XQuAD dataset.
    """

    def __init__(self, llm_judge):
        self.prompt = "<|endoftext|>\nTask: Given the question and the passage, extract the most relevant answer from the passage."
        self.embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        self.llm_judge = llm_judge


    def get_data(self, language, dataset_name, points_per_language):
        """
        Loads the XQuAD dataset from Hugging Face.

        :param language: The language for which data should be retrieved (ar, de, el, en, es, hi, ro, ru, th, tr, vi, zh)
        :param dataset_name: The dataset name ('google/xquad')
        :param points_per_language: Number of samples to return
        :return: Processed dataset and prompt
        """
        dataset = load_dataset("google/xquad", f"xquad.{language}", split="validation", trust_remote_code=True)
        data = self.extract_text(dataset, points_per_language)
        inst = translate(language, self.prompt)
        self.lang = language
        return data, inst[0]

    def extract_text(self, dataset, points_per_language):
        """
        Extracts passages, questions, and answers.

        :param dataset: The dataset containing text data
        :return: List of dictionaries with context, question, and answers
        """
        data = []
        for i, item in enumerate(dataset):
            if i >= points_per_language:
                break
            data.append({
                "text": f"Passage: {item['context']}\nQuestion: {item['question']}",
                "answers": item["answers"]["text"],  # List of possible correct answers
                "question": item['question']
            })
        return data

    def get_true(self, data):
        """
        :return: A list of true answers for the dataset
        """
        return [entry["answers"] for entry in data]


    def evaluate(self, true_answers, predicted_answers, questions):
        """
        Evaluates the model's extracted answers using BLEU, METEOR, and Cosine Similarity,
        or with LLM-based scoring if self.llm_judge is enabled.

        :param true_answers: List of correct answers (each is a **list with one token**)
        :param predicted_answers: List of extracted answers (each is a full string)
        :param questions: List of questions (each is a **list with one token**)
        :return: Dictionary with BLEU, METEOR, and Cosine Similarity scores.
        """
        if len(true_answers) != len(predicted_answers):
            raise ValueError("true_answers and predicted_answers must have the same length")

        if self.llm_judge:
            prompts = []
            for true_list, pred_answer, question in zip(true_answers, predicted_answers, questions):
                true_answer = true_list[0] if true_list else ""
                pred_answer = pred_answer or ""
                question = question or ""

                prompt = (
                    "You are evaluating how well a generated answer responds to a given question. "
                    f"All content is in {get_language_from_code(self.lang)}. Use the real (true) answer as a reference to determine what a correct answer should look like. "
                    "Your task is to rate how well the generated answer answers the question, based on meaning and correctness, using the following scale:\n\n"
                    "5 - Fully answers the question with the same meaning as the real answer.\n"
                    "4 - Mostly answers the question with only minor differences from the real answer.\n"
                    "3 - Answers the question partially or with noticeable differences.\n"
                    "2 - Barely answers the question or includes significant inaccuracies.\n"
                    "1 - Does not answer the question or is entirely incorrect.\n\n"
                    "Return only the number.\n\n"
                    f"Question: {question.strip()}\n"
                    f"Real answer (reference): {true_answer.strip()}\n"
                    f"Generated answer: {pred_answer.strip()}\n"
                    "Score:"
                )
                prompts.append(prompt)

            scores = self.llm_judge.judge(prompts)

            # Convert string scores to floats
            numeric_scores = []

            for raw_score in scores:
                if isinstance(raw_score, (int, float)):
                    numeric_scores.append(float(raw_score))
                    continue

                if not isinstance(raw_score, str):
                    numeric_scores.append(0.0)
                    continue

                # Search for the first valid number between 1 and 5
                match = re.search(r"\b([1-5](?:\.0)?)\b", raw_score)
                if match:
                    try:
                        numeric_scores.append(float(match.group(1)))
                    except ValueError:
                        numeric_scores.append(0.0)
                else:
                    numeric_scores.append(0.0)

            store_judge(scores, numeric_scores, self.lang)

            return {
                "LLM Similarity": np.mean(numeric_scores) if numeric_scores else 0.0
            }
        else:
            smoothing_function = SmoothingFunction().method1
            bleu_scores = []
            meteor_scores = []
            cosine_similarities = []

            for true_list, pred_answer in zip(true_answers, predicted_answers):
                # Extract true answer as a single string (it's wrapped in a list)
                true_answer = true_list[0] if true_list else ""

                # Handle empty predictions
                if not pred_answer or pred_answer.strip() == "":
                    bleu_scores.append(0.0)
                    meteor_scores.append(0.0)
                    cosine_similarities.append(0.0)
                    continue  # Skip further processing

                # Normalize: Remove newlines, trim spaces, lowercase
                true_answer = true_answer.strip().lower()
                pred_answer = pred_answer.strip().lower()

                try:
                    # Tokenize for BLEU and METEOR
                    tokenized_true = word_tokenize(true_answer)  # Single-token reference
                    tokenized_pred = word_tokenize(pred_answer)

                    # BLEU Score
                    bleu = sentence_bleu([tokenized_true], tokenized_pred, smoothing_function=smoothing_function)
                    bleu_scores.append(bleu)

                    # METEOR Score (NLTK expects a list of one reference, so we wrap it)
                    meteor = meteor_score([tokenized_true], tokenized_pred)
                    meteor_scores.append(meteor)

                except Exception as e:
                    print(f"Tokenization Error: {e}")
                    bleu_scores.append(0.0)
                    meteor_scores.append(0.0)

                # Cosine Similarity (BERT embeddings)
                try:
                    true_embedding = self.embedding_model.encode([true_answer])[0].reshape(1, -1)  # Ensure correct shape
                    pred_embedding = self.embedding_model.encode([pred_answer])[0].reshape(1, -1)

                    # Compute cosine similarity
                    cosine_sim = cosine_similarity(pred_embedding, true_embedding)[0][0]
                    cosine_similarities.append(cosine_sim)

                except Exception as e:
                    print(f"Embedding Error: {e}")
                    cosine_similarities.append(0.0)

            return {
                "BLEU Score": np.mean(bleu_scores) if bleu_scores else 0.0,
                "METEOR Score": np.mean(meteor_scores) if meteor_scores else 0.0,
                "Cosine Similarity": np.mean(cosine_similarities) if cosine_similarities else 0.0
            }



    def extract_labels_from_generated_text(self, generated_texts):
        """
        Extracts the generated answers as predictions.

        :param generated_texts: List of generated answers
        :return: List of predicted answers
        """
        return generated_texts  # Directly return the generated answers as predictions

    def evaluate_results(self, results, all_true, all_predicted):
        """
        Prints BLEU, METEOR and Cosine Similarity scores for the dataset.
        """
        for lang, metrics in results.items():
            print(f"Results for {lang}:")
            if self.llm_judge:
                print(f"LLM Judge Similarity: {metrics['LLM Similarity']}")
            else:
                print(f"BLEU Score: {metrics['BLEU Score']}")
                print(f"METEOR Score: {metrics['METEOR Score']}")
                print(f"Cosine Similarity: {metrics['Cosine Similarity']}")






"""
Datasets from decoding trust paper: https://arxiv.org/pdf/2306.11698
"""
class SST2(Dataset):
    """
    SST-2 dataset from the GLUE benchmark.
    """

    def __init__(self):
        self.label_options = [0, 1]
        self.prompt = "<|endoftext|>\nTask: Label the sentiment of the text as either negative or positive. The answer should be exact 'positive' or 'negative'."

    def get_data(self, language, dataset_name, points_per_language):
        """
        Loads the SST-2 dataset from Hugging Face.

        :param language: Not needed for SST-2 (single language)
        :param dataset_name: The dataset name (GLUE benchmark)
        :param points_per_language: Number of samples to return
        :return: Processed dataset, label options, and prompt
        """
        dataset = load_dataset("glue", "sst2", split="train", trust_remote_code=True)
        data = self.extract_text(dataset, points_per_language)
        return data, self.label_options, self.prompt

    def extract_text(self, dataset, points_per_language):
        """
        Extracts text and labels.

        :param dataset: The dataset containing text data
        :return: List of dictionaries with text and labels
        """
        data = []
        for i, item in enumerate(dataset):
            if i >= points_per_language:
                break
            data.append({"text": item["sentence"], "label": item["label"]})
        return data

    def get_true_labels(self, data):
        """
        :return: List of true labels
        """
        return [entry["label"] for entry in data]

    def evaluate(self, true_labels, predicted_labels):
        """
        Evaluates the model using accuracy.

        :param true_labels: List of true labels
        :param predicted_labels: List of predicted labels
        """
        accuracy = accuracy_score(true_labels, predicted_labels)
        return {"Accuracy": accuracy}


    def extract_labels_from_generated_text(self, generated_texts):
        """
        Extracts sentiment labels (positive or negative) from generated text.

        :param generated_texts: List of generated text responses
        :return: List of extracted labels (0 for negative, 1 for positive)
        """
        all_labels = []

        for text in generated_texts:
            if text != None:
                text_lower = text.lower()

                if re.search(r"\bpositive\b", text_lower):
                    all_labels.append(1)
                elif re.search(r"\bnegative\b", text_lower):
                    all_labels.append(0)
                else:
                    all_labels.append(None)
            else:
                all_labels.append(None)
        return all_labels

    def get_true(self, data):
        """
        :return: a list of true labels for the dataset
        """
        print(data)
        true_labels = [entry['label'] for entry in data]
        return true_labels

    def evaluate_results(self, results, all_true, all_predicted):
        # Print out the results for each language
        for lang, metric in results.items():
            print(f"Results for {lang}:")
            print(f"Accuracy: {metric['Accuracy']}")
            print(f"True Labels: {all_true[lang]}, Predicted Labels: {all_predicted[lang]}")

    def get_mapped_data(self, data):
        new_data = copy.deepcopy(data)
        for entry in new_data:
            if entry["label"] == 0:
                entry["label"] = "negative"
            if entry["label"] == 1:
                entry["label"] = "positive"
        return new_data

class QQP(Dataset):
    """
    QQP dataset from the GLUE benchmark.
    """

    def __init__(self):
        self.label_options = [0, 1]  # 0: Not duplicate, 1: Duplicate
        self.prompt = "<|endoftext|>\nTask: Please identify whether Question 1 has the same meaning as Question 2. The answer should be exact 'yes' or 'no'."

    def get_data(self, language, dataset_name, points_per_language):
        """
        Loads the QQP dataset from Hugging Face.

        :param language: Not needed for QQP (single language)
        :param dataset_name: The dataset name (GLUE benchmark)
        :param points_per_language: Number of samples to return
        :return: Processed dataset, label options, and prompt
        """
        dataset = load_dataset("glue", "qqp", split="train", trust_remote_code=True)
        data = self.extract_text(dataset, points_per_language)
        return data, self.label_options, self.prompt

    def extract_text(self, dataset, points_per_language):
        """
        Extracts question pairs and labels.

        :param dataset: The dataset containing text data
        :return: List of dictionaries with question pairs and labels
        """
        data = []
        for i, item in enumerate(dataset):
            if i >= points_per_language:
                break
            data.append({
                "text": f"Question 1: {item['question1']}, Question 2: {item['question2']}",
                "label": item["label"]
            })
        return data

    def get_true_labels(self, data):
        """
        :return: List of true labels
        """
        return [entry["label"] for entry in data]

    def evaluate(self, true_labels, predicted_labels):
        """
        Evaluates the model using accuracy.

        :param true_labels: List of true labels
        :param predicted_labels: List of predicted labels
        """
        accuracy = accuracy_score(true_labels, predicted_labels)
        return {"Accuracy": accuracy}

    def extract_labels_from_generated_text(self, generated_texts):
        """
        Extracts duplicate/not duplicate labels from generated text.

        :param generated_texts: List of generated text responses
        :return: List of extracted labels (0 for not duplicate, 1 for duplicate)
        """
        all_labels = []

        for text in generated_texts:
            if text is not None:
                text_lower = text.lower()

                if re.search(r"\byes\b", text_lower):
                    all_labels.append(1)
                elif re.search(r"\bno\b", text_lower):
                    all_labels.append(0)
                else:
                    all_labels.append(None)
            else:
                all_labels.append(None)
        return all_labels

    def get_true(self, data):
        """
        :return: A list of true labels for the dataset
        """
        return [entry['label'] for entry in data]

    def evaluate_results(self, results, all_true, all_predicted):
        """
        Prints accuracy and results for each language (even though QQP is monolingual).
        """
        for lang, metric in results.items():
            print(f"Results for {lang}:")
            print(f"Accuracy: {metric['Accuracy']}")
            print(f"True Labels: {all_true[lang]}, Predicted Labels: {all_predicted[lang]}")

    def get_mapped_data(self, data):
        new_data = copy.deepcopy(data)
        for entry in new_data:
            if entry["label"] == 0:
                entry["label"] = "no"
            if entry["label"] == 1:
                entry["label"] = "yes"
        return new_data

class MNLI(Dataset):
    """
    MNLI dataset from the GLUE benchmark.
    """

    def __init__(self):
        self.label_options = [0, 1, 2]  # 0: Contradiction, 1: Neutral, 2: Entailment
        self.prompt = "<|endoftext|>\nTask: Please identify whether the premise entails or contradicts the hypothesis, or neither. The answer should be exactly 'entailment', 'neutral', or 'contradiction'."

    def get_data(self, language, dataset_name, points_per_language):
        """
        Loads the MNLI dataset from Hugging Face.

        :param language: Not needed for MNLI (single language)
        :param dataset_name: The dataset name (GLUE benchmark)
        :param points_per_language: Number of samples to return
        :return: Processed dataset, label options, and prompt
        """
        dataset = load_dataset("glue", "mnli", split="train", trust_remote_code=True)
        data = self.extract_text(dataset, points_per_language)
        return data, self.label_options, self.prompt

    def extract_text(self, dataset, points_per_language):
        """
        Extracts premise-hypothesis pairs and labels.

        :param dataset: The dataset containing text data
        :return: List of dictionaries with premise, hypothesis, and labels
        """
        data = []
        for i, item in enumerate(dataset):
            if i >= points_per_language:
                break
            data.append({
                "text": f"Premise: {item['premise']}, Hypothesis: {item['hypothesis']}",
                "label": item["label"]
            })
        return data

    def get_true_labels(self, data):
        """
        :return: List of true labels
        """
        return [entry["label"] for entry in data]

    def evaluate(self, true_labels, predicted_labels):
        """
        Evaluates the model using accuracy.

        :param true_labels: List of true labels
        :param predicted_labels: List of predicted labels
        """
        accuracy = accuracy_score(true_labels, predicted_labels)
        return {"Accuracy": accuracy}

    def extract_labels_from_generated_text(self, generated_texts):
        """
        Extracts contradiction/neutral/entailment labels from generated text.

        :param generated_texts: List of generated text responses
        :return: List of extracted labels (0 for contradiction, 1 for neutral, 2 for entailment)
        """
        all_labels = []

        for text in generated_texts:
            if text is not None:
                text_lower = text.lower()

                if re.search(r"\bentailment\b", text_lower):
                    all_labels.append(0)
                elif re.search(r"\bneutral\b", text_lower):
                    all_labels.append(1)
                elif re.search(r"\bcontradiction\b", text_lower):
                    all_labels.append(2)
                else:
                    all_labels.append(None)
            else:
                all_labels.append(None)
        return all_labels

    def get_true(self, data):
        """
        :return: A list of true labels for the dataset
        """
        return [entry['label'] for entry in data]

    def evaluate_results(self, results, all_true, all_predicted):
        """
        Prints accuracy and results for each language (even though MNLI is monolingual).
        """
        for lang, metric in results.items():
            print(f"Results for {lang}:")
            print(f"Accuracy: {metric['Accuracy']}")
            print(f"True Labels: {all_true[lang]}, Predicted Labels: {all_predicted[lang]}")

    def get_mapped_data(self, data):
        new_data = copy.deepcopy(data)
        for entry in new_data:
            if entry["label"] == 0:
                entry["label"] = "entailment"
            if entry["label"] == 1:
                entry["label"] = "neutral"
            if entry["label"] == 2:
                entry["label"] = "contradiction"
        return new_data

class QNLI(Dataset):
    """
    QNLI dataset from the GLUE benchmark.
    """

    def __init__(self):
        self.label_options = [0, 1]  # 0: Entailment, 1: Not Entailment
        self.prompt = "<|endoftext|>\nTask: Determine whether the sentence answers the question. The answer should be exactly 'yes' or 'no'."

    def get_data(self, language, dataset_name, points_per_language):
        """
        Loads the QNLI dataset from Hugging Face.

        :param language: Not needed for QNLI (single language)
        :param dataset_name: The dataset name (GLUE benchmark)
        :param points_per_language: Number of samples to return
        :return: Processed dataset, label options, and prompt
        """
        dataset = load_dataset("glue", "qnli", split="train", trust_remote_code=True)
        data = self.extract_text(dataset, points_per_language)
        return data, self.label_options, self.prompt

    def extract_text(self, dataset, points_per_language):
        """
        Extracts question-passage pairs and labels.

        :param dataset: The dataset containing text data
        :return: List of dictionaries with question, passage, and labels
        """
        data = []
        for i, item in enumerate(dataset):
            if i >= points_per_language:
                break
            data.append({
                "text": f"Question: {item['question']}, Sentence: {item['sentence']}",
                "label": item["label"]
            })
        return data

    def get_true_labels(self, data):
        """
        :return: List of true labels
        """
        return [entry["label"] for entry in data]

    def evaluate(self, true_labels, predicted_labels):
        """
        Evaluates the model using accuracy.

        :param true_labels: List of true labels
        :param predicted_labels: List of predicted labels
        """
        accuracy = accuracy_score(true_labels, predicted_labels)
        return {"Accuracy": accuracy}

    def extract_labels_from_generated_text(self, generated_texts):
        """
        Extracts entailment/not entailment labels from generated text.

        :param generated_texts: List of generated text responses
        :return: List of extracted labels (0 for entailment, 1 for not entailment)
        """
        all_labels = []

        for text in generated_texts:
            if text is not None:
                text_lower = text.lower()

                if re.search(r"\byes\b", text_lower):
                    all_labels.append(0)
                elif re.search(r"\bno\b", text_lower):
                    all_labels.append(1)
                else:
                    all_labels.append(None)  # Can't determine
            else:
                all_labels.append(None)

        return all_labels

    def get_true(self, data):
        """
        :return: A list of true labels for the dataset
        """
        return [entry['label'] for entry in data]

    def evaluate_results(self, results, all_true, all_predicted):
        """
        Prints accuracy and results for each language (even though QNLI is monolingual).
        """
        for lang, metric in results.items():
            print(f"Results for {lang}:")
            print(f"Accuracy: {metric['Accuracy']}")
            print(f"True Labels: {all_true[lang]}, Predicted Labels: {all_predicted[lang]}")

    def get_mapped_data(self, data):
        new_data = copy.deepcopy(data)
        for entry in new_data:
            if entry["label"] == 0:
                entry["label"] = "yes"
            if entry["label"] == 1:
                entry["label"] = "no"
        return new_data



