import os
import csv
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import re
from deep_translator import GoogleTranslator


class Dataset:
    """
    Base Dataset class with a factory method to return the appropriate dataset object.
    """

    def get_data(self, language):
        """
        Abstract method to get data in a specific language.
        This should be implemented by child classes.
        """
        raise NotImplementedError("Child class must implement this method")

    def get_true_labels(self):
        """
        Abstract method to get the true labels for the dataset.
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
        elif name.lower() == 'go_emotions':
            return Go_Emotions()
        elif name.lower() == 'casehold':
            return CaseHOLD()
        elif name.lower() == 'xnli':
            return XNLI()
        else:
            raise ValueError(f"Dataset '{name}' is not available")

    def extract_labels_from_generated_text(self, generated_text, label_options):
        relevant_labels = []
        for label in label_options:
            if label.lower() in generated_text.lower():
                relevant_labels.append(label)
        return relevant_labels


class Multi_Eurlex(Dataset):
    """
    Child class of Dataset representing the Multi-EUR-Lex dataset.
    """

    def __init__(self):
        self.languages = ['en', 'da', 'de', 'nl', 'sv', 'es', 'fr', 'it', 'pt', 'ro', 'bg', 'cs', 'hr', 'pl', 'sl',
                          'et', 'fi', 'hu', 'lt', 'lv', 'el']
        self.label_options = [
            "POLITICS", "INTERNATIONAL RELATIONS", "EUROPEAN UNION", "LAW", "ECONOMICS",
            "TRADE", "FINANCE", "SOCIAL QUESTIONS", "EDUCATION AND COMMUNICATIONS", "SCIENCE",
            "BUSINESS AND COMPETITION", "EMPLOYMENT AND WORKING CONDITIONS", "TRANSPORT",
            "ENVIRONMENT", "AGRICULTURE, FORESTRY AND FISHERIES", "AGRI-FOODSTUFFS",
            "PRODUCTION, TECHNOLOGY AND RESEARCH", "ENERGY", "INDUSTRY", "GEOGRAPHY",
            "INTERNATIONAL ORGANISATIONS"
        ]
        self.prompt = "<|endoftext|>" + (
                "Question: Which of the following labels apply? (You can select more than one): "
                + ', '.join(self.label_options) + " "
                                                  "Answer:"
        )

    def get_data(self, language):
        """
        :param language: the language for which data should be retrieved
        :return: the data corresponding to the language parameter
        """
        dataset = load_dataset('multi_eurlex', language, split='test', trust_remote_code=True)
        self.language = language
        if language == 'all_languages':
            data = self.extract_text_all_languages(dataset)
        else:
            data = self.extract_text(dataset)
        return data, self.label_options

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

    def extract_text(self, dataset):
        """
        :param dataset: the dataset containing the text data
        :return: a list of text data in the specified language
        """
        data = []
        count = 0
        for item in dataset:
            if count == 100:
                break
            data.append({"text": item['text'], "labels": item['labels']})
            count += 1
        return data

    def get_true_labels(self, data):
        """
        :return: a list of true labels for the dataset
        """
        true_labels = [entry['labels'] for entry in data]
        return true_labels

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

        # Print the results
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        file_path = "MultiEurlex_evaluation.csv"
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Language", "Precision", "Recall", "F1 Score"])
            writer.writerow([self.language, precision, recall, f1])


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
        return self.extract_text(dataset), self.label_options

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
        return self.extract_text(dataset), self.label_options

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

    def extract_labels_from_generated_text(self, generated_text):
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
        self.prompt = ("<|endoftext|>"
                       "Decide if the hypothesis logically follows from the premise (entailment), is "
                       "contradictory (contradiction),"
                       "or is neutral with respect to the hypothesis (neutral). "
                       "Answer only with one of the following options: 0 for entailment, 1 for neutral, "
                       "or 2 contradiction. Give no further explanation."
                       )

    def get_data(self, language):
        """
        Loads the XNLI dataset for the specified language.
        :param language: the language of the dataset
        :return: the data and label options
        """
        dataset = load_dataset('xnli', language, split='test', trust_remote_code=True)
        print(dataset)
        self.language = language
        if language == 'all_languages':
            data = self.extract_text_all_languages(dataset)
        else:
            data = self.extract_text(dataset)
        return data, self.label_options

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

    def extract_text(self, dataset):
        """
        :param dataset: the dataset containing the text data
        :return: a list of text data in the specified language
        """
        data = []
        count = 0
        for item in dataset:
            if count == 300:
                break
            text = "Premise: " + item["premise"] + " Hypothesis: " + item["hypothesis"]
            data.append({"text": text, "label": item['label']})
            count += 1
        print(f"Data extracted: {data}")
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
        with open(file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Language", "Accuracy"])
            writer.writerow([self.language, accuracy])

        print(f"Accuracy: {accuracy}")

    def extract_labels_from_generated_text(self, generated_text):
        """
        Extracts the first predicted label from the model's response.
        :param response: The model's output as a string
        :return: The first valid label (Entailment, Contradiction, Neutral) found in the response, or None if not found
        """
        print(f"Reached extract_labels in XNLI class for generated text {generated_text}")
        for i, label in enumerate(self.label_options):
            if label in generated_text.lower():
                return i
        return -1

    def get_true_labels(self, data):
        """
        :return: list of true labels for the dataset
        """
        return [entry['label'] for entry in data]

