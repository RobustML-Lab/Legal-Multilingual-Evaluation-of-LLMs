from transformers import BartTokenizer, BartForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, \
    LlamaTokenizer, LlamaForCausalLM


class Model:
    """
    Base Model class with a factory method to return the appropriate model object.
    """

    def classify_text(self, text):
        """
        Abstract method to classify text.
        This should be implemented by child classes.
        """
        raise NotImplementedError("Child class must implement this method")

    @staticmethod
    def get_model(name, label_options, multi_class=False):
        """
        :param name: the name of the model
        :return: the model object
        """
        if name.lower() == 'bart':
            return Bart(label_options, multi_class)
        elif name.lower() == 'llama':
            return LLaMa(label_options, multi_class)
        else:
            raise ValueError(f"Model '{name}' is not available")

    def map_labels_to_indices(self, label_names, label_options):
        """
        :param label_names: the names of the labels predicted
        :param label_options: a list of all the labels
        :return: the indices of the predicted labels
        """
        label_indices = [label_options.index(label) for label in label_names if label in label_options]
        return label_indices

    def extract_labels_from_generated_text(self, generated_text, label_options):
        relevant_labels = []
        for label in label_options:
            if label.lower() in generated_text.lower():
                relevant_labels.append(label)
        return relevant_labels

    def predict(self, dataset):
        pass


class Bart(Model):
    """
    The BART model
    """

    def __init__(self, label_options, multi_class=False):
        self.label_options = label_options
        self.multi_class = multi_class
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

    def extract_labels_from_generated_text(self, generated_text, label_options):
        return super().extract_labels_from_generated_text(generated_text, label_options)

    def map_labels_to_indices(self, label_names, label_options):
        return super().map_labels_to_indices(label_names, label_options)

    def generate_text(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_length=100, num_return_sequences=1)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def classify_text(self, text):
        """
        :param text: the text that needs to be classified
        :return: a list of all the labels corresponding to the given text
        """
        prompt = text + "<|endoftext|>" + ("Question: Which of the following labels apply?{', '.join(label_options)} "
                                           "Answer:")
        generated_text = self.generate_text(prompt)
        prediction = self.extract_labels_from_generated_text(generated_text, self.label_options)
        predicted_labels_indexed = self.map_labels_to_indices(prediction, self.label_options)
        return predicted_labels_indexed

    def predict(self, dataset):
        """
        :param dataset: a list of text samples
        :return: a list of lists, where each inner list contains the predicted labels for the corresponding text sample
        """
        predicted_labels = []
        for item in dataset:
            text = item['text']
            predicted_labels.append(self.classify_text(text))
        return predicted_labels

class LLaMa(Model):
    """
    The LLaMA model
    """

    def __init__(self, label_options, multi_class=False):
        self.label_options = label_options
        self.multi_class = multi_class
        model_dir = "huggyllama/llama-7b"
        self.tokenizer = LlamaTokenizer.from_pretrained(model_dir)
        self.model = LlamaForCausalLM.from_pretrained(model_dir)

    def extract_labels_from_generated_text(self, generated_text, label_options):
        return super().extract_labels_from_generated_text(generated_text, label_options)

    def map_labels_to_indices(self, label_names, label_options):
        return super().map_labels_to_indices(label_names, label_options)

    def generate_text(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_length=1000, num_return_sequences=1)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def classify_text(self, text):
        """
        :param text: the text that needs to be classified
        :return: a list of all the labels corresponding to the given text
        """
        prompt = text + "<|endoftext|>" + ("Question: Which of the following labels apply?{', '.join(label_options)} "
                                           "Answer:")
        generated_text = self.generate_text(prompt)
        prediction = self.extract_labels_from_generated_text(generated_text, self.label_options)
        predicted_labels_indexed = self.map_labels_to_indices(prediction, self.label_options)
        return predicted_labels_indexed

    def predict(self, dataset):
        """
        :param dataset: a list of text samples
        :return: a list of lists, where each inner list contains the predicted labels for the corresponding text sample
        """
        predicted_labels = []
        for item in dataset:
            text = item['text']
            predicted_labels.append(self.classify_text(text))
        return predicted_labels