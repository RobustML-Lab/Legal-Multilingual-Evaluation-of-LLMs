from transformers import BartTokenizer, BartForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, \
    LlamaTokenizer, LlamaForCausalLM
from deep_translator import GoogleTranslator
import ollama

import google.generativeai as ggai
import re
import time


class Model:
    """
    Base Model class with a.py factory method to return the appropriate model object.
    """

    def predict(self, dataset: list, prompt: str):
        """
        Predict labels for a.py dataset.

        :param dataset: A list of text samples.
        :param prompt: The prompt for label prediction.
        :return: A list of lists, where each inner list contains the predicted label indices for each text sample.
        """
        return [self.classify_text(item['text'], prompt=prompt) for item in dataset], None

    @staticmethod
    def get_model(name, label_options, multi_class=False, api_key = None, generation = False):
        """
        :param name: the name of the model
        :return: the model object
        """
        if name.lower() == 'llama':
            return LLaMa(label_options, multi_class, generation)
        elif name.lower() == 'google':
            return Google(label_options, multi_class, api_key, generation)
        elif name.lower() == 'ollama':
            return OLLaMa(label_options, multi_class, generation)
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


class Bart(Model):
    """
    The BART model
    """

    def __init__(self, label_options, multi_class=False):
        self.label_options = label_options
        self.multi_class = multi_class
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")


class LLaMa(Model):
    """
    The LLaMA model
    """

    def __init__(self, label_options, multi_class=False, generation = False):
        self.label_options = label_options
        self.multi_class = multi_class
        # model_dir = "meta-llama/Meta-Llama-
        # 3.1-8B-Instruct"
        model_dir = "meta-llama/Llama-3.2-1B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.generation = generation
        # self.tokenizer = LlamaTokenizer.from_pretrained(model_dir)
        # self.model = LlamaForCausalLM.from_pretrained(model_dir)

    def generate_text(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_length=800, num_return_sequences=1)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def classify_text(self, text, prompt):
        """
        :param text: the text that needs to be classified
        :return: a.py list of all the labels corresponding to the given text
        """
        try:
            quoted_labels = "', '".join(f"{i}: {label}" for i, label in enumerate(self.label_options))
        except Exception as e:
            quoted_labels = ""
        complete_prompt = f"{text}{prompt}'{quoted_labels}'."
        generated_text = self.generate_text(complete_prompt)
        prediction = self.dataset.extract_labels_from_generated_text(generated_text, self.label_options)
        predicted_labels_indexed = self.map_labels_to_indices(prediction, self.label_options)
        return predicted_labels_indexed

class OLLaMa(Model):
    """
    Using the OLLaMa models
    """
    def __init__(self, label_options, multi_class=False, generation=False):
        self.label_options = label_options
        self.multi_class = multi_class
        self.generation = generation

    def generate_text(self, prompt):
        generated_stream = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        response = ""
        for chunk in generated_stream:
            response += chunk["message"]["content"]
        return response

    # def generate_text(self, prompt):
    #     # API endpoint for the Ollama model
    #     url = "http://localhost:11434/api/generate"
    #
    #     # Setting up the payload for the request
    #     data = {
    #         "model": "llama3.2",
    #         "messages": [{"role": "user", "content": prompt}]
    #     }
    #
    #     # Create an HTTP client with proxy configuration
    #     proxy_address = "http://81.171.3.101:3128"
    #     proxies = {
    #         "http://": proxy_address,
    #         "https://": proxy_address
    #     }
    #
    #     # Make the request to Ollama through the proxy
    #     response_text = ""
    #     try:
    #         with httpx.Client(proxies=proxies) as client:
    #             response = client.post(url, json=data)
    #             response.raise_for_status()  # Raise an exception if the request failed
    #             response_data = response.json()
    #
    #             # Extract the generated text from the response
    #             for message in response_data.get("choices", []):
    #                 response_text += message["message"]["content"]
    #
    #     except httpx.RequestError as e:
    #         print(f"An error occurred while making the request: {e}")
    #     except httpx.HTTPStatusError as e:
    #         print(f"Request returned an unsuccessful status code: {e}")
    #
    #     print("Response: ", response_text)
    #     return response_text

    def classify_text(self, text, prompt, language='en'):
        """
        :param text: the text that needs to be classified
        :return: a list of all the labels corresponding to the given text
        """
        print("Reached classify_text")
        translator = GoogleTranslator(source="en", target=language)
        print("Type of the prompt: ", type(prompt))
        translated_prompt = translator.translate(prompt)
        complete_prompt = text + translated_prompt
        generated_text = self.generate_text(complete_prompt)
        with open("responses.txt", "a", encoding="utf-8") as file:
            file.write(generated_text+"\n###################################################\n")
        if self.generation:
            prediction = generated_text
        else:
            prediction = self.extract_labels_from_generated_text(generated_text, self.label_options)
        return prediction

class Google(Model):
    """
    The Google model
    """

    def __init__(self, label_options, multi_class=False, api_key=None, generation = False):
        self.label_options = label_options
        self.generation = generation
        self.multi_class = multi_class
        ggai.configure(api_key=api_key)
        self.model = ggai.GenerativeModel('gemini-1.5-flash')

    def generate_text(self, prompt):
        # Generate the text using the model
        response = self.model.generate_content(prompt)

        if not response:
            print("Error: No response from the API.")
            return ""

        return response.text

    def predict(self, dataset: list, prompt: str):
        all_predicted = []
        first_ten_answers = []
        count = 0  # Track the number of requests
        false_count = 0
        count_ten = 0

        for index, entry in enumerate(dataset):
            text = entry['text']
            if self.generation:
                complete_prompt = f"{text}{prompt}"
            else:
                quoted_labels = "', '".join(f"{i}: {label}" for i, label in enumerate(self.label_options))
                complete_prompt = f"{text}{prompt}'{quoted_labels}'."
            if false_count > 20:
                print(f"More than 20 errors.\n{index}\n{prompt}")
                for i in range(index, len(dataset)):
                    all_predicted.append(None)
                break
            try:
                # Rate limiting: Ensure no more than 15 requests per minute
                if count >= 15:
                    print("Reached request limit (15 per minute). Sleeping for 60 seconds...")
                    time.sleep(60)  # Sleep for 60 seconds to comply with rate limits
                    count = 0  # Reset the request count after sleeping

                # Get Gemini's generated labels
                generated_text = self.generate_text(complete_prompt)

                # Store true and predicted labels for comparison
                all_predicted.append(generated_text)

                if count_ten < 10:
                    first_ten_answers.append(generated_text)
                    count_ten += 1

                # Update request count
                count += 1
                false_count = 0

            except Exception as e:
                # Handle any request-related exceptions, like rate-limiting or network errors
                print(f"Error occurred: {e}. Retrying after 60 seconds...")
                time.sleep(60)  # Sleep for 60 seconds before retrying
                count = 0  # Reset the request count
                false_count += 1
                all_predicted.append(None)

        return all_predicted, first_ten_answers


