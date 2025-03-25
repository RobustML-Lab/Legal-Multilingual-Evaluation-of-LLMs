import google.generativeai as ggai
import time


class JudgeEvaluator:
    def __init__(self, api_key: str):
        """
        Initializes the evaluator with an API.
        :param api_key: API key for authentication.
        """
        ggai.configure(api_key=api_key)
        self.model = ggai.GenerativeModel('gemini-2.0-flash')

    def generate_text(self, prompt):
        # Generate the text using the model
        response = self.model.generate_content(prompt)

        if not response:
            print("Error: No response from the API.")
            return ""

        return response.text

    def judge(self, prompts):
        all_predicted = []
        count = 0  # Track requests per minute
        false_count = 0
        index = 0

        while index < len(prompts):
            prompt = prompts[index]

            if false_count > 20:
                print(f"More than 20 errors.\n{index}\n{prompt}")
                all_predicted.extend([None] * (len(prompts) - index))
                break

            try:
                # Rate limiting: 15 requests per minute
                if count >= 15:
                    print("Reached 15 requests/minute. Sleeping for 60 seconds...")
                    time.sleep(60)
                    count = 0

                # Get Gemini's response
                generated_text = self.generate_text(prompt)

                all_predicted.append(generated_text)

                count += 1
                false_count = 0
                index += 1

            except Exception as e:
                print(f"Error occurred: {e}. Retrying prompt after 60 seconds...")
                time.sleep(60)
                count = 0
                false_count += 1

        return all_predicted
