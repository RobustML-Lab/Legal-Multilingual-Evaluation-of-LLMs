from judges.classifiers.correctness import PollZeroShotCorrectness
from judges.graders.response_quality import MTBenchChatBotResponseQuality

class JudgeEvaluator:
    def __init__(self, api_key: str, model: str = "ollama/llama2"):
        """
        Initializes the evaluator with an API key and model.
        :param api_key: API key for authentication.
        :param model: Model name for evaluation.
        """
        # Set up authentication (if required by the library)
        self.api_key = api_key
        self.model = model

        # Initialize judges
        self.correctness_judge = PollZeroShotCorrectness(model=self.model)
        self.quality_grader = MTBenchChatBotResponseQuality(model=self.model)

    def evaluate_true_false(self, input_text: str, output_text: str, truth_value: str) -> bool:
        """
        Evaluates if the given output is factually correct compared to the truth value.
        :param input_text: The original prompt/question.
        :param output_text: The AI-generated response.
        :param truth_value: The correct answer that the AI output should match.
        :return: True if the AI response matches the truth value, otherwise False.
        """
        # Judge if AI's response is correct
        ai_correct = self.correctness_judge.judge(input=input_text, output=output_text, expected=truth_value)

        # Compare with the expected truth value
        return ai_correct and (output_text.strip().lower() == truth_value.strip().lower())

    def evaluate_score(self, input_text: str, output_text: str, truth_value: str) -> float:
        """
        Evaluates the quality of the response and returns a score.
        :param input_text: The original prompt/question.
        :param output_text: The AI-generated response.
        :param truth_value: The correct answer for comparison.
        :return: A numerical score for the response quality.
        """
        return self.quality_grader.judge(input=input_text, output=output_text, expected=truth_value)


if __name__ == "__main__":
    api_key = "your_api_key_here"  # Replace with your actual API key
    evaluator = JudgeEvaluator(api_key)

    input_text = "What is the capital of France?"
    output_text = "The capital of France is Paris."
    truth_value = "Paris"

    # True/False Evaluation
    # result = evaluator.evaluate_true_false(input_text, output_text, truth_value)
    # print(f"True/False Evaluation: {'Correct' if result else 'Incorrect'}")

    # Score Evaluation
    score = evaluator.evaluate_score(input_text, output_text, truth_value)
    print(f"Score Evaluation: {score}")