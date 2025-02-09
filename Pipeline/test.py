import torch
import transformers
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline_instance = None

# Function from your code
def get_pipeline(**kwargs):
    """Initializes Huggingface pipeline for a local model if not already initialized."""
    defaults = {
        "model_id": "meta-llama/Llama-3.2-1B",
        "torch_dtype": torch.float16,
        "device_map": "auto"
    }
    params = {**defaults, **kwargs}

    logger.info(f"Initializing pipeline with params: {params}")
    global pipeline_instance
    if pipeline_instance is None:
        pipeline_instance = transformers.pipeline(
            "text-generation",
            model=params["model_id"],
            device_map=params["device_map"],
            torch_dtype=params["torch_dtype"]
        )
        pipeline_instance.tokenizer.pad_token_id = pipeline_instance.tokenizer.eos_token_id

        # Compile the model if supported for slight increase in inference speed
        try:
            pipeline_instance.model = torch.compile(pipeline_instance.model, mode='max-autotune')
            logger.info("Model successfully compiled with torch.compile.")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Falling back to uncompiled model.")

    return pipeline_instance

# Example usage
if __name__ == "__main__":
    pipeline = get_pipeline()

    question = "What is the capital of France?"
    response = pipeline(question, max_length=50, num_return_sequences=1)

    print(response[0]['generated_text'])
