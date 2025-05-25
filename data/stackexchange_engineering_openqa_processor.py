import os
import logging
from typing import List, Dict, Tuple

from datasets import load_dataset, Dataset, DatasetDict
from dotenv import load_dotenv

# Handle relative import when running directly vs as module
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from data.base_openans_processor import BaseOpenQAProcessor
else:
    from .base_openans_processor import BaseOpenQAProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StackExchangeEngineeringOpenAnswerProcessor(BaseOpenQAProcessor):
    """Processor for open-answer questions from the StackExchange Engineering dataset."""

    def process_dataset(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        logger.info("Loading StackExchange Engineering dataset...")
        try:
            dataset = load_dataset("mlfoundations-dev/stackexchange_engineering")
            logger.info("Dataset loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        raw_data = dataset["train"]
        logger.info(f"Processing {len(raw_data)} examples...")

        openqa_data = []

        for item in raw_data:
            question = item.get("instruction", "")
            answer = item.get("completion", "")

            # if not question or not answer:
            #     continue

            openqa_data.append({
                "question": question,
                "answer": answer,
                "source": "stackexchange_engineering",
                "explanation": ""  # No explanation field provided
            })

        logger.info(f"Collected {len(openqa_data)} open-answer examples")

        # Shuffle and split
        n = len(openqa_data)
        train_data = openqa_data[:int(0.9 * n)]
        val_data = openqa_data[int(0.9 * n):int(0.95 * n)]
        test_data = openqa_data[int(0.95 * n):]

        return train_data, val_data, test_data

    def push_to_hub(self, repo_name: str, env_token_key: str = "HF_TOKEN"):
        load_dotenv()
        token = os.getenv(env_token_key)
        if not token:
            raise ValueError(f"Hugging Face token not found in environment variable '{env_token_key}'")

        logger.info("Processing and validating OpenQA dataset...")
        train_data, val_data, test_data = self.process_dataset()

        if not self.validate_openqa_format(train_data):
            raise ValueError("Training data does not follow OpenQA format")
        if not self.validate_openqa_format(val_data):
            raise ValueError("Validation data does not follow OpenQA format")
        if not self.validate_openqa_format(test_data):
            raise ValueError("Test data does not follow OpenQA format")

        dataset_dict = self.create_dataset_dict(train_data, val_data, test_data)

        logger.info(f"Pushing to Hugging Face Hub at: {repo_name}...")
        dataset_dict.push_to_hub(
            repo_name,
            token=token,
            commit_message="Upload OpenQA version of StackExchange Engineering dataset"
        )
        logger.info("✅ Successfully pushed OpenQA dataset!")


def main():
    processor = StackExchangeEngineeringOpenAnswerProcessor()
    processor.process_and_save()
    processor.push_to_hub("jonlecumberri/stackexchange_engineering_openqa")


if __name__ == "__main__":
    main()
