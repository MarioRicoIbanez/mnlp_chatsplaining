import json
import os
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple

from datasets import Dataset, DatasetDict

# Handle relative import when running directly vs as module
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from data.base_openans_processor import BaseOpenQAProcessor
else:
    from .base_openans_processor import BaseOpenQAProcessor

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAnswerProcessor(BaseOpenQAProcessor):
    """Processor for the m1_preference_data.json open-answer dataset."""

    def __init__(self, json_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.json_path = json_path

    def process_dataset(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        logger.info("Loading JSON file...")
        with open(self.json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        logger.info("Filtering open_answer questions...")
        data = [
            {
                "question": item["question_body"].strip(),
                "answer": item["question_answer"].strip(),
                "source": "m1_preference_data"
            }
            for item in raw_data
            if item.get("question_type") == "open_answer"
        ]

        logger.info(f"Found {len(data)} open-answer examples")
        random.shuffle(data)

        n = len(data)
        train_data = data[:int(0.9 * n)]
        val_data = data[int(0.9 * n):int(0.95 * n)]
        test_data = data[int(0.95 * n):]

        return train_data, val_data, test_data

    def push_to_hub(self, repo_name: str, env_token_key: str = "HF_TOKEN"):
        load_dotenv()
        token = os.getenv(env_token_key)
        if not token:
            raise ValueError(f"Hugging Face token not found in environment variable '{env_token_key}'")

        logger.info("Processing dataset...")
        train_data, val_data, test_data = self.process_dataset()

        logger.info("Validating format...")
        for split_name, split_data in zip(["train", "validation", "test"], [train_data, val_data, test_data]):
            for item in split_data:
                if not all(k in item for k in ["question", "answer", "source"]):
                    raise ValueError(f"Missing required fields in {split_name} data: {item}")

        logger.info("Creating Hugging Face DatasetDict...")
        dataset_dict = DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data),
            "test": Dataset.from_list(test_data)
        })

        logger.info(f"Pushing to Hugging Face Hub at: {repo_name}...")
        dataset_dict.push_to_hub(
            repo_name,
            token=token,
            commit_message="Upload open-answer STEM dataset from m1_preference_data"
        )
        logger.info("✅ Successfully pushed to Hugging Face Hub")


def main():
    processor = OpenAnswerProcessor(json_path="data/m1_preference_data.json")
    processor.push_to_hub(repo_name="jonlecumberri/stem-open-answer-preferences")


if __name__ == "__main__":
    main()
