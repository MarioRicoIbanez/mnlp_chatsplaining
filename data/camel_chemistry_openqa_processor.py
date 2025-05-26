import os
import random
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


class OpenAnswerProcessor(BaseOpenQAProcessor):
    """Processor for the CAMEL Chemistry dataset in open-answer format."""

    def process_dataset(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        logger.info("Loading CAMEL Chemistry dataset...")
        try:
            dataset = load_dataset("mlfoundations-dev/camel_chemistry_seed_science_20K")
            logger.info("Dataset loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        raw_data = dataset["train"]
        logger.info(f"Processing {len(raw_data)} examples...")

        data = []
        for item in raw_data:
            question = item.get("problem", "")
            solution = item.get("deepseek_solution", "")
            explanation = item.get("reasoning", "")

            # if not question or not isinstance(solutions, list) or not solutions:
            #     continue

            # answer = solutions[0].strip()
            # if not answer:
            #     continue

            data.append(
                {
                    "question": question,
                    "answer": solution,
                    "explanation": explanation,
                    "source": "camel_chemistry_seed_science_20K",
                }
            )

        logger.info(f"Collected {len(data)} open-answer examples")
        random.shuffle(data)

        n = len(data)
        train_data = data[: int(0.9 * n)]
        val_data = data[int(0.9 * n) : int(0.95 * n)]
        test_data = data[int(0.95 * n) :]

        return train_data, val_data, test_data

    def push_to_hub(self, repo_name: str, env_token_key: str = "HF_TOKEN"):
        load_dotenv()
        token = os.getenv(env_token_key)
        if not token:
            raise ValueError(
                f"Hugging Face token not found in environment variable '{env_token_key}'"
            )

        logger.info("Processing dataset...")
        train_data, val_data, test_data = self.process_dataset()

        logger.info("Validating format...")
        for split_name, split_data in zip(
            ["train", "validation", "test"], [train_data, val_data, test_data]
        ):
            for item in split_data:
                if not all(k in item for k in ["question", "answer", "source"]):
                    raise ValueError(
                        f"Missing required fields in {split_name} data: {item}"
                    )

        logger.info("Creating Hugging Face DatasetDict...")
        dataset_dict = DatasetDict(
            {
                "train": Dataset.from_list(train_data),
                "validation": Dataset.from_list(val_data),
                "test": Dataset.from_list(test_data),
            }
        )

        logger.info(f"Pushing to Hugging Face Hub at: {repo_name}...")
        dataset_dict.push_to_hub(
            repo_name,
            token=token,
            commit_message="Upload CAMEL Chemistry dataset in OpenQA format",
        )
        logger.info("✅ Successfully pushed to Hugging Face Hub!")


def main():
    processor = OpenAnswerProcessor()
    processor.push_to_hub(repo_name="jonlecumberri/camel_chemistry_openqa")


if __name__ == "__main__":
    main()
