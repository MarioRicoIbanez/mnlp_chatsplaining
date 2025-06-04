from datasets import load_dataset, Dataset, DatasetDict
from typing import List, Dict, Tuple
import os
import sys
from pathlib import Path

# Add parent directory to path when running as script
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent))
    from data.base_processor import BaseDatasetProcessor
else:
    from .base_processor import BaseDatasetProcessor

import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedMCQAProcessor(BaseDatasetProcessor):
    """Processor for the pszemraj/unified-mcqa-all dataset."""

    def __init__(self):
        super().__init__()

    def process_dataset(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        logger.info("Loading Unified MCQA dataset...")
        try:
            dataset = load_dataset("pszemraj/unified-mcqa-all")
            logger.info("Dataset loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        train_data = self._process_split(dataset.get("train", []), split_name="train")
        return train_data, [], []  # No val/test splits in this dataset

    def _process_split(self, split, split_name: str) -> List[Dict]:
        logger.info(f"Processing {split_name} split...")
        processed = []
        for i, item in enumerate(split):
            if i % 100 == 0:
                logger.info(f"Processed {i} {split_name} examples")
            processed_item = self._process_item(item)
            if processed_item:
                processed.append(processed_item)
        return processed

    def _process_item(self, item: Dict) -> Dict:
        """
        Process a single MCQA item into a unified format.
        """
        try:
            question = item["question"]
            choices = item["choices"]
            label_index = item["label"]

            if not isinstance(choices, list) or len(choices) != 4:
                logger.error(f"Choices must be a list of exactly 4 items: {choices}")
                return None

            return {
                "question": question,
                "choices": choices,
                "answer_index": label_index,
                "answer_text": choices[label_index],
                "source": item.get("source_dataset", "unified-mcqa"),
                "explanation": item.get(
                    "context", ""
                ),  # Optional: context as explanation
            }

        except Exception as e:
            logger.warning(f"Error processing item: {e}\nRaw item: {item}")
            return None

    def push_to_hub(self, repo_name: str, env_token_key: str = "HF_TOKEN"):
        load_dotenv()
        token = os.getenv(env_token_key)
        if not token:
            raise ValueError(
                f"Hugging Face token not found in environment variable '{env_token_key}'"
            )

        train_data, val_data, test_data = self.process_dataset()

        logger.info("Creating Hugging Face datasets...")
        dataset_dict = DatasetDict(
            {
                "train": Dataset.from_list(train_data),
            }
        )

        logger.info(f"Pushing dataset to {repo_name}...")
        try:
            dataset_dict.push_to_hub(repo_name, token=token)
            logger.info("✅ Successfully pushed to Hugging Face Hub!")
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")
            raise


def main():
    processor = UnifiedMCQAProcessor()

    # Save locally
    processor.process_and_save()

    # Push to Hugging Face Hub
    processor.push_to_hub("jonlecumberri/unified_mcqa_all")


if __name__ == "__main__":
    main()
