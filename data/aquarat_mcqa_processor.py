from datasets import load_dataset, Dataset, DatasetDict
from typing import List, Dict, Tuple
import os
import sys
from pathlib import Path
import logging
import re
from dotenv import load_dotenv

# Add parent directory to path when running as script
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent))
    from data.base_processor import BaseDatasetProcessor
else:
    from .base_processor import BaseDatasetProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AquaRATProcessor(BaseDatasetProcessor):
    """Processor for the AQUA-RAT dataset."""

    def process_dataset(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        logger.info("Loading AQUA-RAT dataset (tokenized, train split)...")
        try:
            dataset = load_dataset("deepmind/aqua_rat", "tokenized", split="train")
            dataset = dataset.select(range(60000))  # Limit to first 50k examples
            logger.info("Dataset loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        train_data = self._process_split(dataset, split_name="train")
        return train_data, [], []

    def _process_split(self, split, split_name: str) -> List[Dict]:
        logger.info(f"Processing {split_name} split...")
        processed = []
        invalid_count = 0
        for i, item in enumerate(split):
            if i % 100 == 0:
                logger.info(f"Processed {i} {split_name} examples")
            processed_item = self._process_item(item)
            if processed_item:
                processed.append(processed_item)
            else:
                invalid_count += 1
        logger.info(f"Skipped {invalid_count} invalid items in {split_name} split")
        return processed

    def _normalize_choice(self, choice: str) -> str:
        # Removes leading label like 'A ) ' or 'B)'
        return re.sub(r"^[A-E]\s*\)\s*", "", choice).strip()

    def _process_item(self, item: Dict) -> Dict:
        try:
            question = item["question"].strip()
            raw_choices = [c.strip() for c in item.get("options", []) if isinstance(c, str)]
            choices = [self._normalize_choice(opt) for opt in raw_choices]

            answer_letter = item.get("correct", "").strip()
            letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

            if not choices or answer_letter not in letter_to_index:
                return None

            answer_index = letter_to_index[answer_letter]
            if answer_index >= len(choices):
                return None

            return {
                "question": question,
                "choices": choices,
                "answer_index": answer_index,
                "answer_text": choices[answer_index],
                "source": "aqua_rat",
                "explanation": item.get("rationale", "").strip()
            }
        except Exception as e:
            logger.warning(f"Error processing item: {e}\nRaw item: {item}")
            return None

    def push_to_hub(self, repo_name: str, env_token_key: str = "HF_TOKEN"):
        load_dotenv()
        token = os.getenv(env_token_key)
        if not token:
            raise ValueError(f"Hugging Face token not found in environment variable '{env_token_key}'")

        logger.info("Processing AQUA-RAT dataset for train split...")
        train_data, _, _ = self.process_dataset()

        logger.info(f"Processed {len(train_data)} training examples")

        dataset_dict = DatasetDict({
            "train": Dataset.from_list(train_data)
        })

        logger.info(f"Pushing dataset to {repo_name}...")
        try:
            dataset_dict.push_to_hub(repo_name, token=token)
            logger.info("✅ Successfully pushed to Hugging Face Hub!")
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")
            raise

def main():
    processor = AquaRATProcessor()
    processor.push_to_hub("jonlecumberri/aqua_rat_mcqa")

if __name__ == "__main__":
    main()