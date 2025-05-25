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

class ARCProcessor(BaseDatasetProcessor):
    """Processor for the AI2 ARC dataset (Challenge and Easy sets)."""

    def __init__(self, subset: str = "ARC-Challenge"):
        super().__init__()
        self.subset = subset

    def process_dataset(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Process the AI2 ARC dataset into MCQA format.
        
        Returns:
            Tuple[List[Dict], List[Dict], List[Dict]]: Training, validation, and test data
        """
        logger.info(f"Loading ARC dataset subset: {self.subset}")
        try:
            dataset = load_dataset("allenai/ai2_arc", self.subset)
            logger.info("Dataset loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        train_data = self._process_split(dataset.get("train", []), split_name="train")
        val_data = self._process_split(dataset.get("validation", []), split_name="validation")
        test_data = self._process_split(dataset.get("test", []), split_name="test")

        return train_data, val_data, test_data

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
        Process a single ARC item into MCQA format.

        Args:
            item (Dict): A raw ARC question

        Returns:
            Dict: Processed question or None if invalid
        """
        try:
            question = item["question"]
            choices = item["choices"]["text"]
            correct_label = item["answerKey"]

            # Convert label to index: A=0, B=1, ...
            label_map = {chr(i + 65): i for i in range(len(choices))}
            answer_index = label_map.get(correct_label, -1)

            if answer_index == -1 or len(choices) != 4:
                logger.error(f"Choices must be a list of exactly 4 items: {choices}")
                return None

            return {
                "question": question,
                "choices": choices,
                "answer_index": answer_index,
                "answer_text": choices[answer_index],
                "source": f"ai2_arc_{self.subset.lower()}",
                "explanation": ""  # No explanations in ARC dataset
            }

        except Exception as e:
            logger.warning(f"Error processing item: {e}\nRaw item: {item}")
            return None


    def push_to_hub(self, repo_name: str,  env_token_key: str = "HF_TOKEN"):
        """
        Process the dataset and push all splits to Hugging Face Hub.
        
        Args:
            repo_name (str): The Hugging Face repository name
            token (str, optional): Hugging Face token
        """
        load_dotenv()
        token = os.getenv(env_token_key)
        if not token:
            raise ValueError(f"Hugging Face token not found in environment variable '{env_token_key}'")
        train_data, val_data, test_data = self.process_dataset()

        logger.info("Creating Hugging Face datasets...")
        dataset_dict = DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data),
            "test": Dataset.from_list(test_data)
        })

        logger.info(f"Pushing dataset to {repo_name}...")
        try:
            dataset_dict.push_to_hub(repo_name, token=token)
            logger.info("✅ Successfully pushed all splits to Hugging Face Hub!")
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")
            raise

def main():
    # Change to "ARC-Easy" if needed
    processor = ARCProcessor(subset="ARC-Challenge")

    # Option 1: Save locally
    processor.process_and_save()

    # Option 2: Push to HF Hub
    processor.push_to_hub("jonlecumberri/arc_challenge_mcqa")

if __name__ == "__main__":
    main()
