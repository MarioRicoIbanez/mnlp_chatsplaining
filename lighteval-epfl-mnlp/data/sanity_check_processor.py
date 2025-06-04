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


class SanityCheckProcessor(BaseDatasetProcessor):
    """Processor for the zechen-nlp/MNLP_STEM_mcqa_evals dataset (test split only)."""

    def __init__(self):
        super().__init__()

    def process_dataset(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        logger.info("Loading MNLP_STEM_mcqa_evals dataset...")
        try:
            dataset = load_dataset("zechen-nlp/MNLP_STEM_mcqa_evals")
            logger.info("Dataset loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        test_data = self._process_split(dataset.get("test", []), split_name="test")
        return [], [], test_data  # Only test split exists

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
        Process a single MCQA item into the unified format with lettered choices.
        """
        try:
            question = item["question"].strip()
            choices = item["choices"]
            answer_letter = item["answer"].strip().upper()
            # Letter labels: A, B, C, D, E, F, G
            labels = [chr(ord('A') + i) for i in range(len(choices))]
            formatted_choices = [f"{label}. {choice.strip()}" for label, choice in zip(labels, choices)]
            # Find answer index from letter
            answer_index = labels.index(answer_letter)
            answer_text = formatted_choices[answer_index]
            return {
                "question": question,
                "choices": formatted_choices,
                "answer_index": answer_index,
                "answer_text": answer_text,
                "source": "mnlp_stem_mcqa_evals",
                "explanation": "",
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
        logger.info("Processing and validating dataset for push...")
        _, _, test_data = self.process_dataset()
        dataset_dict = DatasetDict({"test": Dataset.from_list(test_data)})
        logger.info(f"Pushing to Hugging Face Hub at: {repo_name}...")
        dataset_dict.push_to_hub(
            repo_name,
            token=token,
            commit_message="Upload processed MNLP_STEM_mcqa_evals in MCQA format",
        )
        logger.info("✅ Successfully pushed to Hugging Face Hub!")

def main():
    processor = SanityCheckProcessor()
    processor.process_and_save()
    # Descomenta la siguiente línea y pon tu repo destino para hacer push
    processor.push_to_hub("RikoteMaster/sanity_check_dataset")

if __name__ == "__main__":
    main() 