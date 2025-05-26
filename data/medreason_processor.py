from datasets import load_dataset, Dataset, DatasetDict
from typing import List, Dict, Tuple
import os
import sys
from pathlib import Path
import logging
import re
import string
from dotenv import load_dotenv

# Add parent directory to path when running as script
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent))
    from data.base_processor import BaseDatasetProcessor
else:
    from .base_processor import BaseDatasetProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedReasonProcessor(BaseDatasetProcessor):
    """Processor for the MedReason dataset."""

    def process_dataset(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        logger.info("Loading MedReason dataset...")
        try:
            dataset = load_dataset("UCSC-VLAA/MedReason")
            logger.info("Dataset loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        train_data = self._process_split(dataset.get("train", []), split_name="train")
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

    def _extract_answer_and_explanation(self, answer_field: str) -> Tuple[str, str]:
        parts = answer_field.split("Explanation:", 1)
        raw_answer = parts[0].strip()
        # Strip trailing punctuation (.,!?) and whitespace
        matchable_answer = raw_answer.rstrip(string.punctuation + " ").strip()
        full_answer_with_explanation = answer_field.strip()
        return matchable_answer, full_answer_with_explanation

    def _extract_choices(self, options_field: str) -> List[str]:
        """
        Extracts 2–5 answer options labeled A. to E.
        """
        pattern = r"[A-E]\.\s+(.*?)\n"
        matches = re.findall(pattern, options_field + "\n")  # ensure trailing newline
        return matches if 2 <= len(matches) <= 5 else []

    def _process_item(self, item: Dict) -> Dict:
        try:
            question = item["question"].strip()
            answer_field = item["answer"]
            reasoning_field = item.get("reasoning", "")
            options_field = item.get("options", "")

            answer, full_answer = self._extract_answer_and_explanation(answer_field)
            choices = self._extract_choices(options_field)

            # if type(choices) < 2 or len(choices) > 5:
            #     logger.warning(f"Invalid number of choices: {choices}")
            #     return None

            # try:
            #     answer_index = choices.index(answer)
            # except ValueError:
            #     logger.warning(f"Answer not found in choices: {answer}")
            #     return None

            return {
                "question": question,
                "choices": choices,
                "answer_index": (
                    choices.index(answer) if answer in choices else -1
                ),  # -1 if not found
                "answer_text": answer,  # matchable answer only
                "source": "medreason",
                "explanation": reasoning_field.strip(),
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
        logger.info("Processing MedReason dataset for train split...")
        train_data, _, _ = self.process_dataset()

        logger.info(f"Processed {len(train_data)} training examples")

        dataset_dict = DatasetDict({"train": Dataset.from_list(train_data)})

        logger.info(f"Pushing dataset to {repo_name}...")
        try:
            dataset_dict.push_to_hub(repo_name, token=token)
            logger.info("✅ Successfully pushed to Hugging Face Hub!")
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")
            raise


def main():
    processor = MedReasonProcessor()
    # processor.process_and_save()
    processor.push_to_hub("jonlecumberri/medreason_mcqa")


if __name__ == "__main__":

    main()
