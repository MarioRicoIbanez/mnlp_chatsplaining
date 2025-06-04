from datasets import load_dataset, Dataset, DatasetDict
from typing import List, Dict, Tuple
import logging
from pathlib import Path
import sys
from dotenv import load_dotenv
import os

# Import the base processor correctly
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent))
    from base_processor import BaseDatasetProcessor
else:
    from .base_processor import BaseDatasetProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenBookQAProcessor(BaseDatasetProcessor):
    """Processor for the AI2 OpenBookQA dataset with context prepended to the question."""

    def __init__(self, subset: str = "additional", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert subset == "additional", "Only the 'additional' subset is supported."
        self.subset = subset
        self.source_name = f"openbookqa_{subset}"

    def process_dataset(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        logger.info(f"Loading OpenBookQA dataset ({self.subset})...")
        dataset = load_dataset("allenai/openbookqa", self.subset)
        logger.info(f"Loaded splits: {list(dataset.keys())}")

        train_data = self._process_split(dataset.get("train", []), split_name="train")
        val_data = self._process_split(dataset.get("validation", []), split_name="validation")
        test_data = self._process_split(dataset.get("test", []), split_name="test")
        return train_data, val_data, test_data

    def _process_split(self, split, split_name: str) -> List[Dict]:
        logger.info(f"Processing {split_name} split ({len(split)} examples)...")
        processed = []
        for i, item in enumerate(split):
            if i % 100 == 0:
                logger.info(f"Processed {i} {split_name} examples")
            processed_item = self._process_item(item)
            if processed_item:
                processed.append(processed_item)
        return processed

    def _process_item(self, item: Dict) -> Dict:
        try:
            # Prepend context to the question
            context = item.get("fact1", "").strip()
            question_stem = item.get("question_stem", "").strip()
            question = f"Context related: {context}\n\n{question_stem}"

            # Choices are in a dict with 'text' and 'label'
            choices_text = item["choices"]["text"]
            choices_label = item["choices"]["label"]
            formatted_choices = [f"{label}. {text}" for label, text in zip(choices_label, choices_text)]

            answer_letter = item["answerKey"].strip()
            try:
                answer_index = choices_label.index(answer_letter)
            except ValueError:
                logger.warning(f"Answer letter {answer_letter} not in labels {choices_label}")
                return None
            answer_text = formatted_choices[answer_index]
            return {
                "question": question,
                "choices": formatted_choices,
                "answer_index": answer_index,
                "answer_text": answer_text,
                "source": self.source_name,
                "explanation": context  # Optionally, store context as explanation
            }
        except Exception as e:
            logger.warning(f"Error processing item: {e}\nRaw item: {item}")
            return None

def main():
    load_dotenv()
    processor = OpenBookQAProcessor(subset="additional", output_dir="output/openbookqa_additional")
    train_data, val_data, test_data = processor.process_dataset()
    dataset_dict = processor.create_dataset_dict(train_data, val_data, test_data)
    repo_name = "RikoteMaster/openbookqa_additional_mcqa"
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        logger.warning("No Hugging Face token found in environment. Skipping push to hub.")
    else:
        logger.info(f"Pushing OpenBookQA additional to Hugging Face Hub at {repo_name} ...")
        dataset_dict.push_to_hub(repo_name, token=token)
        logger.info("✅ Successfully pushed OpenBookQA additional to Hugging Face Hub!")

if __name__ == "__main__":
    main() 