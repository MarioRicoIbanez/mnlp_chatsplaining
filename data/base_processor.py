from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
import logging
from datasets import Dataset, DatasetDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseDatasetProcessor(ABC):
    """Base class for all dataset processors."""

    def __init__(
        self,
        output_dir: str = "output",
        hf_repo: Optional[str] = None,
        hf_subset: Optional[str] = None,
    ):
        """
        Initialize the dataset processor.

        Args:
            output_dir (str): Directory to save processed datasets
            hf_repo (str, optional): Hugging Face repository name
            hf_subset (str, optional): Hugging Face dataset subset name
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hf_repo = hf_repo
        self.hf_subset = hf_subset

    @abstractmethod
    def process_dataset(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Process the dataset into MCQA format.

        Returns:
            Tuple[List[Dict], List[Dict], List[Dict]]: Training, validation, and test data in MCQA format
        """
        pass

    def save_to_json(self, data: List[Dict], filename: str) -> None:
        """
        Save processed data to a JSON file.

        Args:
            data (List[Dict]): Data to save
            filename (str): Name of the output file
        """
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} examples to {output_path}")

    def validate_mcqa_format(self, data: List[Dict]) -> bool:
        """
        Validate that the data follows the MCQA format required for training.

        Required format for each item:
        {
            "question": str,
            "choices": List[str],  # Must be between 2 and 5 choices
            "answer_index": int,   # Index of correct answer
            "answer_text": str,    # Text of correct answer
            "source": str,         # Source of the question
            "explanation": str     # Explanation for the answer
        }
        """
        required_fields = {
            "question",
            "choices",
            "answer_index",
            "answer_text",
            "source",
            "explanation",
        }

        for item in data:
            # Check all required fields are present
            if not all(field in item for field in required_fields):
                logger.error(f"Missing required fields in item: {item}")
                return False

            # Check choices is a list of 2–5 items
            if not isinstance(item["choices"], list) or not (
                2 <= len(item["choices"]) <= 7
            ):
                logger.error(
                    f"Choices must be a list of 2 to 5 items: {item['choices']}"
                )
                return False

            # Check answer_index is within range
            if not isinstance(item["answer_index"], int) or not (
                0 <= item["answer_index"] < len(item["choices"])
            ):
                logger.error(
                    f"answer_index must be an integer within the choices range: {item['answer_index']}"
                )
                return False

            # Check answer_text matches the choice at answer_index
            if item["answer_text"] != item["choices"][item["answer_index"]]:
                logger.error(
                    f"answer_text does not match the choice at answer_index: {item}"
                )
                return False

        return True

    def create_dataset_dict(
        self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]
    ) -> DatasetDict:
        """
        Create a DatasetDict from the processed data.

        Args:
            train_data (List[Dict]): Training data
            val_data (List[Dict]): Validation data
            test_data (List[Dict]): Test data

        Returns:
            DatasetDict: Hugging Face dataset dictionary
        """
        return DatasetDict(
            {
                "train": Dataset.from_list(train_data),
                "validation": Dataset.from_list(val_data),
                "test": Dataset.from_list(test_data),
            }
        )

    def process_and_save(self) -> None:
        """
        Process the dataset and save training, validation, and test data.
        """
        train_data, val_data, test_data = self.process_dataset()

        if not self.validate_mcqa_format(train_data):
            raise ValueError("Training data does not follow the required MCQA format")
        if not self.validate_mcqa_format(val_data):
            raise ValueError("Validation data does not follow the required MCQA format")
        if not self.validate_mcqa_format(test_data):
            raise ValueError("Test data does not follow the required MCQA format")

        self.save_to_json(train_data, "mcqa_train.json")
        self.save_to_json(val_data, "mcqa_validation.json")
        self.save_to_json(test_data, "mcqa_test.json")

        logger.info(f"✅ Successfully processed and saved dataset")
        logger.info(f"✅ Training examples: {len(train_data)}")
        logger.info(f"✅ Validation examples: {len(val_data)}")
        logger.info(f"✅ Test examples: {len(test_data)}")
