from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
import logging
from datasets import Dataset, DatasetDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseOpenQAProcessor(ABC):
    """Base class for all open-answer dataset processors."""

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
        Process the dataset into OpenQA format.

        Returns:
            Tuple[List[Dict], List[Dict], List[Dict]]: Training, validation, and test data in OpenQA format
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

    def validate_openqa_format(self, data: List[Dict]) -> bool:
        """
        Validate that the data follows the open-answer format.

        Required format for each item:
        {
            "question": str,
            "answer": str,
            "source": str,
            "explanation": Optional[str]  # Explanation is optional but recommended
        }
        """
        required_fields = {"question", "answer", "source"}

        for item in data:
            if not all(field in item for field in required_fields):
                logger.error(f"Missing required fields in item: {item}")
                return False
            if not isinstance(item["question"], str) or not item["question"].strip():
                logger.error(f"Invalid question: {item['question']}")
                return False
            if not isinstance(item["answer"], str) or not item["answer"].strip():
                logger.error(f"Invalid answer: {item['answer']}")
                return False
            if not isinstance(item["source"], str):
                logger.error(f"Invalid source: {item['source']}")
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

        if not self.validate_openqa_format(train_data):
            raise ValueError("Training data does not follow the required OpenQA format")
        if not self.validate_openqa_format(val_data):
            raise ValueError(
                "Validation data does not follow the required OpenQA format"
            )
        if not self.validate_openqa_format(test_data):
            raise ValueError("Test data does not follow the required OpenQA format")

        self.save_to_json(train_data, "openqa_train.json")
        self.save_to_json(val_data, "openqa_validation.json")
        self.save_to_json(test_data, "openqa_test.json")

        logger.info(f"✅ Successfully processed and saved OpenQA dataset")
        logger.info(f"✅ Training examples: {len(train_data)}")
        logger.info(f"✅ Validation examples: {len(val_data)}")
        logger.info(f"✅ Test examples: {len(test_data)}")
