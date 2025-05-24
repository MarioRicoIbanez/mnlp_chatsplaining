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
    
    def __init__(self, output_dir: str = "output", hf_repo: Optional[str] = None, hf_subset: Optional[str] = None):
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
        Process the dataset into chat format.
        
        Returns:
            Tuple[List[Dict], List[Dict], List[Dict]]: Training, validation, and test data in chat format
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
    
    def validate_chat_format(self, data: List[Dict]) -> bool:
        """
        Validate that the data follows the chat format required for training.
        
        Required format for each item:
        {
            "conversations": [
                {"role": "user", "content": str},
                {"role": "assistant", "content": str}
            ]
        }
        """
        for item in data:
            # Check if conversations field exists
            if "conversations" not in item:
                logger.error(f"Missing 'conversations' field in item: {item}")
                return False
            
            # Check if conversations is a list
            if not isinstance(item["conversations"], list):
                logger.error(f"'conversations' must be a list: {item['conversations']}")
                return False
            
            # Check each conversation turn
            for turn in item["conversations"]:
                if not isinstance(turn, dict):
                    logger.error(f"Conversation turn must be a dictionary: {turn}")
                    return False
                
                if "role" not in turn or "content" not in turn:
                    logger.error(f"Missing 'role' or 'content' in turn: {turn}")
                    return False
                
                if turn["role"] not in ["user", "assistant"]:
                    logger.error(f"Invalid role in turn: {turn['role']}")
                    return False
                
                if not isinstance(turn["content"], str):
                    logger.error(f"Content must be a string: {turn['content']}")
                    return False
        
        return True
    
    def create_dataset_dict(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]) -> DatasetDict:
        """
        Create a DatasetDict from the processed data.
        
        Args:
            train_data (List[Dict]): Training data
            val_data (List[Dict]): Validation data
            test_data (List[Dict]): Test data
            
        Returns:
            DatasetDict: Hugging Face dataset dictionary
        """
        return DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data),
            "test": Dataset.from_list(test_data)
        })
    
    def process_and_save(self) -> None:
        """
        Process the dataset and save training, validation, and test data.
        """
        train_data, val_data, test_data = self.process_dataset()
        
        if not self.validate_chat_format(train_data):
            raise ValueError("Training data does not follow the required chat format")
        if not self.validate_chat_format(val_data):
            raise ValueError("Validation data does not follow the required chat format")
        if not self.validate_chat_format(test_data):
            raise ValueError("Test data does not follow the required chat format")
        
        self.save_to_json(train_data, "chat_train.json")
        self.save_to_json(val_data, "chat_validation.json")
        self.save_to_json(test_data, "chat_test.json")
        
        logger.info(f"✅ Successfully processed and saved dataset")
        logger.info(f"✅ Training examples: {len(train_data)}")
        logger.info(f"✅ Validation examples: {len(val_data)}")
        logger.info(f"✅ Test examples: {len(test_data)}") 