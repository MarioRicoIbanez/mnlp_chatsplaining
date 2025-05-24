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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SciQProcessor(BaseDatasetProcessor):
    """Processor for the SciQ dataset."""
    
    def process_dataset(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Process the SciQ dataset into MCQA format.
        
        Returns:
            Tuple[List[Dict], List[Dict], List[Dict]]: Training, validation, and test data in MCQA format
        """
        logger.info("Loading SciQ dataset...")
        try:
            # Load the SciQ dataset with streaming to avoid memory issues
            dataset = load_dataset("sciq", streaming=True)
            logger.info("Dataset loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        
        train_format = []
        val_format = []
        test_format = []
        
        # Process training data
        logger.info("Processing training data...")
        for i, item in enumerate(dataset["train"]):
            if i % 100 == 0:
                logger.info(f"Processed {i} training examples")
            processed_item = self._process_item(item)
            if processed_item:
                train_format.append(processed_item)
        
        # Process validation data
        logger.info("Processing validation data...")
        for i, item in enumerate(dataset["validation"]):
            if i % 10 == 0:
                logger.info(f"Processed {i} validation examples")
            processed_item = self._process_item(item)
            if processed_item:
                val_format.append(processed_item)
        
        # Process test data
        logger.info("Processing test data...")
        for i, item in enumerate(dataset["test"]):
            if i % 10 == 0:
                logger.info(f"Processed {i} test examples")
            processed_item = self._process_item(item)
            if processed_item:
                test_format.append(processed_item)
        
        logger.info(f"Finished processing. Got {len(train_format)} training, {len(val_format)} validation, and {len(test_format)} test examples")
        return train_format, val_format, test_format
    
    def _process_item(self, item: Dict) -> Dict:
        """
        Process a single item from the SciQ dataset.
        
        Args:
            item: Dictionary containing the item data
            
        Returns:
            Dict: Processed item in MCQA format or None if invalid
        """
        try:
            question = item["question"]
            correct = item["correct_answer"]
            distractors = [item["distractor1"], item["distractor2"], item["distractor3"]]
            
            # Skip examples with fewer than 3 distractors
            if len(distractors) != 3:
                return None
            
            # Combine correct answer and distractors
            choices = distractors + [correct]
            choices = list(set(choices))  # deduplicate
            
            # Skip if we don't have exactly 4 unique choices
            if len(choices) != 4:
                return None
            
            # Sort choices for deterministic order
            choices = sorted(choices)
            
            try:
                answer_index = choices.index(correct)
            except ValueError:
                return None  # correct answer not found
            
            # Create MCQA format example
            return {
                "question": question,
                "choices": choices,
                "answer_index": answer_index,
                "answer_text": choices[answer_index],
                "source": "sciq",
                "explanation": item["support"]
            }
        except Exception as e:
            logger.warning(f"Error processing item: {e}")
            return None
    
    def push_to_hub(self, repo_name: str = "RikoteMaster/sciq_treated_epfl_mcqa", token: str = None):
        """
        Process the dataset and push all splits (train, validation, test) to Hugging Face Hub.
        
        Args:
            repo_name (str): The Hugging Face repository name
            token (str, optional): Hugging Face token for authentication
        """
        logger.info("Processing SciQ dataset for all splits...")
        
        # Process the dataset
        train_data, val_data, test_data = self.process_dataset()
        
        logger.info(f"Processed {len(train_data)} training examples")
        logger.info(f"Processed {len(val_data)} validation examples") 
        logger.info(f"Processed {len(test_data)} test examples")
        
        # Create datasets
        logger.info("Creating Hugging Face datasets...")
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        test_dataset = Dataset.from_list(test_data)
        
        # Create DatasetDict
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })
        
        # Push to hub
        logger.info(f"Pushing dataset to {repo_name}...")
        try:
            dataset_dict.push_to_hub(
                repo_name,
                token=token,
                commit_message="Add validation and test splits from allenai/sciq"
            )
            logger.info("✅ Successfully pushed all splits to Hugging Face Hub!")
            logger.info(f"✅ Repository: {repo_name}")
            logger.info(f"✅ Train examples: {len(train_data)}")
            logger.info(f"✅ Validation examples: {len(val_data)}")
            logger.info(f"✅ Test examples: {len(test_data)}")
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")
            raise

def main():
    processor = SciQProcessor()
    
    # Option 1: Save locally
    processor.process_and_save()
    
    # Option 2: Push to Hugging Face Hub (uncomment and provide token)
    # processor.push_to_hub(token="your_hf_token_here")

if __name__ == "__main__":
    main() 