from datasets import load_dataset, Dataset, DatasetDict
from typing import List, Dict, Tuple, Optional
import os
import sys
from pathlib import Path
import re
import json

# Add parent directory to path when running as script
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent))
    from data.base_openans_processor import BaseOpenQAProcessor
else:
    from .base_openans_processor import BaseOpenQAProcessor

import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenCodeProcessor(BaseOpenQAProcessor):
    """Processor for the NVIDIA OpenCode dataset."""

    def __init__(self, max_examples: Optional[int] = None, *args, **kwargs):
        """
        Initialize the processor.
        
        Args:
            max_examples (int, optional): Maximum number of examples to process. If None, process all examples.
            *args, **kwargs: Additional arguments passed to BaseOpenQAProcessor
        """
        # Set the output directory to be inside the data folder
        if 'output_dir' not in kwargs:
            # Get the directory where this script is located (data folder)
            data_dir = Path(__file__).parent
            kwargs['output_dir'] = str(data_dir / "output")
        
        super().__init__(*args, **kwargs)
        self.max_examples = max_examples
        
        # Load the source datasets for question retrieval
        logger.info("Loading source datasets for question retrieval...")
        self.hf_datasets = {
            "taco": load_dataset("BAAI/TACO", trust_remote_code=True),
            "apps": load_dataset("codeparrot/apps", trust_remote_code=True),
            "code_contests": load_dataset("deepmind/code_contests"),
            "open-r1/codeforces": load_dataset("open-r1/codeforces")
        }
        logger.info("Source datasets loaded successfully")

    def _is_valid_difficulty(self, difficulty: str) -> bool:
        """
        Check if the difficulty level is valid.
        
        Args:
            difficulty: The difficulty level to check
            
        Returns:
            bool: True if the difficulty is valid, False otherwise
        """
        invalid_difficulties = {"EASY", "0", "UNKNOWN_DIFFICULTY", "2"}
        return difficulty not in invalid_difficulties

    def get_question(self, ds_name: str, split: str, index: int) -> Optional[str]:
        """
        Retrieve the actual question from the source dataset.
        
        Args:
            ds_name: Name of the source dataset
            split: Dataset split
            index: Index in the dataset
            
        Returns:
            The question text or None if not found
        """
        try:
            benchmark = self.hf_datasets[ds_name][split][int(index)]
            
            if ds_name == "code_contests":
                if not benchmark["description"]:
                    return None
                return benchmark["description"]
            elif ds_name in ["taco", "apps"]:
                return benchmark["question"]
            elif ds_name == "open-r1/codeforces":
                if not benchmark["description"]:
                    return None
                question = benchmark["description"]
                if benchmark["input_format"]:
                    question += "\n\nInput\n\n" + benchmark["input_format"]
                if benchmark["output_format"]:
                    question += "\n\nOutput\n\n" + benchmark["output_format"]
                if benchmark["examples"]:
                    question += "\n\nExamples"
                    for example in benchmark["examples"]:
                        if "input" in example:
                            question += "\n\nInput\n\n" + example["input"]
                        if "output" in example:
                            question += "\n\nOutput\n\n" + example["output"]
                if benchmark["note"]:
                    question += "\n\nNote\n\n" + benchmark["note"]
                return question
            
            return None
        except Exception as e:
            logger.warning(f"Failed to retrieve question for {ds_name}/{split}/{index}: {e}")
            return None

    def process_dataset(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Process the OpenCode dataset into OpenQA format.
        Filters out examples with invalid difficulty levels and low pass rates.
        Continues streaming until we get enough valid examples.
        
        Returns:
            Tuple[List[Dict], List[Dict], List[Dict]]: Training data and empty lists for validation/test
        """
        logger.info("Loading OpenCode dataset...")
        try:
            # Load the dataset with streaming to avoid memory issues
            dataset = load_dataset("nvidia/OpenCodeReasoning-2", split="python", streaming=True)
            logger.info("Dataset loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        # Process data into a list, respecting max_examples if set
        train_data = []
        target_valid_examples = self.max_examples if self.max_examples is not None else float('inf')
        last_log_percentage = 0
        total_checked = 0
        
        logger.info(f"Starting to process dataset, looking for {target_valid_examples} valid examples...")
        
        for item in dataset:
            total_checked += 1
            
            # Log progress every 10% of target valid examples found
            if self.max_examples is not None:
                current_percentage = int((len(train_data) / self.max_examples) * 100)
                if current_percentage >= last_log_percentage + 10 and current_percentage <= 100:
                    logger.info(f"Found {current_percentage}% of target valid examples ({len(train_data)}/{self.max_examples} valid found, checked {total_checked} total)")
                    last_log_percentage = current_percentage
            
            # Debug: Print first few items to understand structure
            if total_checked <= 3:
                logger.info(f"Item {total_checked} structure: {list(item.keys()) if isinstance(item, dict) else type(item)}")
                if isinstance(item, dict):
                    logger.info(f"Item {total_checked} difficulty: {item.get('difficulty')}, pass_rate: {item.get('pass_rate')}")
                
            # Skip if item is not a dictionary
            if not isinstance(item, dict):
                logger.warning(f"Item {total_checked} is not a dictionary: {type(item)}")
                continue
                
            # Skip examples that don't meet our criteria
            difficulty = str(item.get("difficulty", ""))
            if not self._is_valid_difficulty(difficulty):
                if total_checked <= 10:  # Debug first 10 items
                    logger.info(f"Skipping item {total_checked} due to difficulty: {difficulty}")
                continue
                
            pass_rate = float(item.get("pass_rate", 0))
            if pass_rate < 0.8:
                if total_checked <= 10:  # Debug first 10 items
                    logger.info(f"Skipping item {total_checked} due to pass_rate: {pass_rate}")
                continue
                
            processed_item = self._process_item(item)
            if processed_item:
                train_data.append(processed_item)
                
                # Stop if we've reached our target number of valid examples
                if len(train_data) >= target_valid_examples:
                    break

        logger.info(f"Finished processing. Got {len(train_data)} training examples after checking {total_checked} total examples")
        return train_data, [], []  # Return empty lists for val/test since we only have one split

    def _extract_thinking_and_answer(self, generated_solution: str) -> Tuple[str, str]:
        """
        Extract thinking process and answer from the generated solution.
        
        Args:
            generated_solution (str): The full generated solution containing <think> tags
            
        Returns:
            Tuple[str, str]: (thinking process, answer)
        """
        # Extract content between <think> tags
        think_match = re.search(r'<think>(.*?)</think>', generated_solution, re.DOTALL)
        if not think_match:
            return "", generated_solution.strip()
            
        thinking = think_match.group(1).strip()
        
        # Get the answer (everything after </think>)
        answer = generated_solution[think_match.end():].strip()
        
        return thinking, answer

    def _process_item(self, item: Dict) -> Dict:
        """
        Process a single item from the OpenCode dataset.
        
        Args:
            item: Dictionary containing the item data
            
        Returns:
            Dict: Processed item in OpenQA format or None if invalid
        """
        try:
            # Get the actual question from the source dataset
            ds_name = item.get("dataset")
            ds_split = item.get("split")
            ds_index = item.get("index")
            
            if not all([ds_name, ds_split, ds_index is not None]):
                logger.warning(f"Missing dataset info: dataset={ds_name}, split={ds_split}, index={ds_index}")
                return None
                
            question = self.get_question(ds_name, ds_split, ds_index)
            if question is None:
                logger.warning(f"Could not retrieve question for {ds_name}/{ds_split}/{ds_index}")
                return None
            
            # Extract thinking and answer from r1_generation
            thinking, answer = self._extract_thinking_and_answer(item["r1_generation"])
            
            # If no answer was found after </think>, use the expected_answer
            if not answer:
                answer = item.get("expected_answer", "")
            
            return {
                "question": question,
                "answer": answer,
                "source": f"{ds_name}NvidiaOpenCode",
                "explanation": thinking
            }
        except Exception as e:
            logger.warning(f"Error processing item: {e}")
            return None

    def push_to_hub(self, repo_name: str, env_token_key: str = "HF_TOKEN"):
        """
        Process the dataset and push to Hugging Face Hub.
        
        Args:
            repo_name (str): The Hugging Face repository name
            env_token_key (str): Environment variable name for the Hugging Face token
        """
        load_dotenv()
        token = os.getenv(env_token_key)
        if not token:
            raise ValueError(f"Hugging Face token not found in environment variable '{env_token_key}'")

        train_data, _, _ = self.process_dataset()

        logger.info("Creating Hugging Face datasets...")
        dataset_dict = DatasetDict({
            "train": Dataset.from_list(train_data)
        })

        logger.info(f"Pushing dataset to {repo_name}...")
        try:
            dataset_dict.push_to_hub(
                repo_name,
                token=token,
                commit_message="Add OpenCode dataset in OpenQA format"
            )
            logger.info("✅ Successfully pushed to Hugging Face Hub!")
            logger.info(f"✅ Repository: {repo_name}")
            logger.info(f"✅ Train examples: {len(train_data)}")
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")
            raise


def main():
    # Process only 10 examples for testing
    processor = OpenCodeProcessor(max_examples=100_000)
    
    # Option 1: Save locally
    processor.process_and_save()
    
    # Option 2: Push to Hugging Face Hub (uncomment and provide token)
    # processor.push_to_hub("your-repo-name")


if __name__ == "__main__":
    main() 