from datasets import load_dataset, Dataset, DatasetDict
from typing import List, Dict, Tuple, Optional
import os
import sys
from pathlib import Path
import re

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


class OpenMathProcessor(BaseOpenQAProcessor):
    """Processor for the NVIDIA OpenMathReasoning dataset."""

    def __init__(self, max_examples: Optional[int] = None, *args, **kwargs):
        """
        Initialize the processor.
        
        Args:
            max_examples (int, optional): Maximum number of examples to process. If None, process all examples.
            *args, **kwargs: Additional arguments passed to BaseOpenQAProcessor
        """
        super().__init__(*args, **kwargs)
        self.max_examples = max_examples

    def process_dataset(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Process the OpenMathReasoning dataset into OpenQA format.
        Uses the entire cot split as training data.
        
        Returns:
            Tuple[List[Dict], List[Dict], List[Dict]]: Training data and empty lists for validation/test
        """
        logger.info("Loading OpenMathReasoning dataset...")
        try:
            # Load the dataset with streaming to avoid memory issues
            dataset = load_dataset("nvidia/OpenMathReasoning", split="cot", streaming=True)
            logger.info("Dataset loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        # Process data into a list, respecting max_examples if set
        train_data = []
        total_examples = self.max_examples if self.max_examples is not None else float('inf')
        last_log_percentage = 0
        
        for i, item in enumerate(dataset):
            if self.max_examples is not None and i >= self.max_examples:
                break
                
            # Log progress every 10%
            current_percentage = int((i / total_examples) * 100)
            if current_percentage >= last_log_percentage + 10:
                logger.info(f"Processed {current_percentage}% of examples ({i} examples)")
                last_log_percentage = current_percentage
                
            processed_item = self._process_item(item)
            if processed_item:
                train_data.append(processed_item)

        logger.info(f"Finished processing. Got {len(train_data)} training examples")
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
        Process a single item from the OpenMathReasoning dataset.
        
        Args:
            item: Dictionary containing the item data
            
        Returns:
            Dict: Processed item in OpenQA format or None if invalid
        """
        try:
            # Extract thinking and answer from generated_solution
            thinking, answer = self._extract_thinking_and_answer(item["generated_solution"])
            
            # If no answer was found after </think>, use the expected_answer
            if not answer:
                answer = item["expected_answer"]
            
            return {
                "question": item["problem"],
                "answer": answer,
                "source": f"{item.get('problem_source', 'unknown')}NvidiaOpenMath",
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
                commit_message="Add OpenMathReasoning dataset in OpenQA format"
            )
            logger.info("✅ Successfully pushed to Hugging Face Hub!")
            logger.info(f"✅ Repository: {repo_name}")
            logger.info(f"✅ Train examples: {len(train_data)}")
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")
            raise


def main():
    # Process only 1000 examples
    processor = OpenMathProcessor(max_examples=100_000)
    
    # Option 1: Save locally
    processor.process_and_save()
    
    # Option 2: Push to Hugging Face Hub (uncomment and provide token)
    # processor.push_to_hub("your-repo-name")


if __name__ == "__main__":
    main() 