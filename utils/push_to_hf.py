from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
import pandas as pd
import os
from typing import Union, Dict, List, Optional
import logging
import json
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetUploader:
    def __init__(self, repo_id: str, token: str = None, output_dir: str = "output"):
        """
        Initialize the DatasetUploader with Hugging Face credentials.
        
        Args:
            repo_id (str): The repository ID on Hugging Face (format: "username/dataset-name")
            token (str, optional): Hugging Face API token. If None, will look for HF_TOKEN environment variable
            output_dir (str): Directory containing the processed dataset files
        """
        self.repo_id = repo_id
        self.token = token or os.getenv('HF_TOKEN')
        if not self.token:
            raise ValueError("Hugging Face token is required. Either pass it directly or set HF_TOKEN environment variable.")
        
        self.api = HfApi(token=self.token)
        self.output_dir = Path(output_dir)
    
    def validate_mcqa_format(self, data: List[Dict]) -> bool:
        """
        Validate that the data follows the MCQA format used in lighteval.
        
        Required format for each item:
        {
            "question": str,
            "choices": List[str],  # Must be exactly 4 choices
            "answer_index": int,   # Index of correct answer (0-3)
            "answer_text": str,    # Text of correct answer
            "source": str,         # Source of the question
            "explanation": str     # Explanation for the answer
        }
        """
        required_fields = {"question", "choices", "answer_index", "answer_text", "source", "explanation"}
        
        for item in data:
            # Check all required fields are present
            if not all(field in item for field in required_fields):
                logger.error(f"Missing required fields in item: {item}")
                return False
            
            # Check choices is a list of exactly 4 items
            if not isinstance(item["choices"], list) or len(item["choices"]) != 4:
                logger.error(f"Choices must be a list of exactly 4 items: {item['choices']}")
                return False
            
            # Check answer_index is valid
            if not isinstance(item["answer_index"], int) or item["answer_index"] not in range(4):
                logger.error(f"answer_index must be an integer between 0 and 3: {item['answer_index']}")
                return False
            
            # Check answer_text matches the choice at answer_index
            if item["answer_text"] != item["choices"][item["answer_index"]]:
                logger.error(f"answer_text does not match the choice at answer_index: {item}")
                return False
        
        return True
    
    def create_dataset(self, data: List[Dict]) -> Dataset:
        """
        Create a Hugging Face dataset from MCQA format data.
        
        Args:
            data: List of dictionaries in MCQA format
            
        Returns:
            Dataset: Hugging Face dataset object
        """
        if not self.validate_mcqa_format(data):
            raise ValueError("Data does not follow the required MCQA format")
            
        return Dataset.from_list(data)
    
    def create_dataset_dict(self, datasets: Dict[str, Dataset]) -> DatasetDict:
        """
        Create a DatasetDict from multiple datasets.
        
        Args:
            datasets (Dict[str, Dataset]): Dictionary mapping split names to datasets
            
        Returns:
            DatasetDict: Hugging Face dataset dictionary
        """
        return DatasetDict(datasets)
    
    def push_to_hub(self, dataset: Union[Dataset, DatasetDict], 
                    commit_message: str = "Update dataset",
                    private: bool = False) -> None:
        """
        Push the dataset to Hugging Face Hub.
        
        Args:
            dataset: Dataset or DatasetDict to push
            commit_message (str): Commit message for the push
            private (bool): Whether the dataset should be private
        """
        try:
            dataset.push_to_hub(
                repo_id=self.repo_id,
                token=self.token,
                commit_message=commit_message,
                private=private
            )
            logger.info(f"Successfully pushed dataset to {self.repo_id}")
        except Exception as e:
            logger.error(f"Failed to push dataset: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Upload MCQA dataset to Hugging Face Hub.")
    parser.add_argument('--name', type=str, required=True, help='Hugging Face dataset repo name (e.g. username/dataset-name)')
    parser.add_argument('--output-dir', type=str, default="output", help='Directory containing the processed dataset files')
    parser.add_argument('--private', action='store_true', help='Make the dataset private')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        raise ValueError(f"Output directory {output_dir} does not exist")

    files = [f for f in output_dir.glob("mcqa_*.json")]
    if not files:
        print("No mcqa_*.json files found in output directory.")
        return

    print("Available dataset files in output directory:")
    for idx, fname in enumerate(files):
        print(f"[{idx}] {fname.name}")
    
    selection = input("Enter the number(s) of the file(s) to upload (comma-separated, e.g. 0 or 0,1): ")
    selected_indices = [int(i.strip()) for i in selection.split(",") if i.strip().isdigit() and int(i.strip()) < len(files)]
    if not selected_indices:
        print("No valid selection made. Exiting.")
        return

    selected_files = [files[i] for i in selected_indices]
    datasets = {}
    
    uploader = DatasetUploader(repo_id=args.name, output_dir=args.output_dir)
    
    for fname in selected_files:
        split_name = fname.stem.replace("mcqa_", "")
        with open(fname, "r", encoding="utf-8") as f:
            data = json.load(f)
        ds = uploader.create_dataset(data)
        datasets[split_name] = ds

    dataset_dict = uploader.create_dataset_dict(datasets)
    uploader.push_to_hub(
        dataset_dict, 
        commit_message=f"Upload {', '.join(datasets.keys())} splits",
        private=args.private
    )

if __name__ == "__main__":
    main() 