import os
import logging
from typing import List, Dict
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from dotenv import load_dotenv
from tqdm import tqdm  # Progress bar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_datasets(dataset_paths: List[str]) -> DatasetDict:
    logger.info("Loading and merging datasets...")
    merged_splits = {"train": [], "validation": [], "test": []}
    required_columns = ["question", "answer", "source", "explanation"]

    for path in tqdm(dataset_paths, desc="Datasets"):
        logger.info(f"Loading dataset: {path}")
        try:
            ds = load_dataset(path)
        except Exception as e:
            logger.error(f"Failed to load dataset {path}: {e}")
            continue

        for split in tqdm(merged_splits, desc=f"→ Processing splits for {path}", leave=False):
            if split in ds:
                dataset_split = ds[split].remove_columns(
                    [col for col in ds[split].column_names if col not in required_columns]
                )
                dataset_split = dataset_split.select_columns(required_columns)

                merged_splits[split].append(dataset_split)
                logger.info(f"   ✔ {split}: {len(dataset_split)} examples")

    logger.info("Concatenating splits...")
    merged_dataset = DatasetDict()
    for split, datasets in tqdm(merged_splits.items(), desc="Concatenating splits"):
        if datasets:
            merged_dataset[split] = concatenate_datasets(datasets)
            logger.info(f"✅ Merged {split}: {len(merged_dataset[split])} examples")
        else:
            logger.warning(f"⚠️ No data found for split: {split}")

    return merged_dataset


def push_merged_dataset(dataset_dict: DatasetDict, repo_name: str, env_token_key: str = "HF_TOKEN"):
    load_dotenv()
    token = os.getenv(env_token_key)
    if not token:
        raise ValueError(f"Hugging Face token not found in environment variable '{env_token_key}'")

    logger.info(f"Pushing merged dataset to {repo_name}...")
    dataset_dict.push_to_hub(
        repo_name,
        token=token,
        commit_message="Merged OpenQA datasets"
    )
    logger.info("✅ Successfully pushed merged dataset!")


def main():
    dataset_paths = [
        "RikoteMaster/OpenCodeTreated",
        "jRikoteMaster/OpenMathTreated",
        "jonlecumberri/stackexchange_engineering_openqa",
        "jonlecumberri/camel_chemistry_openqa"
    ]

    repo_name = "jonlecumberri/openqa_merged"

    merged_dataset = merge_datasets(dataset_paths)
    push_merged_dataset(merged_dataset, repo_name)


if __name__ == "__main__":
    main()
