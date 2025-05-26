import os
import logging
from typing import List
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
# from dotenv import load_dotenv
from tqdm import tqdm  # Progress bar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_datasets(dataset_paths: List[str]) -> DatasetDict:
    logger.info("Loading and merging datasets...")
    merged_splits = {"train": [], "validation": [], "test": []}
    required_columns = ["question", "answer", "source", "explanation"]
    MAX_EXAMPLES = 10000  # Limit for OpenCode and OpenMath
    LIMITED_DATASETS = ["RikoteMaster/OpenCodeTreated", "RikoteMaster/OpenMathTreated"]

    for path in tqdm(dataset_paths, desc="Datasets"):
        logger.info(f"Loading dataset: {path}")
        try:
            ds = load_dataset(path)
            # Limit only OpenCode and OpenMath to 10k examples
            if path in LIMITED_DATASETS:
                for split in ds:
                    if len(ds[split]) > MAX_EXAMPLES:
                        ds[split] = ds[split].select(range(MAX_EXAMPLES))
                        logger.info(f"   ⚠ Limited {path} {split} to {MAX_EXAMPLES} examples")
        except Exception as e:
            logger.warning(f"⚠ Failed to load {path} normally. Trying streaming fallback... Error: {e}")
            try:
                ds_stream = load_dataset(path, split="train", streaming=True)
                # Use different limits based on dataset
                limit = MAX_EXAMPLES if path in LIMITED_DATASETS else 500
                examples = [ex for _, ex in zip(range(limit), ds_stream)]
                ds = DatasetDict({"train": Dataset.from_list(examples)})
                logger.info(f"   ✔ Fallback stream loaded: {len(examples)} examples")
            except Exception as e2:
                logger.error(f"❌ Failed to load dataset {path} even via fallback: {e2}")
                continue

        for split in tqdm(merged_splits, desc=f"→ Processing splits for {path}", leave=False):
            if split in ds:
                try:
                    dataset_split = ds[split].remove_columns(
                        [col for col in ds[split].column_names if col not in required_columns]
                    )
                    dataset_split = dataset_split.select_columns(required_columns)
                    merged_splits[split].append(dataset_split)
                    logger.info(f"   ✔ {split}: {len(dataset_split)} examples")
                except Exception as e:
                    logger.warning(f"⚠ Failed to process split '{split}' in {path}: {e}")

    logger.info("Concatenating splits...")
    merged_dataset = DatasetDict()
    for split, datasets in tqdm(merged_splits.items(), desc="Concatenating splits"):
        if datasets:
            merged_dataset[split] = concatenate_datasets(datasets)
            logger.info(f"✅ Merged {split}: {len(merged_dataset[split])} examples")
        else:
            logger.warning(f"⚠ No data found for split: {split}")

    return merged_dataset


def push_merged_dataset(dataset_dict: DatasetDict, repo_name: str, env_token_key: str = "HF_TOKEN"):
    # load_dotenv()
    # token = os.getenv(env_token_key)
    token = ""
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
        "RikoteMaster/OpenMathTreated",
        "jonlecumberri/stackexchange_engineering_openqa",
        "jonlecumberri/camel_chemistry_openqa"
    ]

    repo_name = "RikoteMaster/OpenQA_merged"

    merged_dataset = merge_datasets(dataset_paths)
    push_merged_dataset(merged_dataset, repo_name)


if __name__ == "__main__":
    main()

