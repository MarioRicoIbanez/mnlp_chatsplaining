import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
import logging
from typing import Dict, List
import statistics
from tqdm import tqdm
import psutil
import os
import torch
from torch.utils.data import DataLoader
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_token_lengths_batch(
    tokenizer, batch: Dict[str, List], device: str
) -> List[Dict[str, int]]:
    """Calculate token lengths for a batch of examples using GPU."""
    # Tokenize in batches
    with torch.no_grad():
        question_tokens = tokenizer(
            batch["question"], return_tensors="pt", padding=True, truncation=True
        ).to(device)
        answer_tokens = tokenizer(
            batch["answer"], return_tensors="pt", padding=True, truncation=True
        ).to(device)
        explanation_tokens = tokenizer(
            batch["explanation"], return_tensors="pt", padding=True, truncation=True
        ).to(device)

        # Combine texts for total length
        combined_texts = [
            f"{q} {a} {e}"
            for q, a, e in zip(batch["question"], batch["answer"], batch["explanation"])
        ]
        total_tokens = tokenizer(
            combined_texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)

    # Calculate lengths (excluding padding)
    lengths = []
    for i in range(len(batch["question"])):
        lengths.append(
            {
                "question_tokens": (question_tokens.attention_mask[i] == 1)
                .sum()
                .item(),
                "answer_tokens": (answer_tokens.attention_mask[i] == 1).sum().item(),
                "explanation_tokens": (explanation_tokens.attention_mask[i] == 1)
                .sum()
                .item(),
                "total_tokens": (total_tokens.attention_mask[i] == 1).sum().item(),
            }
        )

    return lengths


def print_token_statistics(token_lengths: List[Dict[str, int]], field: str):
    """Print comprehensive statistics for token lengths."""
    values = [x[field] for x in token_lengths]
    logger.info(f"\n📊 {field.replace('_', ' ').title()} Statistics:")
    logger.info(f"   Min: {min(values):.2f}")
    logger.info(f"   Max: {max(values):.2f}")
    logger.info(f"   Mean: {statistics.mean(values):.2f}")
    logger.info(f"   Median: {statistics.median(values):.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze token lengths in OpenQA_merged dataset"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        help="Optional: Number of examples to analyze (default: full dataset)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for GPU processing"
    )
    args = parser.parse_args()

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load Qwen tokenizer
    logger.info("Loading Qwen3-0.6B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", trust_remote_code=True, padding_side="right"
    )
    # Ensure we have the correct special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    logger.info("Loading OpenQA_merged dataset...")
    dataset = load_dataset("RikoteMaster/OpenQA_merged", split="train")

    # Get dataset size
    dataset_size = len(dataset)
    if args.num_examples:
        dataset_size = min(args.num_examples, dataset_size)
        logger.info(f"Analyzing {dataset_size} examples (limited by --num_examples)")
    else:
        logger.info(f"Analyzing full dataset ({dataset_size} examples)")

        # Check available memory
        available_memory = psutil.virtual_memory().available / (
            1024 * 1024 * 1024
        )  # in GB
        estimated_memory = dataset_size * 0.0001  # rough estimate: 0.1MB per example
        logger.warning(f"Available memory: {available_memory:.1f}GB")
        logger.warning(f"Estimated memory needed: {estimated_memory:.1f}GB")
        if (
            estimated_memory > available_memory * 0.8
        ):  # if we need more than 80% of available memory
            logger.warning(
                "⚠️ Warning: This might use a lot of memory. Consider using --num_examples to limit the analysis."
            )

    # Create DataLoader for efficient batching
    dataset = dataset.select(range(dataset_size))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: {k: [d[k] for d in x] for k in x[0].keys()},
    )

    # Get token lengths for all examples
    logger.info("Calculating token lengths...")
    token_lengths = []
    for batch in tqdm(dataloader, desc="Processing examples"):
        batch_lengths = get_token_lengths_batch(tokenizer, batch, device)
        token_lengths.extend(batch_lengths)

    # Sort by explanation length to find longest explanations
    token_lengths.sort(key=lambda x: x["explanation_tokens"], reverse=True)

    # Print statistics for each field
    for field in [
        "question_tokens",
        "answer_tokens",
        "explanation_tokens",
        "total_tokens",
    ]:
        print_token_statistics(token_lengths, field)


if __name__ == "__main__":
    main()
