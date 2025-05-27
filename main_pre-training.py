# Standard library imports
import logging
import os
from pathlib import Path
import torch
import argparse

# Third-party imports
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, Trainer
from dotenv import load_dotenv
from torch.utils.data import DataLoader

# Local imports
from utils.dataset_utils import (
    tokenize_func,
    SFTDataCollator,
    process_open_answer_dataset,
    process_mcq_dataset,
)
from utils.model_utils import load_model
from utils.train_utils import plot_training_loss
from utils.eval_utils import (
    evaluate_openqa,
    evaluate_model_on_samples,
    load_mcqa_test_data,
    load_openqa_test_data,
)
from utils.batching import SmartPaddingTokenBatchSampler


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a language model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="RikoteMaster/OpenQA_merged",
        help="Dataset to use for training (default: RikoteMaster/OpenQA_merged)",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="Qwen3-0.6B-SFT-aux",
        help="Name for output directory and HuggingFace push (default: Qwen3-0.6B-SFT-MCQA)",
    )
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=None,
        help="Number of training samples to use (default: use full dataset)",
    )
    return parser.parse_args()


# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
FIGS_DIR = SCRIPT_DIR / "figs"

# Load environment variables from .env file
load_dotenv()

# Get HuggingFace token from environment
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.warning(
        "HF_TOKEN not found in environment variables. Model pushing will be skipped."
    )


def main():
    # Parse command line arguments
    args = parse_args()

    # Create output directory based on output name
    output_dir = SCRIPT_DIR / "results_model" / args.output_name
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Using dataset: {args.dataset}")
    logger.info(f"Output name: {args.output_name}")
    logger.info(f"Output directory: {output_dir}")

    # Clear any existing GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. Model and Tokenizer Setup
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model(
        model_name="RikoteMaster/Qwen3-0.6B-SFT-Open", load_in_4bit=False
    )
    logger.info("Model and tokenizer loaded successfully")

    # 2. Dataset Preparation with streaming and strict memory limits
    logger.info("Loading dataset from HuggingFace with streaming...")
    streaming_dataset = load_dataset(args.dataset, split="train", streaming=True)

    # Convert streaming dataset to regular dataset
    logger.info("Converting dataset...")
    train_dataset = []
    for i, example in enumerate(streaming_dataset):
        if args.num_train_samples is not None and i >= args.num_train_samples:
            break
        train_dataset.append(example)
    train_dataset = Dataset.from_list(train_dataset)
    logger.info(f"Processing {len(train_dataset)} examples")

    # Process with memory-efficient mapping
    logger.info("Processing dataset...")
    is_mcqa = "choices" in train_dataset.column_names
    if is_mcqa:
        logger.info("Processing MCQA dataset...")
        train_dataset = train_dataset.map(
            process_mcq_dataset,
            fn_kwargs={"tokenizer": tokenizer},
            batch_size=1,  # Process one at a time
            remove_columns=train_dataset.column_names,
        )
    else:
        logger.info("Processing OpenAnswer dataset...")
        train_dataset = train_dataset.map(
            process_open_answer_dataset,
            fn_kwargs={"tokenizer": tokenizer},
            batch_size=1,  # Process one at a time
            remove_columns=train_dataset.column_names,
        )
    # Print first 5 samples showing just the text field
    logger.info("\nFirst 5 text samples:")
    for i in range(5):
        logger.info(f"\n{'='*80}")
        logger.info(f"SAMPLE {i+1}")
        logger.info(f"{'='*80}")
        # Access text directly from the dataset's text column
        text = train_dataset["text"][i]
        logger.info(text)
        logger.info(f"{'-'*80}")

    logger.info("Tokenizing dataset...")
    tokenized_dataset = train_dataset.map(
        tokenize_func,
        fn_kwargs={"tokenizer": tokenizer},  # Shorter sequences
        batch_size=1,  # Process one at a time
        remove_columns=train_dataset.column_names,
    )

    # Add sequence lengths to dataset
    logger.info("Computing sequence lengths...")
    tokenized_dataset = tokenized_dataset.map(
        lambda ex: {"length": len(ex["input_ids"])},
        num_proc=4,  # speeds it up
    )

    # Sort dataset by length
    logger.info("Sorting dataset by sequence length...")
    tokenized_dataset = tokenized_dataset.sort("length", reverse=True)

    # 3. Training Setup
    data_collator = SFTDataCollator(tokenizer=tokenizer)

    # Set up token-budget batching
    max_tok_per_gpu = 8_000  # fits comfortably in 24 GB with bf16
    sampler = SmartPaddingTokenBatchSampler(
        tokenized_dataset["length"], max_tok_per_gpu
    )

    # Create custom dataloader
    dl = DataLoader(
        tokenized_dataset,
        batch_sampler=sampler,
        collate_fn=data_collator,
        num_workers=0,  # or >0 if RAM allows
        pin_memory=False,
    )

    training_args = TrainingArguments(
        output_dir=str(SCRIPT_DIR / output_dir),
        overwrite_output_dir=True,
        num_train_epochs=1,
        gradient_accumulation_steps=4,  # Increased for stability
        logging_steps=20,  # More frequent logging for small dataset
        save_steps=1000,
        report_to="wandb",
        bf16=True,
        disable_tqdm=False,
        remove_unused_columns=True,
        gradient_checkpointing=True,  # Enable gradient checkpointing
        max_grad_norm=1.0,  # Gradient clipping for stability
        warmup_steps=1,  # Minimal warmup for small dataset
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,  # <- give None here…
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train_dataset = tokenized_dataset  #  so metrics still work
    trainer.get_train_dataloader = lambda: dl
    # 4. Training
    trainer.train()

    logger.info("Pushing model to HuggingFace...")
    if HF_TOKEN:
        trainer.model.push_to_hub(
            f"RikoteMaster/{args.output_name}", private=True, token=HF_TOKEN
        )
        logger.info("Model pushed to HuggingFace successfully")
    else:
        logger.warning("Skipping model push to HuggingFace - no token provided")
    # Clear GPU cache after training
    torch.cuda.empty_cache()

    # 5. Save model in results_model directory
    results_model_dir = SCRIPT_DIR / "results_model" / args.output_name
    results_model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving model to results directory: {results_model_dir}")
    model.save_pretrained(results_model_dir)
    tokenizer.save_pretrained(results_model_dir)
    logger.info("Model saved to results directory successfully")

    # Save model in original output directory
    logger.info("Saving model to output directory...")
    model.save_pretrained(SCRIPT_DIR / output_dir)
    tokenizer.save_pretrained(SCRIPT_DIR / output_dir)
    logger.info("Model saved successfully")

    # 6. Plotting and Saving Results
    logger.info("Generating and saving training loss plot...")
    plot_training_loss(trainer, FIGS_DIR)
    logger.info("Training loss plot saved successfully")

    # 7. Evaluation based on dataset type
    logger.info("Loading test samples for evaluation...")

    if is_mcqa:
        # For MCQA: Load from jonlecumberri/mcqa_merged test partition
        test_samples = load_mcqa_test_data(tokenizer=tokenizer, num_samples=6)

        # Evaluate using MCQA evaluation function
        logger.info("Evaluating MCQA model...")
        evaluate_model_on_samples(
            model=model,
            tokenizer=tokenizer,
            test_samples=test_samples,
            max_new_tokens=2000,
            temperature=0.7,
        )
    else:
        # For Open Answer: Load OpenCode and OpenMath samples
        test_samples = load_openqa_test_data(num_samples_per_dataset=3)

        # Evaluate using OpenQA evaluation function
        logger.info("Evaluating OpenQA model...")
        evaluate_openqa(
            model=model,
            tokenizer=tokenizer,
            test_samples=test_samples,
            max_new_tokens=2000,
            temperature=0.7,
        )

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
