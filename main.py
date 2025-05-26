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

# Local imports
from utils.dataset_utils import tokenize_func, SFTDataCollator, process_open_answer_dataset, process_mcq_dataset
from utils.model_utils import load_model
from utils.train_utils import plot_training_loss
from utils.eval_utils import evaluate_openqa, evaluate_model_on_samples, load_mcqa_test_data, load_openqa_test_data

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate a language model')
    parser.add_argument('--dataset', type=str, default="jonlecumberri/mcqa_merged",
                      help='Dataset to use for training (default: jonlecumberri/mcqa_merged)')
    parser.add_argument('--output_name', type=str, default="Qwen3-0.6B-SFT-MCQA",
                      help='Name for output directory and HuggingFace push (default: Qwen3-0.6B-SFT-MCQA)')
    parser.add_argument('--num_train_samples', type=int, default=None,
                      help='Number of training samples to use (default: use full dataset)')
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
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.warning("HF_TOKEN not found in environment variables. Model pushing will be skipped.")


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory based on output name
    output_dir = SCRIPT_DIR / args.output_name
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Using dataset: {args.dataset}")
    logger.info(f"Output name: {args.output_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Clear any existing GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 1. Model and Tokenizer Setup
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model(model_name="Qwen/Qwen3-0.6B", load_in_4bit=False)
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
            remove_columns=train_dataset.column_names
        )
    else: 
        logger.info("Processing OpenAnswer dataset...")
        train_dataset = train_dataset.map(
            process_open_answer_dataset, 
            fn_kwargs={"tokenizer": tokenizer},
            batch_size=1,  # Process one at a time  
            remove_columns=train_dataset.column_names
        )

    logger.info("Tokenizing dataset...")
    tokenized_dataset = train_dataset.map(
        tokenize_func, 
        fn_kwargs={"tokenizer": tokenizer}, # Shorter sequences
        batch_size=1,  # Process one at a time
        remove_columns=train_dataset.column_names
    )

    # 3. Training Setup
    data_collator = SFTDataCollator(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(SCRIPT_DIR / output_dir),
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Increased for stability
        logging_steps=5,  # More frequent logging for small dataset
        save_steps=1000,
        report_to="wandb",
        bf16=True,
        disable_tqdm=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,  # Enable gradient checkpointing
        dataloader_num_workers=0,     # Reduce memory usage
        dataloader_pin_memory=False,  # Disable pin memory
        max_grad_norm=1.0,           # Gradient clipping for stability
        warmup_steps=1,              # Minimal warmup for small dataset
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        args=training_args,
    )
    logger.info("Trainer initialized successfully")

    # 4. Training
    trainer.train()

    logger.info("Pushing model to HuggingFace...")
    if HF_TOKEN:
        trainer.model.push_to_hub(f"RikoteMaster/{args.output_name}", private=True, token=HF_TOKEN)
        logger.info("Model pushed to HuggingFace successfully")
    else:
        logger.warning("Skipping model push to HuggingFace - no token provided")
    # Clear GPU cache after training
    torch.cuda.empty_cache()

    # 5. Save model
    logger.info("Saving model...")
    model.save_pretrained(SCRIPT_DIR / output_dir)
    tokenizer.save_pretrained(SCRIPT_DIR / output_dir)
    logger.info("Model saved successfully")

    # Push to HuggingFace

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
            temperature=0.7
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
            temperature=0.7
        )
    
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()




