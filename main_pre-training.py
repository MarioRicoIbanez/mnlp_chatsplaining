# Standard library imports
import logging
import os
from pathlib import Path
import torch
import argparse
import tempfile
import shutil

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR  # Assuming the script is in the project root

# Set up project directories relative to script location
RESULTS_DIR = PROJECT_ROOT / "results_model"
FIGS_DIR = RESULTS_DIR / "figs"
HF_CACHE_HOME_DIR = RESULTS_DIR / "hf_cache"

# Create necessary directories
RESULTS_DIR.mkdir(exist_ok=True)
FIGS_DIR.mkdir(exist_ok=True)

print(f"Script location: {SCRIPT_DIR}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Results directory: {RESULTS_DIR}")

# Set HuggingFace cache directory to faster location
# Try /tmp first (usually faster), fallback to project location

# Check available space in /tmp
def get_free_space_gb(path):
    """Get free space in GB for a given path"""
    try:
        statvfs = os.statvfs(path)
        return (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
    except (OSError, AttributeError):
        return 0

tmp_space = get_free_space_gb("/tmp")
home_space = get_free_space_gb(str(RESULTS_DIR))

print(f"Available space in /tmp: {tmp_space:.1f} GB")
print(f"Available space in project results: {home_space:.1f} GB")

# Use /tmp if it has at least 50GB free, otherwise use project directory
if tmp_space > 50:
    HF_CACHE_DIR = Path("/tmp") / f"hf_cache_ricoiban_{os.getpid()}"
    print(f"Using faster cache location: {HF_CACHE_DIR}")
else:
    HF_CACHE_DIR = HF_CACHE_HOME_DIR
    print(f"Using project cache location: {HF_CACHE_DIR}")

HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_DIR / "datasets")

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
        "--num_samples",
        type=int,
        default=None,
        help="Number of training samples to use (default: use full dataset)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Name of the model to load from HuggingFace (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to train for (default: 1)",
    )
    return parser.parse_args()


# Load environment variables from .env file (look in script directory first)
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"Loaded .env from: {env_file}")
else:
    load_dotenv()  # Try default locations
    print("Loaded .env from default locations")

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

    # Create output directory based on output name (relative to results directory)
    output_dir = RESULTS_DIR / args.output_name
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Script directory: {SCRIPT_DIR}")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Using dataset: {args.dataset}")
    logger.info(f"Output name: {args.output_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Cache directory: {HF_CACHE_DIR}")

    # Clear any existing GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. Model and Tokenizer Setup
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model(
        model_name=args.model_name, load_in_4bit=False
    )
    logger.info("Model and tokenizer loaded successfully")

    # 2. Dataset Preparation with streaming and strict memory limits
    logger.info("Loading dataset from HuggingFace with streaming...")
    streaming_dataset = load_dataset(args.dataset, split="train", streaming=True)

    # Convert streaming dataset to regular dataset
    logger.info("Converting dataset...")
    train_dataset = []
    for i, example in enumerate(streaming_dataset):
        if args.num_samples is not None and i >= args.num_samples:
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

    logger.info("INFO: Smart Batching Info:")
    logger.info(f"  - Max tokens per GPU: {max_tok_per_gpu}")
    logger.info(f"  - Dataset length: {len(tokenized_dataset)}")

    # Debug: Count total batches that will be created - do this properly
    batch_count = 0
    sample_batch_sizes = []
    temp_sampler = SmartPaddingTokenBatchSampler(
        tokenized_dataset["length"], max_tok_per_gpu
    )

    # Count ALL batches that will actually be generated
    logger.info("INFO: Counting actual batches that will be generated...")
    for batch_indices in temp_sampler:
        batch_count += 1
        sample_batch_sizes.append(len(batch_indices))
        if batch_count <= 5:  # Show first 5 batch sizes
            max_len = max([tokenized_dataset[i]["length"] for i in batch_indices])
            logger.info(f"  - Batch {batch_count}: {len(batch_indices)} samples, max_len: {max_len}")

    logger.info(f"  - Total ACTUAL batches: {batch_count}")
    logger.info(f"  - Average batch size: {sum(sample_batch_sizes) / len(sample_batch_sizes):.1f}")
    logger.info(f"  - Sampler.__len__() estimate: {len(sampler)}")

    # Validate our count
    if batch_count == 0:
        logger.error("❌ ERROR: No batches generated! Check your sampler configuration.")
        exit(1)

    # Create fresh sampler for actual training
    sampler = SmartPaddingTokenBatchSampler(
        tokenized_dataset["length"], max_tok_per_gpu
    )

    # Create custom dataloader with correct length reporting
    class CustomDataLoader(DataLoader):
        def __init__(self, actual_length, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Store the actual batch count we calculated
            self._actual_length = actual_length
        
        def __len__(self):
            return self._actual_length

    dl = CustomDataLoader(
        batch_count,  # Pass the actual batch count
        tokenized_dataset,
        batch_sampler=sampler,
        collate_fn=data_collator,
        num_workers=0,  # or >0 if RAM allows
        pin_memory=False,
    )

    # Create a wrapper dataloader that handles multiple epochs
    class EpochDataLoader:
        def __init__(self, dataloader, num_epochs):
            self.dataloader = dataloader
            self.num_epochs = num_epochs
            self.current_epoch = 0
            self.current_iter = None
        
        def __iter__(self):
            self.current_epoch = 0
            self.current_iter = iter(self.dataloader)
            return self
        
        def __next__(self):
            try:
                return next(self.current_iter)
            except StopIteration:
                self.current_epoch += 1
                if self.current_epoch >= self.num_epochs:
                    raise StopIteration
                self.current_iter = iter(self.dataloader)
                return next(self.current_iter)
        
        def __len__(self):
            return len(self.dataloader) * self.num_epochs

    # Wrap the dataloader to handle multiple epochs
    dl = EpochDataLoader(dl, args.epochs)

    # Try to set up evaluation dataloader
    eval_dl = None
    try:
        logger.info("Attempting to load evaluation dataset...")
        eval_dataset = load_dataset(args.dataset, split="validation", streaming=True)
        
        # Convert streaming dataset to regular dataset
        eval_dataset_list = []
        for example in eval_dataset:
            eval_dataset_list.append(example)
        eval_dataset = Dataset.from_list(eval_dataset_list)
        
        # Process evaluation dataset
        if is_mcqa:
            eval_dataset = eval_dataset.map(
                process_mcq_dataset,
                fn_kwargs={"tokenizer": tokenizer},
                batch_size=1,
                remove_columns=eval_dataset.column_names,
            )
        else:
            eval_dataset = eval_dataset.map(
                process_open_answer_dataset,
                fn_kwargs={"tokenizer": tokenizer},
                batch_size=1,
                remove_columns=eval_dataset.column_names,
            )
        
        # Tokenize evaluation dataset
        eval_dataset = eval_dataset.map(
            tokenize_func,
            fn_kwargs={"tokenizer": tokenizer},
            batch_size=1,
            remove_columns=eval_dataset.column_names,
        )
        
        # Add sequence lengths
        eval_dataset = eval_dataset.map(
            lambda ex: {"length": len(ex["input_ids"])},
            num_proc=4,
        )
        
        # Sort by length
        eval_dataset = eval_dataset.sort("length", reverse=True)
        
        # Create evaluation sampler
        eval_sampler = SmartPaddingTokenBatchSampler(
            eval_dataset["length"], max_tok_per_gpu
        )
        
        # Create evaluation dataloader
        eval_dl = CustomDataLoader(
            len(eval_sampler),
            eval_dataset,
            batch_sampler=eval_sampler,
            collate_fn=data_collator,
            num_workers=0,
            pin_memory=False,
        )
        
        logger.info(f"Successfully loaded evaluation dataset with {len(eval_dataset)} examples")
        
    except Exception as e:
        logger.warning(f"Could not load evaluation dataset: {str(e)}")
        logger.warning("Training will proceed without evaluation")
        eval_dl = None

    # Calculate expected steps for smart batching
    logger.info("INFO: Dataset info:")
    logger.info(f"  - Training samples: {len(tokenized_dataset)}")
    logger.info(f"  - Smart batch sampler will create dynamic batches")
    logger.info(f"  - Number of epochs: {args.epochs}")
    logger.info(f"  - Total steps per epoch: {batch_count}")
    logger.info(f"  - Total steps across all epochs: {batch_count * args.epochs}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),  # Use the output_dir directly
        overwrite_output_dir=True,
        # Use max_steps with actual batch count * number of epochs
        max_steps=batch_count * args.epochs,
        gradient_accumulation_steps=4,  # Increased for stability
        logging_steps=5,  # More frequent logging for small dataset
        learning_rate=1e-5,
        save_steps=1000,
        report_to="wandb",
        bf16=True,
        disable_tqdm=False,
        remove_unused_columns=True,
        gradient_checkpointing=True,  # Enable gradient checkpointing
        max_grad_norm=1.0,  # Gradient clipping for stability
        warmup_steps=1,  # Minimal warmup for small dataset
        # Add evaluation settings
        eval_strategy="steps" if eval_dl is not None else "no",
        eval_steps=10 if eval_dl is not None else None,
        load_best_model_at_end=True if eval_dl is not None else False,
        metric_for_best_model="eval_loss" if eval_dl is not None else None,
    )

    logger.info("INFO: Training will run for exactly {training_args.max_steps} steps (actual batch count * epochs)")

    # Custom Trainer Class
    class CustomTrainer(Trainer):
        def __init__(self, custom_train_dataloader=None, custom_eval_dataloader=None, expected_steps=None, **kwargs):
            super().__init__(**kwargs)
            self.custom_train_dataloader = custom_train_dataloader
            self.custom_eval_dataloader = custom_eval_dataloader
            self.expected_steps = expected_steps
        
        def get_train_dataloader(self):
            if self.custom_train_dataloader is not None:
                return self.custom_train_dataloader
            return super().get_train_dataloader()
            
        def get_eval_dataloader(self, eval_dataset=None):
            if self.custom_eval_dataloader is not None:
                return self.custom_eval_dataloader
            return super().get_eval_dataloader(eval_dataset)
        
        def _get_train_sampler(self):
            # Override to prevent Trainer from creating its own sampler
            return None

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,  # Set this properly from the start
        eval_dataset=eval_dataset if eval_dl is not None else None,  # Add evaluation dataset
        data_collator=data_collator,
        tokenizer=tokenizer,
        custom_train_dataloader=dl,
        custom_eval_dataloader=eval_dl,
        expected_steps=batch_count * args.epochs,
    )
    # 4. Training
    logger.info("INFO: Starting training...")
    logger.info("INFO: Debug Info:")
    logger.info(f"  - TrainingArguments.max_steps: {training_args.max_steps}")
    logger.info(f"  - Custom dataloader length: {len(dl)}")
    logger.info(f"  - Expected to run {batch_count * args.epochs} steps")

    trainer.train()

    logger.info("INFO: Training completed!")
    logger.info(f"  - Total steps trained: {trainer.state.global_step}")
    logger.info(f"  - Expected steps: {batch_count * args.epochs}")

    logger.info("INFO: Pushing model and tokenizer to HuggingFace...")
    if HF_TOKEN:
        trainer.model.push_to_hub(
            f"RikoteMaster/{args.output_name}", private=False, token=HF_TOKEN
        )
        tokenizer.push_to_hub(
            f"RikoteMaster/{args.output_name}", private=False, token=HF_TOKEN
        )
        logger.info("INFO: Model and tokenizer pushed to HuggingFace successfully")
    else:
        logger.warning("WARNING: Skipping model and tokenizer push to HuggingFace - no token provided")
    
    # Clear GPU cache after training
    torch.cuda.empty_cache()

    # 5. Save model in results_model directory
    logger.info("INFO: Saving model to results directory: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("INFO: Model saved to results directory successfully")

    # 6. Plotting and Saving Results
    logger.info("INFO: Generating and saving training loss plot...")
    plot_training_loss(trainer, FIGS_DIR)
    logger.info("INFO: Training loss plot saved successfully")

    # 7. Clean up temporary cache if used
    if str(HF_CACHE_DIR).startswith("/tmp/"):
        logger.info("INFO: Cleaning up temporary cache: {HF_CACHE_DIR}")
        try:
            shutil.rmtree(HF_CACHE_DIR)
            logger.info("INFO: Temporary cache cleaned up successfully")
        except Exception as e:
            logger.warning(f"WARNING: Could not clean up temporary cache: {e}")

    logger.info("INFO: Training pipeline completed successfully")


if __name__ == "__main__":
    main()
