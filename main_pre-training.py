# Standard library imports
import logging
import os
from pathlib import Path
import torch
import argparse
import tempfile
import shutil
import pandas as pd
import matplotlib.pyplot as plt

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR  # Assuming the script is in the project root

# Set up project directories relative to script location
RESULTS_DIR = PROJECT_ROOT / "results_model"
FIGS_DIR = RESULTS_DIR / "figs"
HF_CACHE_DIR = RESULTS_DIR / "hf_cache"  # Persistent cache directory

# Create necessary directories
RESULTS_DIR.mkdir(exist_ok=True)
FIGS_DIR.mkdir(exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Script location: {SCRIPT_DIR}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Results directory: {RESULTS_DIR}")
print(f"HF Cache directory: {HF_CACHE_DIR}")

# Set HuggingFace cache directory to project location
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_DIR / "datasets")

# Third-party imports
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from trl import DataCollatorForCompletionOnlyLM

# Local imports
from utils.dataset_utils import (
    tokenize_func,
    SFTDataCollator,
    process_open_answer_dataset,
    process_mcq_dataset,
    MMLU_CHAT_TEMPLATE_JINJA,
    format_mmlu_prompt,
    format_mmlu_target,
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
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B", help="Model name")
    parser.add_argument("--output_name", type=str, default="Qwen3-0.6B-SFT-aux", help="Output dir/repo name")
    parser.add_argument("--dataset", type=str, default="RikoteMaster/OpenQA_merged", help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use")
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

    # 1. Model and Tokenizer Setup - ADAPTED FROM sanity_check.py
    logger.info("Loading model and tokenizer...")
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Log original chat template
    logger.info("Original chat template (from Qwen3 default):")
    logger.info(tokenizer.chat_template)
    
    # Set custom MMLU chat template
    logger.info("Setting custom MMLU chat template...")
    tokenizer.chat_template = MMLU_CHAT_TEMPLATE_JINJA
    logger.info("Custom MMLU chat template:")
    logger.info(tokenizer.chat_template)

    # Load model
    model, _ = load_model(model_name=args.model_name, load_in_4bit=False)
    logger.info("Model and tokenizer loaded successfully")

    # Determine response template for DataCollatorForCompletionOnlyLM
    logger.info("Determining response template for data collator...")
    try:
        # Try to get response template programmatically using our custom template
        dummy_messages = [
            {"role": "system", "content": "System test"},
            {"role": "user", "content": "User test"}
        ]
        prompt_with_gen = tokenizer.apply_chat_template(
            dummy_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_without_gen = tokenizer.apply_chat_template(
            dummy_messages,
            tokenize=False,
            add_generation_prompt=False
        )
        response_template_str = prompt_with_gen[len(prompt_without_gen):]
        
        if not response_template_str.strip():
            raise ValueError("Empty response template")
            
        # Verify the response template matches what we expect
        logger.info(f"Response template string: '{response_template_str}'")
        logger.info(f"Response template tokens: {tokenizer.encode(response_template_str, add_special_tokens=False)}")
            
    except Exception as e:
        logger.warning(f"Could not determine response template programmatically: {e}")
        # Fallback to known template
        response_template_str = "<|im_start|>assistant\n"
        logger.warning(f"Using fallback response template: '{response_template_str}'")
    
    logger.info(f"Using response template: '{response_template_str}'")
    response_template_ids = tokenizer.encode(response_template_str, add_special_tokens=False)
    
    if not response_template_ids:
        raise ValueError(f"Response template '{response_template_str}' encoded to empty IDs")

    # 2. Dataset Preparation - UNIFIED WITH main_lora.py
    logger.info("Loading dataset...")
    dataset = load_dataset(args.dataset)["train"]  # Use train split instead of test
    dataset = dataset.shuffle(seed=42)

    # Limit number of samples if specified
    if args.num_samples is not None:
        logger.info(f"Limiting dataset to {args.num_samples} samples")
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]

    logger.info(f"Dataset size: {len(dataset)} samples")
    logger.info(f"Training set size: {len(train_dataset)} samples")
    logger.info(f"Validation set size: {len(val_dataset)} samples")

    # Process dataset using MMLU formatting
    logger.info("Processing dataset with MMLU format...")
    
    # Formatting function that uses our custom MMLU template
    def formatting_func(example):
        """Format example using our custom MMLU template."""
        # Format prompt using our custom template
        prompt = format_mmlu_prompt(example["question"], example["choices"], tokenizer)
        
        # Format target exactly as lighteval expects for loglikelihood
        target = format_mmlu_target(example["answer_index"])
        
        # Combine prompt and target for training
        full_text = prompt + target
        
        # Calculate prompt length for masking (like other dataset functions)
        prompt_len = len(tokenizer(prompt)["input_ids"])
        
        return {
            "text": full_text,
            "prompt": prompt,
            "prompt_len": prompt_len
        }

    is_mcqa = "choices" in train_dataset.column_names
    if is_mcqa:
        logger.info("Processing MCQA dataset with MMLU format...")
        train_dataset = train_dataset.map(formatting_func)
        val_dataset = val_dataset.map(formatting_func)
    else:
        logger.info("Processing OpenAnswer dataset...")
        train_dataset = train_dataset.map(process_open_answer_dataset)
        val_dataset = val_dataset.map(process_open_answer_dataset)

    # Print first 5 samples showing just the text field
    logger.info("\nFirst 5 text samples:")
    for i in range(5):
        logger.info(f"\n{'='*80}")
        logger.info(f"SAMPLE {i+1}")
        logger.info(f"{'='*80}")
        text = train_dataset["text"][i]
        logger.info(text)
        logger.info(f"{'-'*80}")

    logger.info("Tokenizing dataset...")
    tokenized_train_dataset = train_dataset.map(
        tokenize_func,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=train_dataset.column_names,
    )

    # Process validation dataset if available
    tokenized_val_dataset = None
    if val_dataset is not None:
        tokenized_val_dataset = val_dataset.map(
            tokenize_func,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=val_dataset.column_names,
        )

    # Add sequence lengths to dataset
    logger.info("Computing sequence lengths...")
    tokenized_train_dataset = tokenized_train_dataset.map(
        lambda ex: {"length": len(ex["input_ids"])},
        num_proc=4,
    )

    if tokenized_val_dataset is not None:
        tokenized_val_dataset = tokenized_val_dataset.map(
            lambda ex: {"length": len(ex["input_ids"])},
            num_proc=4,
        )

    # Sort dataset by length
    logger.info("Sorting dataset by sequence length...")
    tokenized_train_dataset = tokenized_train_dataset.sort("length", reverse=True)

    # 3. Training Setup
    # Use DataCollatorForCompletionOnlyLM instead of SFTDataCollator
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer
    )

    # Set up token-budget batching
    max_tok_per_gpu = 8_000  # fits comfortably in 24 GB with bf16
    sampler = SmartPaddingTokenBatchSampler(
        tokenized_train_dataset["length"], max_tok_per_gpu
    )

    logger.info("INFO: Smart Batching Info:")
    logger.info(f"  - Max tokens per GPU: {max_tok_per_gpu}")
    logger.info(f"  - Dataset length: {len(tokenized_train_dataset)}")

    # Debug: Count total batches that will be created - do this properly
    batch_count = 0
    sample_batch_sizes = []
    temp_sampler = SmartPaddingTokenBatchSampler(
        tokenized_train_dataset["length"], max_tok_per_gpu
    )

    # Count ALL batches that will actually be generated
    logger.info("INFO: Counting actual batches that will be generated...")
    for batch_indices in temp_sampler:
        batch_count += 1
        sample_batch_sizes.append(len(batch_indices))
        if batch_count <= 5:  # Show first 5 batch sizes
            max_len = max([tokenized_train_dataset[i]["length"] for i in batch_indices])
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
        tokenized_train_dataset["length"], max_tok_per_gpu
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
        tokenized_train_dataset,
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
        if tokenized_val_dataset is not None:
            logger.info("Setting up evaluation dataloader...")
            # Create evaluation sampler
            eval_sampler = SmartPaddingTokenBatchSampler(
                tokenized_val_dataset["length"], max_tok_per_gpu
            )
            
            # Create evaluation dataloader
            eval_dl = CustomDataLoader(
                len(eval_sampler),
                tokenized_val_dataset,
                batch_sampler=eval_sampler,
                collate_fn=data_collator,
                num_workers=0,
                pin_memory=False,
            )
            
            logger.info(f"Successfully set up evaluation dataloader with {len(tokenized_val_dataset)} examples")
        else:
            logger.info("No validation dataset available")
        
    except Exception as e:
        logger.warning(f"Could not set up evaluation dataloader: {str(e)}")
        logger.warning("Training will proceed without evaluation")
        eval_dl = None

    # Calculate expected steps for smart batching
    logger.info("INFO: Dataset info:")
    logger.info(f"  - Training samples: {len(tokenized_train_dataset)}")
    logger.info(f"  - Smart batch sampler will create dynamic batches")
    logger.info(f"  - Number of epochs: {args.epochs}")
    logger.info(f"  - Total steps per epoch: {batch_count}")
    logger.info(f"  - Total steps across all epochs: {batch_count * args.epochs}")

    # === TRAINING ARGS ===
    logger.info("Setting up training...")

    # Create output directory based on output name (same logic as main_lora.py)
    output_dir = RESULTS_DIR / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR = str(output_dir)

    logger.info(f"INFO: Output directory: {OUTPUT_DIR}")
    logger.info(f"INFO: HuggingFace repo: RikoteMaster/{args.output_name}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        logging_steps=5,
        save_steps=1000,
        eval_steps=10,  # Evaluate less frequently 
        eval_strategy="steps" if eval_dl is not None else "no",  # Enable evaluation
        report_to="wandb",
        bf16=True,
        disable_tqdm=False,
        remove_unused_columns=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        warmup_steps=1,
        learning_rate=6e-4,  # Match sanity_check.py
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        dataloader_pin_memory=False,
        label_names=["labels"],
        prediction_loss_only=False,
        load_best_model_at_end=True if eval_dl is not None else False,
        metric_for_best_model="eval_loss" if eval_dl is not None else None,
    )

    logger.info(f"INFO: Training will run for exactly {training_args.max_steps} steps (actual batch count * {args.epochs} epochs)")

    # === CUSTOM TRAINER CLASS ===
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

    # === TRAINER ===
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,  # Set this properly from the start
        eval_dataset=tokenized_val_dataset if eval_dl is not None else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        custom_train_dataloader=dl,
        custom_eval_dataloader=eval_dl,
        expected_steps=batch_count * args.epochs,
    )

    # === TRAIN ===
    logger.info("Starting training...")
    logger.info(f"INFO: Debug Info:")
    logger.info(f"  - TrainingArguments.max_steps: {training_args.max_steps}")
    logger.info(f"  - Custom dataloader length: {len(dl)}")
    logger.info(f"  - Expected to run {batch_count * args.epochs} steps")

    trainer.train()

    logger.info(f"INFO: Training completed!")
    logger.info(f"  - Total steps trained: {trainer.state.global_step}")
    logger.info(f"  - Expected steps: {batch_count * args.epochs}")

    # === SAVE LOGS ===
    logger.info("Saving training log and plot...")

    # Load training logs
    log_df = pd.DataFrame(trainer.state.log_history)
    log_df.to_csv(os.path.join(OUTPUT_DIR, "training_log.csv"), index=False)

    # Plot
    plt.figure(figsize=(8, 5))
    plotted = False
    loss_columns = []

    logger.info(f"INFO: Available log columns: {list(log_df.columns)}")

    if "loss" in log_df.columns:
        # Filter out NaN values for training loss
        train_loss_data = log_df.dropna(subset=["loss"])
        if not train_loss_data.empty:
            plt.plot(train_loss_data["step"], train_loss_data["loss"], label="Training Loss", marker='o', markersize=2)
            loss_columns.append("loss")
            plotted = True

    if "eval_loss" in log_df.columns:
        # Filter out NaN values for eval loss
        eval_loss_data = log_df.dropna(subset=["eval_loss"])
        if not eval_loss_data.empty:
            plt.plot(eval_loss_data["step"], eval_loss_data["eval_loss"], label="Eval Loss", marker='s', markersize=3)
            loss_columns.append("eval_loss")
            plotted = True

    if plotted:
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training and Evaluation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Safe autoscaling - only use columns that exist and have data
        if loss_columns:
            available_data = log_df[loss_columns].dropna()
            if not available_data.empty:
                min_loss = available_data.min().min()
                max_loss = available_data.max().max()
                plt.ylim(bottom=min_loss * 0.95, top=max_loss * 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "loss_plot_scaled.png"), dpi=150, bbox_inches='tight')
        logger.info("INFO: Loss plot saved successfully!")
    else:
        logger.warning("⚠ No loss data found in logs.")

    # === SAVE MODEL ===
    logger.info("Saving model...")
    model.save_pretrained(OUTPUT_DIR)
    
    # Save tokenizer with custom template
    logger.info("Saving tokenizer with custom MMLU template...")
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Verify saved chat template
    logger.info("Verifying saved chat template...")
    saved_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
    logger.info("Chat template from saved tokenizer (should be custom MMLU):")
    logger.info(saved_tokenizer.chat_template)
    
    logger.info("INFO: Model saved successfully!")

    # === PUSH TO HUB ===
    if HF_TOKEN:
        logger.info("INFO: Pushing model to HuggingFace...")
        try:
            model.push_to_hub(
                f"RikoteMaster/{args.output_name}",
                token=HF_TOKEN,
                private=False
            )
            tokenizer.push_to_hub(
                f"RikoteMaster/{args.output_name}",
                token=HF_TOKEN,
                private=False
            )
            logger.info("INFO: Model pushed to HuggingFace successfully!")
        except Exception as e:
            logger.error(f"ERROR: Failed to push model to HuggingFace: {e}")
    else:
        logger.warning("WARNING: No HF_TOKEN found, skipping HuggingFace push")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("INFO: Training pipeline completed successfully")


if __name__ == "__main__":
    main()
