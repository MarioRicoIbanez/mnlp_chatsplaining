# Standard library imports
import logging
import os
from pathlib import Path
import torch

# Third-party imports
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, Trainer

# Local imports
from utils.dataset_utils import process_mcq_dataset, tokenize_func, SFTDataCollator, process_open_answer_dataset
from utils.model_utils import load_model
from utils.train_utils import plot_training_loss
from utils.eval_utils import evaluate_openqa

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
FIGS_DIR = SCRIPT_DIR / "figs"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting training process...")
    
    # Clear any existing GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 1. Model and Tokenizer Setup
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model(model_name="Qwen/Qwen3-0.6B", load_in_4bit=False)
    logger.info("Model and tokenizer loaded successfully")

    # 2. Dataset Preparation with streaming and strict memory limits
    logger.info("Loading dataset from HuggingFace with streaming...")
    streaming_dataset = load_dataset("RikoteMaster/OpenQA_merged", split="train", streaming=True)
    
    # Convert streaming dataset to regular dataset with first 10 examples
    logger.info("Converting first 10 examples to regular dataset...")
    train_dataset = []
    for i, example in enumerate(streaming_dataset):
        train_dataset.append(example)
    train_dataset = Dataset.from_list(train_dataset)
    logger.info(f"Processing {len(train_dataset)} examples")

    # Process with memory-efficient mapping
    logger.info("Processing dataset...")
    if "choices" in train_dataset.column_names:
        train_dataset = train_dataset.map(
            process_mcq_dataset, 
            fn_kwargs={"tokenizer": tokenizer},
            batch_size=1,  # Process one at a time
            remove_columns=train_dataset.column_names
        )
    else: 
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
        output_dir=str(SCRIPT_DIR / "qwen_sft_demo"),
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

    # Clear GPU cache after training
    torch.cuda.empty_cache()

    # 5. Save model
    logger.info("Saving model...")
    model.save_pretrained(SCRIPT_DIR / "qwen_sft_demo")
    tokenizer.save_pretrained(SCRIPT_DIR / "qwen_sft_demo")
    logger.info("Model saved successfully")

    # Push to HuggingFace
    logger.info("Pushing model to HuggingFace...")
    model.push_to_hub("RikoteMaster/Qwen3-0.6B-SFT-OpenQA", private=True, token="")
    logger.info("Model pushed to HuggingFace successfully")
    

    # 6. Plotting and Saving Results
    logger.info("Generating and saving training loss plot...")
    plot_training_loss(trainer, FIGS_DIR)
    logger.info("Training loss plot saved successfully")

    # 7. Evaluation on OpenCode and OpenMath samples
    logger.info("Loading test samples for evaluation...")
    
    # Load OpenCode samples
    opencode_dataset = load_dataset("RikoteMaster/OpenCodeTreated", split="train", streaming=True)
    logger.info("Loading OpenMath samples...")
    openmath_dataset = load_dataset("RikoteMaster/OpenMathTreated", split="train", streaming=True)
    
    # Get 3 samples from each dataset
    logger.info("Selecting samples for evaluation...")
    opencode_samples = []
    openmath_samples = []
    
    for i, example in enumerate(opencode_dataset):
        if i >= 3:  # Get 3 samples from OpenCode
            break
        opencode_samples.append(example)
    
    for i, example in enumerate(openmath_dataset):
        if i >= 3:  # Get 3 samples from OpenMath
            break
        openmath_samples.append(example)
    
    # Combine samples and add source information
    test_samples = []
    for sample in opencode_samples:
        sample['source'] = 'OpenCode'
        test_samples.append(sample)
    for sample in openmath_samples:
        sample['source'] = 'OpenMath'
        test_samples.append(sample)


    
    test_samples = Dataset.from_list(test_samples)
    
    # Evaluate model using the OpenQA evaluation function
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




