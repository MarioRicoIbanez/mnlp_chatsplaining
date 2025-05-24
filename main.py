# Standard library imports
import logging
import os
from pathlib import Path
import torch

# Third-party imports
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

# Local imports
from utils.dataset_utils import process_mcq_dataset, tokenize_func, SFTDataCollator
from utils.model_utils import load_model
from utils.train_utils import plot_training_loss

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
    
    # 1. Model and Tokenizer Setup
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model(model_name="Qwen/Qwen3-0.6B", load_in_4bit=False)
    logger.info("Model and tokenizer loaded successfully")

    # 2. Dataset Preparation
    logger.info("Loading dataset from HuggingFace...")
    train_dataset = load_dataset("RikoteMaster/sciq_treated_epfl_mcqa", split="train")
    logger.info(f"Dataset loaded with {len(train_dataset)} examples")
    
    if "choices" in train_dataset.column_names:
        # logger.info("Selecting first 100 examples for DEBUGGING...")
        # train_dataset = train_dataset.select(range(100))
        train_dataset = train_dataset.map(process_mcq_dataset, fn_kwargs={"tokenizer": tokenizer})

    tokenized_dataset = train_dataset.map(
        tokenize_func, 
        fn_kwargs={"tokenizer": tokenizer}, 
        remove_columns=train_dataset.column_names
    )

    # 3. Training Setup
    data_collator = SFTDataCollator(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(SCRIPT_DIR / "qwen_sft_demo"),
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=6,
        gradient_accumulation_steps=1,
        logging_steps=20,
        save_steps=0,
        report_to=[],
        bf16=True,
        disable_tqdm=False,
        remove_unused_columns=False,
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

    # 5. Plotting and Saving Results
    logger.info("Generating and saving training loss plot...")
    plot_training_loss(trainer, FIGS_DIR)
    logger.info("Training loss plot saved successfully")

    # 6. Evaluation on test dataset
    logger.info("Loading test dataset for evaluation...")
    test_dataset = load_dataset("RikoteMaster/sciq_treated_epfl_mcqa", split="test")
    logger.info(f"Test dataset loaded with {len(test_dataset)} examples")
    
    # Process first 10 test samples for evaluation
    logger.info("Evaluating model on first 10 test samples...")
    test_samples = test_dataset.select(range(10))
    processed_test_samples = test_samples.map(process_mcq_dataset, fn_kwargs={"tokenizer": tokenizer})
    
    print("\n" + "="*80)
    print("EVALUATION: Model predictions vs Ground Truth")
    print("="*80)
    
    correct_predictions = 0
    total_predictions = 0
    
    for i, sample in enumerate(processed_test_samples):
        print(f"\n--- SAMPLE {i+1} ---")
        print(f"Question: {sample.get('question', 'N/A')}")
        print(f"Choices: {sample.get('choices', 'N/A')}")
        print(f"Ground Truth Answer: {sample.get('answer_text', 'N/A')}")
        
        # Use only the prompt (without the answer) for generation
        prompt_only = sample['prompt']
        
        # Tokenize and generate
        inputs = tokenizer(prompt_only, return_tensors="pt").to(model.device)
        
        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the generated part (excluding the input prompt)
        generated_text = tokenizer.decode(
            output_ids[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        print(f"\nModel Generated:")
        print(f"'{generated_text}'")
        
        # Simple evaluation: check if the correct answer letter appears in generated text
        ground_truth = sample.get('answer_text', '')
        choices = sample.get('choices', [])
        
        if isinstance(choices, list) and ground_truth in choices:
            correct_answer_index = choices.index(ground_truth)
            correct_letter = chr(65 + correct_answer_index)  # A, B, C, D
            
            # Check if the correct letter appears in the generated response
            is_correct = correct_letter in generated_text.upper()
            correct_predictions += int(is_correct)
            total_predictions += 1
            
            print(f"Expected Answer: {correct_letter}. {ground_truth}")
            print(f"Prediction: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        else:
            print(f"Could not evaluate this sample")
        
        print("-" * 60)
    
    # Print overall accuracy
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions * 100
        print(f"\nOVERALL ACCURACY: {correct_predictions}/{total_predictions} = {accuracy:.1f}%")
    else:
        print(f"\nCould not compute accuracy")

if __name__ == "__main__":
    main()




