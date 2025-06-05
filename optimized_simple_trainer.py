#!/usr/bin/env python3
"""
Usage:
    python simple_mcqa_training.py     --dataset_name "jonlecumberri/MNLP_M2_mcqa_dataset"     --output_dir "./models/qwen3-mcqa-optimized"     --max_samples 10000     --batch_size 4     --num_epochs 3     --push_to_hub     --hub_model_id "talphaidze/qwen3-0.6b-mcqa-optimized"
"""

import os
import json
import argparse
import logging
import torch
import numpy as np
from typing import Dict, List, Optional
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import torch.nn.functional as F

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleMCQACollator:
    """Simple data collator for MCQA training."""
    
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, examples):
        # Extract texts
        full_texts = [ex["full_text"] for ex in examples]
        prompts = [ex["prompt"] for ex in examples]
        
        # Tokenize everything
        batch = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Create labels
        labels = batch["input_ids"].clone()
        
        # Mask prompt tokens
        for i, prompt in enumerate(prompts):
            prompt_tokens = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )["input_ids"][0]
            
            prompt_len = len(prompt_tokens)
            labels[i, :prompt_len] = -100
            
            # Mask padding tokens
            labels[i][batch["input_ids"][i] == self.tokenizer.pad_token_id] = -100
        
        batch["labels"] = labels
        return batch

def format_mcqa_example(example):
    """Format a single MCQA example for training."""
    question = example.get("question", "").strip()
    
    # Handle different choice formats
    if "choices" in example:
        choices = example["choices"]
        if isinstance(choices, str):
            choices = [c.strip() for c in choices.split("|")]
    elif all(k in example for k in ["A", "B", "C", "D"]):
        choices = [example["A"], example["B"], example["C"], example["D"]]
    else:
        return None
    
    # Get correct answer
    if "answer_index" in example:
        correct_idx = int(example["answer_index"])
        correct_letter = chr(65 + correct_idx)
    elif "answer" in example:
        answer = example["answer"].strip().upper()
        if answer in ["A", "B", "C", "D"]:
            correct_letter = answer
        else:
            # Try to extract letter
            for letter in ["A", "B", "C", "D"]:
                if letter in answer:
                    correct_letter = letter
                    break
            else:
                return None
    else:
        return None
    
    # Validate
    if len(choices) != 4 or not question or correct_letter not in ["A", "B", "C", "D"]:
        return None
    
    # Format prompt
    prompt = f"Question: {question}\n\n"
    for i, choice in enumerate(choices):
        letter = chr(65 + i)
        choice_text = choice.strip()
        # Remove existing letter prefix
        if choice_text.startswith(f"{letter}.") or choice_text.startswith(f"{letter})"):
            choice_text = choice_text[2:].strip()
        prompt += f"{letter}. {choice_text}\n"
    prompt += "\nAnswer:"
    
    # Answer with space
    answer = f" {correct_letter}"
    full_text = prompt + answer
    
    return {
        "prompt": prompt,
        "answer": answer,
        "full_text": full_text,
        "correct_letter": correct_letter
    }

def process_dataset(dataset):
    """Process the entire dataset."""
    processed = []
    skipped = 0
    
    for example in dataset:
        formatted = format_mcqa_example(example)
        if formatted:
            processed.append(formatted)
        else:
            skipped += 1
    
    logger.info(f"Processed {len(processed)} examples, skipped {skipped}")
    return Dataset.from_list(processed)

def compute_accuracy(eval_pred):
    """Compute training accuracy."""
    predictions, labels = eval_pred
    
    # Only consider non-masked tokens
    mask = labels != -100
    
    # Get predicted tokens
    predicted_ids = np.argmax(predictions, axis=-1)
    
    # Calculate accuracy
    correct = (predicted_ids == labels) & mask
    accuracy = correct.sum() / mask.sum() if mask.sum() > 0 else 0
    
    return {"accuracy": float(accuracy)}

def evaluate_mcqa_accuracy(model, tokenizer, eval_dataset, num_samples=100):
    """Evaluate MCQA accuracy in the same way as official evaluation."""
    model.eval()
    correct = 0
    total = 0
    
    # Random sample
    indices = np.random.choice(len(eval_dataset), min(num_samples, len(eval_dataset)), replace=False)
    
    with torch.no_grad():
        for idx in indices:
            example = eval_dataset[int(idx)]
            prompt = example["prompt"]
            correct_letter = example["correct_letter"]
            
            # Get logits for each choice
            choice_logits = {}
            for letter in ["A", "B", "C", "D"]:
                full_seq = prompt + f" {letter}"
                tokens = tokenizer.encode(full_seq, return_tensors="pt")
                
                if tokens.shape[1] > 0:
                    outputs = model(tokens)
                    last_logits = outputs.logits[0, -1, :]
                    
                    # Get letter token ID
                    letter_tokens = tokenizer.encode(f" {letter}", add_special_tokens=False)
                    if letter_tokens:
                        letter_token_id = letter_tokens[-1]
                        choice_logits[letter] = last_logits[letter_token_id].item()
                    else:
                        choice_logits[letter] = float('-inf')
            
            # Predict
            if choice_logits:
                predicted = max(choice_logits, key=choice_logits.get)
                if predicted == correct_letter:
                    correct += 1
                total += 1
    
    accuracy = correct / total if total > 0 else 0
    logger.info(f"MCQA Accuracy: {correct}/{total} = {accuracy:.3f}")
    return accuracy

class MCQATrainer(Trainer):
    """Custom trainer with MCQA evaluation."""
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Standard evaluation
        result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Add MCQA accuracy
        mcqa_acc = evaluate_mcqa_accuracy(
            self.model, 
            self.tokenizer, 
            eval_dataset or self.eval_dataset,
            num_samples=50
        )
        result[f"{metric_key_prefix}_mcqa_accuracy"] = mcqa_acc
        
        return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B", help="Base model")
    parser.add_argument("--max_samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--eval_split", type=float, default=0.1, help="Eval split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to hub")
    parser.add_argument("--hub_model_id", type=str, help="Hub model ID")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load tokenizer and model
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Resize embeddings if needed
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    
    # Load and process dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    processed_dataset = process_dataset(dataset)
    
    # Split dataset
    split_dataset = processed_dataset.train_test_split(
        test_size=args.eval_split,
        seed=args.seed
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # Show sample
    logger.info("Sample example:")
    logger.info(f"Full text: {repr(train_dataset[0]['full_text'])}")
    
    # Data collator
    data_collator = SimpleMCQACollator(tokenizer, args.max_length)
    
    # Training arguments
    num_train_steps = len(train_dataset) * args.num_epochs // args.batch_size
    warmup_steps = int(num_train_steps * args.warmup_ratio)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        logging_steps=50,
        eval_steps=200,
        save_steps=200,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_mcqa_accuracy",
        greater_is_better=True,
        bf16=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        report_to=None,  # Disable wandb for simplicity
    )
    
    # Trainer
    trainer = MCQATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_accuracy,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save
    logger.info(f"Saving to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Final evaluation
    final_results = trainer.evaluate()
    logger.info(f"Final results: {final_results}")
    
    # Save results
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({
            "final_results": final_results,
            "args": vars(args),
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset)
        }, f, indent=2)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()