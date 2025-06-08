import os
import torch
import logging
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
from huggingface_hub import HfApi
import json
import sys

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR.parent))
from utils.dataset_utils import process_mcq_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Directories
OUTPUT_DIR = SCRIPT_DIR / "safety_model_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B"  # Using available model
DATASET_NAME = "RikoteMaster/sanity_check_dataset"
MAX_LENGTH = 2048
EPOCHS = 10
HF_TOKEN = ""
REPO_NAME = "RikoteMaster/sanity_check_model"

def main():
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)
    
    # Handle different dataset structures
    if "train" in dataset:
        full_dataset = dataset["train"]
    else:
        # If no train split, use the first available split
        full_dataset = dataset[list(dataset.keys())[0]]
    
    # Create train/eval split
    logger.info("Creating train/eval split...")
    split = full_dataset.train_test_split(test_size=0.001, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    
    logger.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Formatting function using unified dataset_utils
    def formatting_func(example):
        """Format the example using unified dataset_utils"""
        processed = process_mcq_dataset(
            {
                "question": example["question"],
                "choices": example["choices"],
                "answer_index": example["answer_index"],
                "answer_text": example["answer_text"],
            },
            tokenizer=tokenizer,
            use_mmlu=True  # Use MMLU-style formatting
        )
        return processed["text"]
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    # SFT config
    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        warmup_steps=5,
        logging_steps=10,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        learning_rate=2e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to="none",
        max_length=MAX_LENGTH,
        packing=False,
        dataset_text_field="text",
        model_init_kwargs={
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        },
    )
    
    # Create trainer with unified formatting
    logger.info("Creating SFT trainer with unified formatting...")
    trainer = SFTTrainer(
        model=MODEL_NAME,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        formatting_func=formatting_func,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model locally first
    logger.info("Saving model locally...")
    trainer.save_model()
    
    # Save tokenizer
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    
    # Fix the adapter_config.json to include base model info
    adapter_config_path = OUTPUT_DIR / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        # Update base model path
        adapter_config["base_model_name_or_path"] = MODEL_NAME
        
        with open(adapter_config_path, 'w') as f:
            json.dump(adapter_config, f, indent=2)
    
    # Load the base model config and save it
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    # Save the base model config
    base_config = base_model.config
    base_config.save_pretrained(str(OUTPUT_DIR))
    
    # Clean up base model
    del base_model
    torch.cuda.empty_cache()
    
    # Push to hub
    logger.info("Pushing to HuggingFace Hub...")
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=REPO_NAME, token=HF_TOKEN, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create repo: {e}")
    
    # Upload all files
    api.upload_folder(
        folder_path=str(OUTPUT_DIR),
        repo_id=REPO_NAME,
        token=HF_TOKEN,
    )
    
    # Create a comprehensive README
    readme_content = f"""---
base_model: {MODEL_NAME}
library_name: peft
license: apache-2.0
tags:
- generated_from_trainer
- trl
- sft
- multiple-choice
- question-answering
datasets:
- {DATASET_NAME}
model-index:
- name: {REPO_NAME.split('/')[-1]}
  results: []
---

# Sanity Check Model

This model is a fine-tuned version of [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) on the [{DATASET_NAME}](https://huggingface.co/datasets/{DATASET_NAME}) dataset.

## Model Description

This is a LoRA (Low-Rank Adaptation) fine-tuned model for multiple choice question answering tasks, trained with MMLU-style formatting.

### Training Details

- **Base Model**: {MODEL_NAME}
- **Fine-tuning Method**: LoRA (r=16, alpha=32)
- **Task**: Multiple Choice Question Answering (MCQA)
- **Training epochs**: {EPOCHS}
- **LoRA Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Training Format**: MMLU-style formatting with letter-only answers

## Usage

### Using the model with PEFT

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from utils.dataset_utils import process_mcq_dataset

# Load model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained("{REPO_NAME}")
tokenizer = AutoTokenizer.from_pretrained("{REPO_NAME}")

# Example usage
question = "What is 2+2?"
choices = ["3", "4", "5", "6"]
answer_index = 1  # Index of correct answer (B)

# Process using unified format
processed = process_mcq_dataset(
    {{
        "question": question,
        "choices": choices,
        "answer_index": answer_index,
        "answer_text": choices[answer_index],
    }},
    tokenizer=tokenizer,
    use_mmlu=True
)

# Generate
inputs = tokenizer(processed["text"], return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.1)

# Extract answer
answer = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
print(f"Answer: {{answer}}")
"""
if __name__ == "__main__":
    main()