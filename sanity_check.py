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
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from huggingface_hub import HfApi
import json
import sys

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR.parent))

# Import MMLU formatting functions and constants
from utils.dataset_utils import (
    MMLU_CHAT_TEMPLATE_JINJA,
    MMLU_SYSTEM_MESSAGE_CONTENT,
    format_mmlu_prompt,
    format_mmlu_target,
)

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
EPOCHS = 5
HF_TOKEN = ""
REPO_NAME = "RikoteMaster/sanity_check_model"

# Define LETTER_INDICES for consistent choice formatting
LETTER_INDICES = "ABCD"

# Content for the system message in our MMLU template
MMLU_SYSTEM_MESSAGE_CONTENT = "The following are multiple choice questions (with answers) about knowledge and skills in advanced master-level STEM courses.\nJust answer with A, B, C, or D."

# Custom MMLU chat template
# This template assumes messages will have:
# 1. An optional "system" message (we'll use MMLU_SYSTEM_MESSAGE_CONTENT)
# 2. A "user" message with the formatted question and "Answer:"
# And will add the assistant prompt
MY_MMLU_CHAT_TEMPLATE_JINJA = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
        "{% elif message['role'] == 'user' %}"
            "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
        "{% elif message['role'] == 'assistant' %}"  # For completion during training
            "{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"  # SFTTrainer/lighteval will use this for assistant turn start
        "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

def format_mmlu_prompt(question: str, choices: list[str], tokenizer) -> str:
    """Format prompt using our custom MMLU chat template.
    
    This matches the format used in lighteval's MCQATask with loglikelihood_acc_norm,
    using our custom MMLU chat template instead of Qwen3's default.
    """
    # Format choices as A. choice1\nB. choice2\n...
    choices_str = "\n".join(f"{LETTER_INDICES[i]}. {choice}" for i, choice in enumerate(choices))
    
    # Content for the user message
    # Note: We don't include the system message content here as it's handled by the template
    user_content = f"{question}\n{choices_str}\nAnswer:"
    
    # Create messages list for our custom template
    messages = [
        {"role": "system", "content": MMLU_SYSTEM_MESSAGE_CONTENT},
        {"role": "user", "content": user_content.strip()}
    ]
    
    # Apply our custom MMLU chat template
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def format_mmlu_target(answer_index: int) -> str:
    """Format target exactly as lighteval expects for loglikelihood calculation.
    
    This matches how lighteval constructs the continuation for loglikelihood calculation.
    The space before the letter is crucial for tokenization alignment.
    """
    return f" {chr(65 + answer_index)}"  # Note the space before the letter

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
    
    # Log original chat template
    logger.info("Original chat template (from Qwen3 default):")
    logger.info(tokenizer.chat_template)
    
    # Set our custom MMLU chat template
    logger.info("Setting custom MMLU chat template...")
    tokenizer.chat_template = MMLU_CHAT_TEMPLATE_JINJA
    logger.info("Custom MMLU chat template:")
    logger.info(tokenizer.chat_template)
    
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
    
    # Create data collator for completion-only loss
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer
    )
    
    # Formatting function that uses our custom MMLU template
    def formatting_func(example):
        """Format example using our custom MMLU template.
        
        This ensures the model learns to assign high probability to the correct
        answer letter in the exact format used by loglikelihood_acc_norm.
        """
        # Format prompt using our custom template
        prompt = format_mmlu_prompt(example["question"], example["choices"], tokenizer)
        
        # Format target exactly as lighteval expects for loglikelihood
        target = format_mmlu_target(example["answer_index"])
        
        # Combine prompt and target for training
        full_text = prompt + target
        
        return full_text
        """Format example using our custom MMLU template.
        
        This ensures the model learns to assign high probability to the correct
        answer letter in the exact format used by loglikelihood_acc_norm.
        """
        # Format prompt using our custom template
        prompt = format_mmlu_prompt(example["question"], example["choices"], tokenizer)
        
        # Format target exactly as lighteval expects for loglikelihood
        target = format_mmlu_target(example["answer_index"])
        
        # Combine prompt and target for training
        full_text = prompt + target
        
        return full_text
    
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
        learning_rate=6e-4,
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
    
    # Create trainer with custom MMLU formatting
    logger.info("Creating SFT trainer with custom MMLU formatting...")
    # Create trainer with custom MMLU formatting
    logger.info("Creating SFT trainer with custom MMLU formatting...")
    trainer = SFTTrainer(
        model=MODEL_NAME,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        formatting_func=formatting_func,
        data_collator=data_collator,  # Use completion-only data collator
        data_collator=data_collator,  # Use completion-only data collator
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model locally first
    logger.info("Saving model locally...")
    trainer.save_model()
    
    # Save tokenizer with our custom template
    logger.info("Saving tokenizer with custom MMLU template...")
    # Save tokenizer with our custom template
    logger.info("Saving tokenizer with custom MMLU template...")
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    
    # Verify saved chat template
    logger.info("Verifying saved chat template...")
    saved_tokenizer = AutoTokenizer.from_pretrained(str(OUTPUT_DIR), trust_remote_code=True)
    logger.info("Chat template from saved tokenizer (should be custom MMLU):")
    logger.info(saved_tokenizer.chat_template)
    
    # Verify saved chat template
    logger.info("Verifying saved chat template...")
    saved_tokenizer = AutoTokenizer.from_pretrained(str(OUTPUT_DIR), trust_remote_code=True)
    logger.info("Chat template from saved tokenizer (should be custom MMLU):")
    logger.info(saved_tokenizer.chat_template)
    
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

This is a LoRA (Low-Rank Adaptation) fine-tuned model for multiple choice question answering tasks, trained to match lighteval's MMLU format exactly.
This is a LoRA (Low-Rank Adaptation) fine-tuned model for multiple choice question answering tasks, trained to match lighteval's MMLU format exactly.

### Training Details

- **Base Model**: {MODEL_NAME}
- **Fine-tuning Method**: LoRA (r=16, alpha=32)
- **Task**: Multiple Choice Question Answering (MCQA)
- **Training epochs**: {EPOCHS}
- **LoRA Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Training Format**: Exact MMLU format for loglikelihood_acc_norm evaluation
- **Chat Template**: Uses custom MMLU chat template for consistent formatting
- **Training Format**: Exact MMLU format for loglikelihood_acc_norm evaluation
- **Chat Template**: Uses custom MMLU chat template for consistent formatting

## Usage

### Using the model with PEFT

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Load model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained("{REPO_NAME}")
tokenizer = AutoTokenizer.from_pretrained("{REPO_NAME}")

# Example usage with exact MMLU format
# Example usage with exact MMLU format
question = "What is 2+2?"
choices = ["3", "4", "5", "6"]
answer_index = 1  # Index of correct answer (B)

# Format choices as MMLU does
choices_str = "\\n".join(f"{{chr(65+i)}}. {{choice}}" for i, choice in enumerate(choices))

# Create messages in MMLU format
messages = [
# Format choices as MMLU does
choices_str = "\\n".join(f"{{chr(65+i)}}. {{choice}}" for i, choice in enumerate(choices))

# Create messages in MMLU format
messages = [
    {{
        "role": "system",
        "content": "The following are multiple choice questions (with answers) about knowledge and skills in advanced master-level STEM courses.\\nJust answer with A, B, C, or D."
    }},
    {{
        "role": "user",
        "content": f"{{question}}\\n{{choices_str}}\\nAnswer:"
    }}
]

# Apply chat template exactly as MMLU does
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
        "role": "system",
        "content": "The following are multiple choice questions (with answers) about knowledge and skills in advanced master-level STEM courses.\\nJust answer with A, B, C, or D."
    }},
    {{
        "role": "user",
        "content": f"{{question}}\\n{{choices_str}}\\nAnswer:"
    }}
]

# Apply chat template exactly as MMLU does
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Format target exactly as MMLU expects
target = f" {{chr(65 + answer_index)}}"  # Note the space before the letter

# Combine for generation
full_text = prompt + target

# Format target exactly as MMLU expects
target = f" {{chr(65 + answer_index)}}"  # Note the space before the letter

# Combine for generation
full_text = prompt + target

# Generate
inputs = tokenizer(full_text, return_tensors="pt")
inputs = tokenizer(full_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.1)

# Extract answer
answer = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
print(f"Answer: {{answer}}")
"""
if __name__ == "__main__":
    main()

