import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from dotenv import load_dotenv
import shutil
import logging
from torch.utils.data import IterableDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
RESULTS_DIR = SCRIPT_DIR / "results_model"
HF_CACHE_DIR = RESULTS_DIR / "hf_cache"  # Persistent cache directory

# Create necessary directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set HuggingFace cache directory to project location
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_DIR / "datasets")

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import DataCollatorForCompletionOnlyLM
from utils.dataset_utils import (
    process_mcq_dataset,
    SFTDataCollator,
    MMLU_CHAT_TEMPLATE_JINJA,
    format_mmlu_prompt,
    format_mmlu_target,
)  # <-- import your ChatML logic here
from transformers.utils import CONFIG_NAME

from utils.dataset_utils import tokenize_func
from utils.batching import SmartPaddingTokenBatchSampler
from torch.utils.data import DataLoader

import wandb

# Load environment variables from .env file
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a language model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B", help="Model name")
    parser.add_argument("--output_name", type=str, default="Qwen3-0.6B-SFT-aux", help="Output dir/repo name")
    parser.add_argument("--dataset", type=str, default="RikoteMaster/OpenQA_merged", help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use")
    return parser.parse_args()

# Parse command line arguments
args = parse_args()

# Get HuggingFace token from environment
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.warning("HF_TOKEN not found in environment variables. Model pushing will be skipped.")

wandb.init(project="chatsplaining")

# === CONFIG ===
MODEL_NAME = args.model_name  # Use model name from arguments
TOKENIZER_NAME = args.model_name  # Use same model for tokenizer
DATASET_NAME = args.dataset  # Use dataset from arguments
HF_REPO_ID = f"RikoteMaster/{args.output_name}"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === LOAD DATASET ===
logger.info("Loading dataset...")
dataset = load_dataset(DATASET_NAME)["train"]  # Use train split instead of test
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

# === TOKENIZER AND MODEL ===
logger.info("Loading model and tokenizer...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

#  Use the tokenizer from the base model
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Log original chat template
logger.info("Original chat template (from Qwen3 default):")
logger.info(tokenizer.chat_template)

# Set custom MMLU chat template
logger.info("Setting custom MMLU chat template...")
tokenizer.chat_template = MMLU_CHAT_TEMPLATE_JINJA
logger.info("Custom MMLU chat template:")
logger.info(tokenizer.chat_template)

#  Load fine-tuned model weights
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)

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

# === LoRA ===
logger.info("Applying LoRA...")
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)
model = get_peft_model(model, peft_config)

# === CRITICAL: Fix gradient issues for Qwen3 models ===
logger.info("Fixing gradient requirements for Qwen3 + LoRA...")

# Since Qwen3 doesn't have enable_input_require_grads(), use alternative approach
def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

# Register forward hook on input embeddings to ensure gradients flow properly
model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

# Ensure model is in training mode and print trainable parameters
model.train()

# Additional fix: Ensure LoRA parameters are properly configured
for name, param in model.named_parameters():
    if param.requires_grad:
        # Ensure gradients can flow through LoRA parameters
        if "lora" in name.lower():
            param.data = param.data.float()  # Convert LoRA params to float32
            
logger.info(f"Model is in training mode: {model.training}")
model.print_trainable_parameters()

# CRITICAL: Enable gradient computation for all parameters that require it
for name, param in model.named_parameters():
    if param.requires_grad:
        param.retain_grad()  # Ensure gradients are retained

# === PREPROCESS ===
logger.info("Preprocessing...")

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
    logger.info("Processing dataset with standard format...")
    train_dataset = train_dataset.map(process_mcq_dataset)
    val_dataset = val_dataset.map(process_mcq_dataset)

tokenized_train_dataset = train_dataset.map(
    tokenize_func,
    fn_kwargs={"tokenizer": tokenizer},  # Shorter sequences
    remove_columns=train_dataset.column_names,
)

tokenized_val_dataset = val_dataset.map(
    tokenize_func,
    fn_kwargs={"tokenizer": tokenizer},  # Shorter sequences
    remove_columns=val_dataset.column_names,
)

tokenized_train_dataset = tokenized_train_dataset.map(
    lambda ex: {"length": len(ex["input_ids"])},
    num_proc=4,  # speeds it up
)
tokenized_eval_dataset = tokenized_val_dataset.map(
    lambda ex: {"length": len(ex["input_ids"])},
    num_proc=4,  # speeds it up
)

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

logger.info(f"INFO: Smart Batching Info:")
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

# Create output directory based on output name (same logic as main_pre-training.py)
output_dir = RESULTS_DIR / args.output_name
output_dir.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = str(output_dir)

logger.info(f"INFO: Output directory: {OUTPUT_DIR}")
logger.info(f"INFO: HuggingFace repo: {HF_REPO_ID}")

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

# === SAVE MODEL LOCALLY ===
logger.info("Saving LoRA adapters locally...")

# Save LoRA adapters (for potential future use)
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)

# Save tokenizer with custom template
logger.info("Saving tokenizer with custom MMLU template...")
tokenizer.save_pretrained(OUTPUT_DIR)

# Verify saved chat template
logger.info("Verifying saved chat template...")
saved_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
logger.info("Chat template from saved tokenizer (should be custom MMLU):")
logger.info(saved_tokenizer.chat_template)

logger.info("INFO: Saving LoRA adapters in separate directory...")
# Save LoRA adapters in a separate directory for backup
lora_backup_dir = OUTPUT_DIR + "_lora_backup"
os.makedirs(lora_backup_dir, exist_ok=True)
model.save_pretrained(lora_backup_dir, safe_serialization=True)
tokenizer.save_pretrained(lora_backup_dir)

# ✅ Save config.json manually (from base model) - CRITICAL for HuggingFace
logger.info("INFO: Preparing model configuration for HuggingFace...")

# Get the base model config and save it
base_model_config = model.base_model.config

# 🔧 Ensure model_type is explicitly set (critical for HuggingFace recognition)
if not hasattr(base_model_config, 'model_type') or base_model_config.model_type is None:
    base_model_config.model_type = "qwen3"
    logger.info("INFO: Set model_type to 'qwen3' in base config")

config_path = os.path.join(OUTPUT_DIR, CONFIG_NAME)
base_model_config.to_json_file(config_path)
logger.info(f"INFO: config.json saved to {config_path}")

# Also create a merged model config directory for HuggingFace upload
merged_config_dir = OUTPUT_DIR + "_config"
os.makedirs(merged_config_dir, exist_ok=True)

# Save all necessary files for HuggingFace
base_model_config.to_json_file(os.path.join(merged_config_dir, CONFIG_NAME))
tokenizer.save_pretrained(merged_config_dir)

# Copy LoRA adapter files to the config directory
adapter_files = ["adapter_model.safetensors", "adapter_config.json"]
for file in adapter_files:
    src_path = os.path.join(OUTPUT_DIR, file)
    dst_path = os.path.join(merged_config_dir, file)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        logger.info(f"INFO: Copied {file} to upload directory")

# 🔧 CRITICAL FIX: Update adapter_config.json with correct base model path
logger.info("INFO: Fixing adapter_config.json to point to correct base model...")
adapter_config_path = os.path.join(OUTPUT_DIR, "adapter_config.json")
if os.path.exists(adapter_config_path):
    import json
    
    # Read current adapter config
    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)
    
    # Update base_model_name_or_path to point to your pre-trained model
    adapter_config["base_model_name_or_path"] = MODEL_NAME  # This is your pre-trained model
    logger.info(f"INFO: Updated base_model_name_or_path to: {MODEL_NAME}")
    
    # Ensure model_type is set correctly
    if "model_type" not in base_model_config.to_dict():
        base_model_config.model_type = "qwen3"  # Explicit model type
        logger.info("INFO: Added model_type: qwen3 to base config")
    
    # Save updated adapter config
    with open(adapter_config_path, 'w') as f:
        json.dump(adapter_config, f, indent=2)
    
    # Also save to merged config directory
    with open(os.path.join(merged_config_dir, "adapter_config.json"), 'w') as f:
        json.dump(adapter_config, f, indent=2)
    
    logger.info("INFO: Adapter config updated successfully")
else:
    logger.warning("⚠️ adapter_config.json not found - this might cause issues")

# 📝 Create README with usage instructions
logger.info("INFO: Creating README with usage instructions...")
readme_content = f"""# {args.output_name}

This is a LoRA (Low-Rank Adaptation) model fine-tuned for MCQA tasks with custom MMLU formatting.

## Base Model
- **Base Model**: `{MODEL_NAME}`
- **LoRA Rank**: 8
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Chat Template**: Custom MMLU chat template for consistent formatting

## Usage

### Option 1: Load with PEFT (Recommended)
```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Load model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained("{HF_REPO_ID}")
tokenizer = AutoTokenizer.from_pretrained("{HF_REPO_ID}")

# Generate text
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Option 2: Merge for Faster Inference
```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Load and merge
model = AutoPeftModelForCausalLM.from_pretrained("{HF_REPO_ID}")
model = model.merge_and_unload()  # Merge LoRA weights
tokenizer = AutoTokenizer.from_pretrained("{HF_REPO_ID}")

# Now use like a regular model
```

## Model Details
- **Training Dataset**: {DATASET_NAME}
- **Fine-tuning Method**: LoRA
- **Task**: Multiple Choice Question Answering (MCQA)
- **Training Format**: Exact MMLU format for loglikelihood_acc_norm evaluation

## Dependencies
```bash
pip install transformers peft torch trl
```
"""

readme_path = os.path.join(OUTPUT_DIR, "README.md")
with open(readme_path, 'w') as f:
    f.write(readme_content)

# Also save to merged config directory
with open(os.path.join(merged_config_dir, "README.md"), 'w') as f:
    f.write(readme_content)

logger.info(f"INFO: README.md created at {readme_path}")

# === PUSH COMPLETE LORA MODEL TO HUB ===
logger.info("INFO: Pushing LoRA model (with configs) to Hugging Face Hub...")
logger.info("INFO: Including base model config and tokenizer for complete functionality")

try:
    # Push the complete LoRA model directory (includes config.json and tokenizer)
    model.push_to_hub(HF_REPO_ID, token=HF_TOKEN, safe_serialization=True, private=False)
    
    # Ensure tokenizer is uploaded with proper config
    tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN, private=False)
    
    # Also push the base model config explicitly
    from huggingface_hub import upload_file
    upload_file(
        path_or_fileobj=os.path.join(merged_config_dir, CONFIG_NAME),
        path_in_repo=CONFIG_NAME,
        repo_id=HF_REPO_ID,
        token=HF_TOKEN,
        repo_type="model"
    )
    
    logger.info("INFO: LoRA adapters and tokenizer pushed successfully")
    logger.info(f"INFO: Usage instructions:")
    logger.info(f"   from peft import AutoPeftModelForCausalLM")
    logger.info(f"   from transformers import AutoTokenizer")
    logger.info(f"   ")
    logger.info(f"   model = AutoPeftModelForCausalLM.from_pretrained('{HF_REPO_ID}')")
    logger.info(f"   tokenizer = AutoTokenizer.from_pretrained('{HF_REPO_ID}')")
    logger.info(f"   ")
    logger.info(f"   # Optional: merge for faster inference")
    logger.info(f"   model = model.merge_and_unload()")
    
except Exception as e:
    logger.error(f"ERROR: Error pushing LoRA adapters: {e}")
    logger.warning("WARNING: Model saved locally only")
    
    # Print local usage instructions
    logger.info(f"INFO: Local usage instructions:")
    logger.info(f"   from peft import AutoPeftModelForCausalLM")
    logger.info(f"   from transformers import AutoTokenizer")
    logger.info(f"   ")
    logger.info(f"   model = AutoPeftModelForCausalLM.from_pretrained('{OUTPUT_DIR}')")
    logger.info(f"   tokenizer = AutoTokenizer.from_pretrained('{OUTPUT_DIR}')")

logger.info("INFO: Done.")

class CustomBatchIterableDataset(IterableDataset):
    def __init__(self, dataset, sampler):
        self.dataset = dataset
        self.sampler = sampler

    def __iter__(self):
        for batch_indices in self.sampler:
            batch = [self.dataset[i] for i in batch_indices]
            yield batch

# Usage:
train_iterable_dataset = CustomBatchIterableDataset(tokenized_train_dataset, sampler)
