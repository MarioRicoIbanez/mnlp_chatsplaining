import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

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
from utils.dataset_utils import (
    process_mcq_dataset,
    SFTDataCollator,
)  # <-- import your ChatML logic here
from transformers.utils import CONFIG_NAME

from utils.dataset_utils import tokenize_func
from utils.batching import SmartPaddingTokenBatchSampler
from torch.utils.data import DataLoader

import wandb

wandb.init(project="chatsplaining")

# === CONFIG ===
MODEL_NAME = "RikoteMaster/Qwen3-0.6B-SFT-Open"  # SFT model for weights
TOKENIZER_NAME = "Qwen/Qwen3-0.6B"  # Base model for tokenizer
DATASET_NAME = "jonlecumberri/MNLP_M2_mcqa_dataset"
OUTPUT_DIR = "qwen_chatml_mcqa_output"
HF_REPO_ID = "RikoteMaster/MNLP_M2_mcqa_model_chatml"
HF_TOKEN = ""

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === LOAD DATASET ===
print(" Loading dataset...")
dataset = load_dataset(DATASET_NAME)["train"]
dataset = dataset.shuffle(seed=42)
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
val_dataset = split["test"]

# === TOKENIZER AND MODEL ===
print(" Loading model and tokenizer...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

#  Use the tokenizer from the base model
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

#  Load fine-tuned model weights
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)

# === LoRA ===
print(" Applying LoRA...")
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
print(" Fixing gradient requirements for Qwen3 + LoRA...")

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
            
print(f"Model is in training mode: {model.training}")
model.print_trainable_parameters()

# CRITICAL: Enable gradient computation for all parameters that require it
for name, param in model.named_parameters():
    if param.requires_grad:
        param.retain_grad()  # Ensure gradients are retained

# === PREPROCESS ===
print(" Preprocessing...")





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
data_collator = SFTDataCollator(tokenizer=tokenizer)

# ALTERNATIVE: If loss is still 0, try using the standard data collator
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False,  # We're doing causal LM, not masked LM
#     pad_to_multiple_of=None,
# )

# Set up token-budget batching
max_tok_per_gpu = 8_000  # fits comfortably in 24 GB with bf16
sampler = SmartPaddingTokenBatchSampler(
    tokenized_train_dataset["length"], max_tok_per_gpu
)

print(f"🔧 Smart Batching Info:")
print(f"  - Max tokens per GPU: {max_tok_per_gpu}")
print(f"  - Dataset length: {len(tokenized_train_dataset)}")

# Debug: Count total batches that will be created - do this properly
batch_count = 0
sample_batch_sizes = []
temp_sampler = SmartPaddingTokenBatchSampler(
    tokenized_train_dataset["length"], max_tok_per_gpu
)

# Count ALL batches that will actually be generated
print("🔍 Counting actual batches that will be generated...")
for batch_indices in temp_sampler:
    batch_count += 1
    sample_batch_sizes.append(len(batch_indices))
    if batch_count <= 5:  # Show first 5 batch sizes
        max_len = max([tokenized_train_dataset[i]["length"] for i in batch_indices])
        print(f"  - Batch {batch_count}: {len(batch_indices)} samples, max_len: {max_len}")

print(f"  - Total ACTUAL batches: {batch_count}")
print(f"  - Average batch size: {sum(sample_batch_sizes) / len(sample_batch_sizes):.1f}")
print(f"  - Sampler.__len__() estimate: {len(sampler)}")

# Validate our count
if batch_count == 0:
    print("❌ ERROR: No batches generated! Check your sampler configuration.")
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



dl_eval = DataLoader(
    tokenized_val_dataset,
    batch_size=4,
    collate_fn=data_collator,
    num_workers=0,  # or >0 if RAM allows
    pin_memory=False,
)

# === TRAINING ARGS ===
print(" Setting up training...")

# Calculate expected steps for smart batching
print(f"📊 Dataset info:")
print(f"  - Training samples: {len(tokenized_train_dataset)}")
print(f"  - Smart batch sampler will create dynamic batches")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    # Use max_steps with actual batch count - this ensures we train on ALL data
    max_steps=batch_count,
    gradient_accumulation_steps=1,  # Temporarily disable to debug
    logging_steps=50,
    save_steps=1000,
    eval_steps=500,  # Evaluate less frequently 
    eval_strategy="steps",  # Enable evaluation
    report_to="wandb",
    bf16=True,
    disable_tqdm=False,
    remove_unused_columns=True,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    warmup_steps=1,
    learning_rate=2e-4,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    dataloader_pin_memory=False,
    label_names=["labels"],
    prediction_loss_only=False,
)

print(f"📈 Training will run for exactly {training_args.max_steps} steps (actual batch count)")

# === CUSTOM TRAINER CLASS ===
class CustomTrainer(Trainer):
    def __init__(self, custom_train_dataloader=None, expected_steps=None, **kwargs):
        super().__init__(**kwargs)
        self.custom_train_dataloader = custom_train_dataloader
        self.expected_steps = expected_steps
    
    def get_train_dataloader(self):
        if self.custom_train_dataloader is not None:
            return self.custom_train_dataloader
        return super().get_train_dataloader()
    
    def _get_train_sampler(self):
        # Override to prevent Trainer from creating its own sampler
        return None

# === TRAINER ===
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,  # Set this properly from the start
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    custom_train_dataloader=dl,
    expected_steps=batch_count,
)

# === TRAIN ===
print(" Starting training...")
print(f"🔍 Debug Info:")
print(f"  - TrainingArguments.max_steps: {training_args.max_steps}")
print(f"  - Custom dataloader length: {len(dl)}")
print(f"  - Expected to run {batch_count} steps")

trainer.train()

print(f"🎯 Training completed!")
print(f"  - Total steps trained: {trainer.state.global_step}")
print(f"  - Expected steps: {batch_count}")

# === SAVE LOGS ===
print(" Saving training log and plot...")

# Load training logs
log_df = pd.DataFrame(trainer.state.log_history)
log_df.to_csv(os.path.join(OUTPUT_DIR, "training_log.csv"), index=False)

# Plot
plt.figure(figsize=(8, 5))
plotted = False
loss_columns = []

print(f"📊 Available log columns: {list(log_df.columns)}")

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
    print("📊 Loss plot saved successfully!")
else:
    print("⚠ No loss data found in logs.")

# === SAVE MODEL LOCALLY ===
print(" Saving model and tokenizer...")

# Save model and tokenizer
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)

# ✅ Save config.json manually (from base model)
base_model_config = model.base_model.config
config_path = os.path.join(OUTPUT_DIR, CONFIG_NAME)
base_model_config.to_json_file(config_path)
print(f"⚙ config.json saved to {config_path}")

# === PUSH TO HUB ===
print("📤 Pushing model to Hugging Face Hub...")

# Push model directory to HF (includes config.json)
model.push_to_hub(HF_REPO_ID, token=HF_TOKEN, safe_serialization=True)
tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)

print(" Done.")
