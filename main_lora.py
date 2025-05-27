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
)
from peft import LoraConfig, get_peft_model, TaskType
from utils.dataset_utils import (
    process_mcq_dataset,
    SFTDataCollator,
)  # <-- import your ChatML logic here
from transformers.utils import CONFIG_NAME

import wandb

wandb.init(project="chatsplaining")

# === CONFIG ===
MODEL_NAME = "RikoteMaster/Qwen3-0.6B-SFT-Open"  # SFT model for weights
TOKENIZER_NAME = "Qwen/Qwen3-0.6B"  # Base model for tokenizer
DATASET_NAME = "jonlecumberri/MNLP_M2_mcqa_dataset"
OUTPUT_DIR = "qwen_chatml_mcqa_output"
HF_REPO_ID = "jonlecumberri/MNLP_M2_mcqa_model_chatml"
HF_TOKEN = ""

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === LOAD DATASET ===
print(" Loading dataset...")
dataset = load_dataset(DATASET_NAME)["train"]
dataset = dataset.shuffle(seed=42)
split = dataset.train_test_split(test_size=0.2, seed=42)
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

# === PREPROCESS ===
print(" Preprocessing...")


def preprocess(example):
    processed = process_mcq_dataset(example, tokenizer=tokenizer)
    tok = tokenizer(processed["text"], truncation=True, max_length=8192)
    tok["prompt_len"] = processed["prompt_len"]
    return tok


train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(preprocess, remove_columns=val_dataset.column_names)
train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "prompt_len"]
)
val_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "prompt_len"]
)

data_collator = SFTDataCollator(tokenizer)

# === TRAINING ARGS ===
print(" Setting up training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="epoch",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=20,
    save_total_limit=2,
    fp16=True,
    report_to="wandb",
    dataloader_num_workers=4,
)

# === TRAINER ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# === TRAIN ===
print(" Starting training...")
trainer.train()

# === SAVE LOGS ===
print(" Saving training log and plot...")

# Load training logs
log_df = pd.DataFrame(trainer.state.log_history)
log_df.to_csv(os.path.join(OUTPUT_DIR, "training_log.csv"), index=False)

# Plot
plt.figure(figsize=(8, 5))
plotted = False

if "loss" in log_df.columns:
    plt.plot(log_df["step"], log_df["loss"], label="Training Loss")
    plotted = True

if "eval_loss" in log_df.columns:
    plt.plot(log_df["step"], log_df["eval_loss"], label="Eval Loss")
    plotted = True

if plotted:
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.grid(True)

    # Optional: autoscale Y axis with buffer
    min_loss = log_df[["loss", "eval_loss"]].min(numeric_only=True).min()
    max_loss = log_df[["loss", "eval_loss"]].max(numeric_only=True).max()
    plt.ylim(bottom=min_loss * 0.95, top=max_loss * 1.05)

    plt.savefig(os.path.join(OUTPUT_DIR, "loss_plot_scaled.png"))
    print(" Rescaled loss plot saved.")
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
