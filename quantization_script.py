from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
import torch
import json

# === Fine-tuned Model ===
MODEL_ID = "finetuned_qwen_mcqa"  # Local folder with model + tokenizer
SAVE_DIR = "qwen3-w8a8-quantized"

# === Load model and tokenizer ===
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# === Prepare Calibration Data ===
# We'll reuse the same dataset used for training if available
# Otherwise use a fallback like SciQ or any short QA dataset

# If training data is in JSON format (like output/mcqa_train_from_sciq.json)
with open("output/mcqa_train_from_sciq.json", "r") as f:
    raw_data = json.load(f)

# Use a small subset for calibration
calibration_data = raw_data[:512]

# Format into Dataset
def format_prompt(example):
    prompt = f"Context: {example['explanation']}\n"
    prompt += f"Question: {example['question']}\n"
    for i, choice in enumerate(example["choices"]):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "Answer:"
    return {"text": prompt}

formatted = [format_prompt(x) for x in calibration_data]
dataset = Dataset.from_list(formatted)

# Tokenize calibration data
MAX_SEQUENCE_LENGTH = 512
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        add_special_tokens=False
    )

dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# === Configure Quantization: SmoothQuant + GPTQ (W8A8) ===
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

# === Run Quantization ===
oneshot(
    model=model,
    dataset=dataset,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=len(dataset),
)

# === Test Generation ===
print("\n========= SAMPLE GENERATION =========")
input_ids = tokenizer("Explain gravity in simple terms.", return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("=====================================\n")

# === Save the Quantized Model ===
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Quantized model saved to {SAVE_DIR}")
