from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
import torch

# === Model ID ===
MODEL_ID = "RikoteMaster/try_ft"
SAVE_DIR = "MNLP_M3_mcqa_model-W8A8-Quantized"

# === Load model and tokenizer ===
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# === Load and Format Calibration Data ===
DATASET_ID = "RikoteMaster/unified_mcqa_4choice"
DATASET_SPLIT = "validation"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 512

# Load subset of dataset
raw_dataset = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
raw_dataset = raw_dataset.shuffle(seed=42)

# Format into prompt style (Context + Question + Choices + Answer:)
def format_prompt(example):
    prompt = f"Context: {example['explanation']}\n"
    prompt += f"Question: {example['question']}\n"
    for i, choice in enumerate(example["choices"]):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "Answer:"
    return {"text": prompt}

formatted_dataset = raw_dataset.map(format_prompt)

# Tokenize inputs
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        add_special_tokens=False,
    )

tokenized_dataset = formatted_dataset.map(tokenize, remove_columns=formatted_dataset.column_names)

# === Configure Quantization: SmoothQuant + GPTQ (W8A8) ===
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

# === Apply Quantization ===
oneshot(
    model=model,
    dataset=tokenized_dataset,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=len(tokenized_dataset),
)

# === Sample Generation Test ===
print("\n========= SAMPLE GENERATION =========")
input_ids = tokenizer("Explain gravity in simple terms.", return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("=====================================\n")

# === Save the Quantized Model ===
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Quantized model saved to {SAVE_DIR}")
