from transformers import AutoModelForCausalLM, AutoTokenizer

SAVE_DIR = "qwen3-w8a8-quantized"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(SAVE_DIR, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)

# Push to your Hugging Face repo
model.push_to_hub("talphaidze/qwen3-w8a8-quantized")
tokenizer.push_to_hub("talphaidze/qwen3-w8a8-quantized")
