import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# Jinja2 is not directly used here anymore for prompt formatting, 
# as tokenizer.apply_chat_template handles it internally based on tokenizer.chat_template.
# However, the template string itself is Jinja.
import logging
import re
import sys

# Configure logging (set to WARNING to minimize output, INFO for debugging)
logging.basicConfig(level=logging.WARNING) # Set to WARNING for final True/False output
# logging.basicConfig(level=logging.INFO) # Uncomment for debugging
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
# Fallback for testing if the above model is not accessible:
# MODEL_ID = "Qwen/Qwen2-0.5B-Instruct" 

LETTER_INDICES = ["A", "B", "C", "D", "E", "F", "G"] # Common for MMLU, extend if needed

# The new MMLU chat template provided by the user
MMLU_CHAT_TEMPLATE_STRING = """{% if messages %}
{% for message in messages %}
{% if message['role'] == 'user' %}
The following are multiple choice questions (with answers) about knowledge and skills in advanced master-level STEM courses.

{{ message['content'] }}
Answer:{% elif message['role'] == 'assistant' %} {{ message['content'] }}{% endif %}
{% endfor %}
{% endif %}"""

def load_model_and_tokenizer(model_id: str):
    """Loads the quantized model and tokenizer, and sets the chat template."""
    logger.info(f"Loading tokenizer for {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load tokenizer for {model_id}. Error: {e}")
        logger.error("Please check the model ID and your internet connection.")
        raise

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.warning("Added new pad_token: [PAD]. This might require model resizing if not already known.")

    # Set the custom chat template for the tokenizer
    tokenizer.chat_template = MMLU_CHAT_TEMPLATE_STRING
    logger.info("Set custom MMLU chat template for the tokenizer.")

    logger.info(f"Loading model {model_id} with 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        logger.error(f"Failed to load model {model_id}. Error: {e}")
        logger.error(f"Attempted to load: {model_id}")
        logger.error("Ensure the model ID is correct and the model is compatible with BitsAndBytes quantization.")
        logger.error("If the model is very new or private, it might not be accessible.")
        logger.error("Consider trying a known public model like 'Qwen/Qwen2-0.5B-Instruct' for testing.")
        raise

    model.eval()
    logger.info("Model and tokenizer loaded successfully.")
    return model, tokenizer

def format_prompt_using_chat_template(
    tokenizer, # Tokenizer is now needed here
    question: str,
    choices: list[str],
    letter_indices: list[str]
) -> str:
    """Formats the prompt using the tokenizer's chat_template."""
    
    # 1. Construct the user_message_content (the actual question and choices part)
    user_message_content = question + "\n"
    num_choices_to_render = len(choices)
    current_letter_indices = letter_indices[:num_choices_to_render]
    for letter, choice_text in zip(current_letter_indices, choices):
        user_message_content += f"{letter}. {choice_text}\n"
    user_message_content = user_message_content.strip() # Remove final newline

    # 2. Create the messages list
    messages = [
        {"role": "user", "content": user_message_content}
    ]

    # 3. Apply the tokenizer's chat template
    # tokenizer.chat_template should have been set to MMLU_CHAT_TEMPLATE_STRING when tokenizer was loaded.
    try:
        final_prompt_string = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True # Ensures the prompt ends correctly for the assistant to reply
        )
    except Exception as e:
        logger.error(f"Error applying chat template: {e}")
        logger.error("Ensure the tokenizer's chat_template is set correctly and is a valid Jinja string.")
        raise
        
    return final_prompt_string

def get_llm_prediction(
    model,
    tokenizer,
    prompt: str, # This prompt is now the output of tokenizer.apply_chat_template
    max_new_tokens: int = 5 
) -> str:
    """Gets the raw prediction from the LLM."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    input_ids_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

    generated_ids = outputs[0, input_ids_length:]
    prediction_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    logger.info(f"Raw LLM prediction: '{prediction_text}'")
    return prediction_text

def extract_answer_letter(prediction_text: str, valid_letters: list[str]) -> str | None:
    """
    Extracts the first valid letter from the prediction.
    """
    if not prediction_text:
        return None
    
    first_char_upper = prediction_text[0].upper()
    if first_char_upper in valid_letters:
        return first_char_upper
    
    for char_candidate in prediction_text.upper():
        if char_candidate in valid_letters:
            logger.warning(f"Fallback extraction: used '{char_candidate}' from '{prediction_text}' as no valid letter was at the start.")
            return char_candidate
            
    logger.warning(f"Could not extract a valid letter from: '{prediction_text}'")
    return None

def run_evaluation(mcq_data: dict):
    """
    Runs the evaluation for a single MCQ item and prints True or False.
    mcq_data format: {"question": str, "choices": list[str], "answer": str (e.g., "A")}
    """
    model = None
    tokenizer = None
    try:
        model, tokenizer = load_model_and_tokenizer(MODEL_ID)
    except Exception as e:
        print(False) # Critical: Output False if model loading fails
        sys.exit(1)

    # The tokenizer itself is now passed to the formatting function
    formatted_prompt = format_prompt_using_chat_template(
        tokenizer,
        mcq_data["question"],
        mcq_data["choices"],
        LETTER_INDICES
    )
    logger.info(f"\n--- Formatted Prompt (using chat template) ---\n{formatted_prompt}\n------------------------")

    raw_prediction = get_llm_prediction(model, tokenizer, formatted_prompt)
    predicted_letter = extract_answer_letter(raw_prediction, LETTER_INDICES)
    logger.info(f"Extracted letter: {predicted_letter}")

    gold_letter = mcq_data["answer"].upper()
    
    is_correct = (predicted_letter == gold_letter)
    
    print(is_correct)

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    sample_mcq_to_evaluate = {
        "question": "What is the chemical symbol for water?",
        "choices": [
            "O2",
            "CO2",
            "H2O",
            "NaCl",
            "CH4"
        ],
        "answer": "C" 
    }
    
    run_evaluation(sample_mcq_to_evaluate)

    # Example 2: Different question
    # sample_mcq_2 = {
    #     "question": "Which planet is known as the Red Planet?",
    #     "choices": ["Earth", "Mars", "Jupiter", "Saturn"],
    #     "answer": "B"
    # }
    # print("\n--- Test Case 2 ---")
    # run_evaluation(sample_mcq_2)