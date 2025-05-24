# Model loading utils 
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, TrainingArguments
import torch 

def load_model(
    model_name: str,
    hf_token: str = None,
    use_fast_tokenizer: bool = True,
    trust_remote_code: bool = True,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    use_cache: bool = False,
    pretraining_tp: int = 1,
    load_in_4bit: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model and tokenizer in 4-bit precision.
    
    Args:
        model_name (str): Name or path of the model to load
        hf_token (str, optional): HuggingFace token for private models. Defaults to None.
        use_fast_tokenizer (bool, optional): Whether to use fast tokenizer. Defaults to True.
        trust_remote_code (bool, optional): Whether to trust remote code. Defaults to True.
        device_map (str, optional): Device mapping strategy. Defaults to "auto".
        torch_dtype (torch.dtype, optional): Torch data type. Defaults to torch.bfloat16.
        use_cache (bool, optional): Whether to use cache. Defaults to False.
        pretraining_tp (int, optional): Pretraining tensor parallelism. Defaults to 1.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit precision. Defaults to False.
    Returns:
        tuple[AutoModelForCausalLM, AutoTokenizer]: Loaded model and tokenizer
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=use_fast_tokenizer,
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure model loading parameters
    model_kwargs = {
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        "token": hf_token,
    }

    # Add quantization config if requested
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
        model_kwargs["quantization_config"] = bnb_config

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    # Configure model for training
    model.config.use_cache = use_cache
    model.config.pretraining_tp = pretraining_tp

    return model, tokenizer 