from training.trainer import ModelTrainer
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train and fine-tune language models using LoRA")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-1.8B",
                      help="Name of the base model to use")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                      help="Maximum sequence length")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                      help="Load model in 4-bit precision")
    parser.add_argument("--load_in_8bit", action="store_true", default=False,
                      help="Load model in 8-bit precision")
    
    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=32,
                      help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                      help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                      help="LoRA dropout")
    
    # Dataset configuration
    parser.add_argument("--reasoning_dataset", type=str, default="Open-Orca/OpenMathReasoning-10k",
                      help="Name of the reasoning dataset")
    parser.add_argument("--non_reasoning_dataset", type=str, default="mlabonne/FineTome-100k",
                      help="Name of the non-reasoning dataset")
    parser.add_argument("--chat_percentage", type=float, default=0.75,
                      help="Percentage of chat data to use")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=2,
                      help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                      help="Number of gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=5,
                      help="Number of warmup steps")
    parser.add_argument("--max_steps", type=int, default=30,
                      help="Maximum number of training steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                      help="Learning rate")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="output",
                      help="Directory to save model outputs")
    parser.add_argument("--save_method", type=str, default="lora",
                      choices=["lora", "merged_16bit", "merged_4bit"],
                      help="Method to save the model")
    
    # Hugging Face configuration
    parser.add_argument("--push_to_hub", action="store_true", default=False,
                      help="Push model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                      help="Model ID on Hugging Face Hub")
    parser.add_argument("--hub_token", type=str, default=None,
                      help="Hugging Face token")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the trainer
    trainer = ModelTrainer(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        output_dir=str(output_dir)
    )
    
    # Prepare LoRA adapters
    trainer.prepare_lora(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_gradient_checkpointing=True
    )
    
    # Prepare datasets
    trainer.prepare_datasets(
        reasoning_dataset_name=args.reasoning_dataset,
        non_reasoning_dataset_name=args.non_reasoning_dataset,
        chat_percentage=args.chat_percentage,
        is_mcqa=True  # Indicate we're using MCQA format
    )
    
    # Train the model
    trainer_stats = trainer.train(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate
    )
    
    # Save the model
    trainer.save_model(
        save_path=str(output_dir / "final_model"),
        save_method=args.save_method,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token
    )
    
    # Example of generating text
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    # Generate with thinking mode enabled
    response = trainer.generate(
        messages=messages,
        max_new_tokens=256,
        temperature=0.7,
        enable_thinking=True
    )
    
    if response["thinking"]:
        logger.info(f"Model thinking: {response['thinking']}")
    logger.info(f"Model response: {response['response']}")
    
    # Example with thinking mode disabled
    response_no_thinking = trainer.generate(
        messages=messages,
        max_new_tokens=256,
        temperature=0.7,
        enable_thinking=False
    )
    
    logger.info(f"Model response (no thinking): {response_no_thinking['response']}")

if __name__ == "__main__":
    main() 