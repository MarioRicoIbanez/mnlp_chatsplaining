import argparse
from training.trainer import ModelTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen models using simplified approach")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B",
                      help="Name of the base model to use")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                      help="Maximum sequence length")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                      help="Load model in 4-bit precision")
    
    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=64,
                      help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                      help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                      help="LoRA dropout")
    
    # Dataset configuration
    parser.add_argument("--reasoning_dataset", type=str, required=True,
                      help="Name of the reasoning dataset")
    parser.add_argument("--max_samples", type=int, default=100,
                      help="Maximum number of samples to load")
    parser.add_argument("--is_mcqa", action="store_true", default=True,
                      help="Whether the dataset is in MCQA format")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=1,
                      help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                      help="Number of gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=10,
                      help="Number of warmup steps")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                      help="Learning rate")
    parser.add_argument("--logging_steps", type=float, default=0.2,
                      help="Logging frequency")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="output",
                      help="Directory to save model outputs")
    parser.add_argument("--model_save_name", type=str, default="qwen3-finetuned",
                      help="Name for the saved model")
    parser.add_argument("--push_to_hub", action="store_true", default=False,
                      help="Push model to Hugging Face Hub")
    
    # Hugging Face configuration
    parser.add_argument("--hf_token", type=str, default=None,
                      help="Hugging Face token")
    
    return parser.parse_args()

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen model")
    parser.add_argument("--reasoning_dataset", type=str, required=True,
                       help="Name of the reasoning dataset to use")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B",
                       help="Model name to use")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum number of samples to use")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory for the model")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting training with:")
    print(f"   Model: {args.model_name}")
    print(f"   Dataset: {args.reasoning_dataset}")
    print(f"   Max samples: {args.max_samples}")
    
    # Initialize trainer
    trainer = ModelTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
    )
    
    # Prepare dataset (assuming MCQA format for the provided dataset)
    _ =trainer.prepare_datasets(
        reasoning_dataset_name=args.reasoning_dataset,
        max_samples=args.max_samples,
        is_mcqa=True,  # Set to True for MCQA datasets like sciq_treated_epfl_mcqa
    )
    
    # Setup LoRA
    trainer.setup_lora()
    
    # Train the model
    print("🔥 Starting training...")
    trainer_stats = trainer.train(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=0.1,  # Log more frequently for small datasets
    )
    
    # Save the model
    trainer.save_model("qwen3-finetuned")
    
    print("✅ Training completed successfully!")
    print(f"📊 Training stats: {trainer_stats}")

     

if __name__ == "__main__":
    main() 