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
    args = parse_args()
    
    logger.info("🚀 Starting training with simplified approach:")
    logger.info(f"   Model: {args.model_name}")
    logger.info(f"   Dataset: {args.reasoning_dataset}")
    logger.info(f"   Max samples: {args.max_samples}")
    logger.info(f"   MCQA format: {args.is_mcqa}")
    
    # Initialize the trainer
    trainer = ModelTrainer(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        hf_token=args.hf_token,
        output_dir=args.output_dir
    )
    
    # Prepare datasets
    trainer.prepare_datasets(
        reasoning_dataset_name=args.reasoning_dataset,
        max_samples=args.max_samples,
        is_mcqa=args.is_mcqa
    )
    
    # Setup LoRA
    trainer.setup_lora(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    # Train the model
    logger.info("🔥 Starting training...")
    trainer_stats = trainer.train(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps
    )
    
    # Save the model
    trainer.save_model(
        model_name=args.model_save_name,
        push_to_hub=args.push_to_hub
    )
    
    # Test generation (optional)
    if hasattr(trainer.train_dataset, '__getitem__') and len(trainer.train_dataset) > 0:
        sample = trainer.train_dataset[0]
        if 'text' in sample:
            # Extract question part for testing
            sample_text = sample['text']
            if "Answer:" in sample_text:
                question_part = sample_text.split("Answer:")[0].strip()
                logger.info("🧪 Testing generation...")
                response = trainer.generate(question_part, max_new_tokens=200)
                logger.info(f"Generated response: {response[:200]}...")
    
    logger.info("✅ Training completed successfully!")
    logger.info(f"📊 Training stats: {trainer_stats}")

if __name__ == "__main__":
    main() 