from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        hf_token: str = None,
        output_dir: str = "output",
    ):
        """
        Initialize the model trainer.
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.hf_token = hf_token
        self.output_dir = output_dir
        
        # Initialize model and tokenizer
        self.model, self.tokenizer = self._load_model()
        
        # Check if model supports thinking mode
        self.supports_thinking = "Qwen3" in self.model_name or "qwen3" in self.model_name.lower()
        
        # EOS token for training
        self.EOS_TOKEN = self.tokenizer.eos_token
        
    def _load_model(self):
        """Load the model and tokenizer."""
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            trust_remote_code=True,
            token=self.hf_token,
        )
        
        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config if self.load_in_4bit else None,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            token=self.hf_token,
        )
        
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        
        return model, tokenizer
    
    def _create_prompt_template(self):
        """Create the training prompt template."""
        if self.supports_thinking:
            # Qwen3 with thinking mode
            return """The following are multiple choice questions (with answers) about knowledge and skills in advanced master-level STEM courses.

{}

Answer:
<think>
{}
</think>
{}"""
        else:
            # Older Qwen models without thinking mode
            return """The following are multiple choice questions (with answers) about knowledge and skills in advanced master-level STEM courses.

{}

Answer:
{}

Explanation: {}"""
    
    def _convert_mcqa_to_chat(self, examples):
        """Convert MCQA format to training format."""
        questions = examples["question"]
        choices = examples["choices"] 
        answers = examples["answer_text"]
        explanations = examples["explanation"]
        
        prompt_template = self._create_prompt_template()
        texts = []
        
        for question, choice_list, answer, explanation in zip(questions, choices, answers, explanations):
            # Handle None values
            question = question or ""
            answer = answer or ""
            explanation = explanation or "No explanation provided."
            choice_list = choice_list or []
            
            # Create question with choices
            question_text = f"{question}\n"
            question_text += "".join([f"{chr(65+i)}. {choice}\n" for i, choice in enumerate(choice_list) if choice])
            
            # Format according to model capabilities
            if self.supports_thinking:
                text = prompt_template.format(question_text, explanation, answer)
            else:
                text = prompt_template.format(question_text, answer, explanation)
            
            # Add EOS token
            if not text.endswith(self.EOS_TOKEN):
                text += self.EOS_TOKEN
                
            texts.append(text)
        
        return {"text": texts}
    
%load_ext autoreload
%autoreload 2
    def setup_lora(
        self,
        r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        """Setup LoRA configuration."""
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=r,
            bias="none",
            task_type="CAUSAL_LM",
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
        
        self.model = get_peft_model(self.model, peft_config)
        logger.info("LoRA setup complete")
    
    def train(
        self,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 2,
        warmup_steps: int = 10,
        num_train_epochs: int = 1,
        learning_rate: float = 2e-4,
        logging_steps: float = 0.2,
    ):
        """Train the model."""
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Training arguments
        training_arguments = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            optim="paged_adamw_32bit",
            logging_strategy="steps",
            fp16=False,
            bf16=False,
            group_by_length=True,
            report_to="none",
        )
        
        # Initialize trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_arguments,
            train_dataset=self.train_dataset,
            data_collator=data_collator,
        )
        
        # Clear cache and train
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        self.model.config.use_cache = False
        
        logger.info("Starting training...")
        trainer_stats = trainer.train()
        logger.info("Training completed!")
        
        return trainer_stats
    
    def save_model(self, model_name: str, push_to_hub: bool = False):
        """Save the model."""
        if push_to_hub:
            self.model.push_to_hub(model_name, token=self.hf_token)
            self.tokenizer.push_to_hub(model_name, token=self.hf_token)
        else:
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Model saved {'to hub' if push_to_hub else 'locally'}")
    
    def generate(self, question: str, max_new_tokens: int = 1200):
        """Generate response for a question."""
        if self.supports_thinking:
            prompt = f"""The following are multiple choice questions (with answers) about knowledge and skills in advanced master-level STEM courses.

{question}

Answer:
<think>
"""
        else:
            prompt = f"""The following are multiple choice questions (with answers) about knowledge and skills in advanced master-level STEM courses.

{question}

Answer:
"""
        
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return response[0].split("Answer:")[1] if "Answer:" in response[0] else response[0] 