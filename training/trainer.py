from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, Dataset
import pandas as pd
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer
from typing import Optional, List, Dict, Union
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(
        self,
        model_name: str = "unsloth/Qwen3-14B",
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        full_finetuning: bool = False,
        hf_token: Optional[str] = None,
        output_dir: str = "output",
    ):
        """
        Initialize the model trainer.
        
        Args:
            model_name (str): Name of the model to load
            max_seq_length (int): Maximum sequence length
            load_in_4bit (bool): Whether to load model in 4-bit precision
            load_in_8bit (bool): Whether to load model in 8-bit precision
            full_finetuning (bool): Whether to do full model finetuning
            hf_token (str, optional): Hugging Face token for gated models
            output_dir (str): Directory to save model outputs
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.full_finetuning = full_finetuning
        self.hf_token = hf_token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and tokenizer
        self.model, self.tokenizer = self._load_model()
        
    def _load_model(self):
        """Load the model and tokenizer."""
        return FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            full_finetuning=self.full_finetuning,
            token=self.hf_token,
        )
    
    def prepare_lora(
        self,
        r: int = 32,
        target_modules: Optional[List[str]] = None,
        lora_alpha: int = 32,
        lora_dropout: float = 0,
        bias: str = "none",
        use_gradient_checkpointing: Union[bool, str] = "unsloth",
        random_state: int = 3407,
        use_rslora: bool = False,
        loftq_config: Optional[Dict] = None,
    ):
        """
        Prepare LoRA adapters for the model.
        
        Args:
            r (int): LoRA rank
            target_modules (List[str]): Target modules for LoRA
            lora_alpha (int): LoRA alpha parameter
            lora_dropout (float): LoRA dropout
            bias (str): Bias type
            use_gradient_checkpointing (Union[bool, str]): Whether to use gradient checkpointing
            random_state (int): Random seed
            use_rslora (bool): Whether to use rank stabilized LoRA
            loftq_config (Dict, optional): LoftQ configuration
        """
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
            
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
            use_rslora=use_rslora,
            loftq_config=loftq_config,
        )
    
    def prepare_datasets(
        self,
        reasoning_dataset_name: str = "unsloth/OpenMathReasoning-mini",
        non_reasoning_dataset_name: str = "mlabonne/FineTome-100k",
        chat_percentage: float = 0.75,
        seed: int = 3407,
    ):
        """
        Prepare training datasets.
        
        Args:
            reasoning_dataset_name (str): Name of the reasoning dataset
            non_reasoning_dataset_name (str): Name of the non-reasoning dataset
            chat_percentage (float): Percentage of chat data to use
            seed (int): Random seed
        """
        # Load datasets
        reasoning_dataset = load_dataset(reasoning_dataset_name, split="cot")
        non_reasoning_dataset = load_dataset(non_reasoning_dataset_name, split="train")
        
        # Convert reasoning dataset to conversational format
        def generate_conversation(examples):
            problems = examples["problem"]
            solutions = examples["generated_solution"]
            conversations = []
            for problem, solution in zip(problems, solutions):
                conversations.append([
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": solution},
                ])
            return {"conversations": conversations}
        
        reasoning_conversations = self.tokenizer.apply_chat_template(
            reasoning_dataset.map(generate_conversation, batched=True)["conversations"],
            tokenize=False,
        )
        
        # Convert non-reasoning dataset
        from unsloth.chat_templates import standardize_sharegpt
        dataset = standardize_sharegpt(non_reasoning_dataset)
        non_reasoning_conversations = self.tokenizer.apply_chat_template(
            dataset["conversations"],
            tokenize=False,
        )
        
        # Sample and combine datasets
        non_reasoning_subset = pd.Series(non_reasoning_conversations)
        non_reasoning_subset = non_reasoning_subset.sample(
            int(len(reasoning_conversations) * (1.0 - chat_percentage)),
            random_state=seed,
        )
        
        data = pd.concat([
            pd.Series(reasoning_conversations),
            pd.Series(non_reasoning_subset)
        ])
        data.name = "text"
        
        self.combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
        self.combined_dataset = self.combined_dataset.shuffle(seed=seed)
        
    def train(
        self,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 5,
        max_steps: int = 30,
        learning_rate: float = 2e-4,
        logging_steps: int = 1,
        weight_decay: float = 0.01,
        seed: int = 3407,
    ):
        """
        Train the model.
        
        Args:
            per_device_train_batch_size (int): Batch size per device
            gradient_accumulation_steps (int): Number of gradient accumulation steps
            warmup_steps (int): Number of warmup steps
            max_steps (int): Maximum number of training steps
            learning_rate (float): Learning rate
            logging_steps (int): Number of steps between logging
            weight_decay (float): Weight decay
            seed (int): Random seed
        """
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.combined_dataset,
            eval_dataset=None,
            args=SFTConfig(
                dataset_text_field="text",
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                max_steps=max_steps,
                learning_rate=learning_rate,
                logging_steps=logging_steps,
                optim="adamw_8bit",
                weight_decay=weight_decay,
                lr_scheduler_type="linear",
                seed=seed,
                report_to="none",
            ),
        )
        
        # Log initial memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        logger.info(f"{start_gpu_memory} GB of memory reserved.")
        
        # Train
        trainer_stats = trainer.train()
        
        # Log final stats
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        
        logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        logger.info(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        logger.info(f"Peak reserved memory = {used_memory} GB.")
        logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
        logger.info(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
        
        return trainer_stats
    
    def save_model(
        self,
        save_path: str,
        save_method: str = "lora",
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
        hub_token: Optional[str] = None,
    ):
        """
        Save the model.
        
        Args:
            save_path (str): Path to save the model
            save_method (str): Method to save the model (lora, merged_16bit, merged_4bit)
            push_to_hub (bool): Whether to push to Hugging Face Hub
            hub_model_id (str, optional): Model ID on Hugging Face Hub
            hub_token (str, optional): Hugging Face token
        """
        if save_method == "lora":
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            if push_to_hub and hub_model_id and hub_token:
                self.model.push_to_hub(hub_model_id, token=hub_token)
                self.tokenizer.push_to_hub(hub_model_id, token=hub_token)
        else:
            self.model.save_pretrained_merged(
                save_path,
                self.tokenizer,
                save_method=save_method,
            )
            if push_to_hub and hub_model_id and hub_token:
                self.model.push_to_hub_merged(
                    hub_model_id,
                    self.tokenizer,
                    save_method=save_method,
                    token=hub_token,
                )
    
    def save_gguf(
        self,
        save_path: str,
        quantization_method: Union[str, List[str]] = "q8_0",
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
        hub_token: Optional[str] = None,
    ):
        """
        Save the model in GGUF format.
        
        Args:
            save_path (str): Path to save the model
            quantization_method (Union[str, List[str]]): Quantization method(s)
            push_to_hub (bool): Whether to push to Hugging Face Hub
            hub_model_id (str, optional): Model ID on Hugging Face Hub
            hub_token (str, optional): Hugging Face token
        """
        if push_to_hub and hub_model_id and hub_token:
            self.model.push_to_hub_gguf(
                hub_model_id,
                self.tokenizer,
                quantization_method=quantization_method,
                token=hub_token,
            )
        else:
            self.model.save_pretrained_gguf(
                save_path,
                self.tokenizer,
                quantization_method=quantization_method,
            )
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        enable_thinking: bool = False,
    ):
        """
        Generate text from the model.
        
        Args:
            messages (List[Dict[str, str]]): List of messages
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling parameter
            top_k (int): Top-k sampling parameter
            enable_thinking (bool): Whether to enable thinking mode
        """
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        
        if enable_thinking:
            max_new_tokens = 1024
            temperature = 0.6
            top_p = 0.95
        
        return self.model.generate(
            **self.tokenizer(text, return_tensors="pt").to("cuda"),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            streamer=TextStreamer(self.tokenizer, skip_prompt=True),
        ) 