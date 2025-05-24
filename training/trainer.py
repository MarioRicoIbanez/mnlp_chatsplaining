from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
import pandas as pd
from trl import SFTTrainer, SFTConfig
from typing import Optional, List, Dict, Union
import logging
from pathlib import Path
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
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
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            trust_remote_code=True,
            token=self.hf_token,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            token=self.hf_token,
        )
        return model, tokenizer
    
    def prepare_lora(
        self,
        r: int = 32,
        target_modules: Optional[List[str]] = None,
        lora_alpha: int = 32,
        lora_dropout: float = 0,
        bias: str = "none",
        use_gradient_checkpointing: bool = True,
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
            use_gradient_checkpointing (bool): Whether to use gradient checkpointing
            random_state (int): Random seed
            use_rslora (bool): Whether to use rank stabilized LoRA
            loftq_config (Dict, optional): LoftQ configuration
        """
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
    
    def _convert_mcqa_to_chat(self, examples):
        """
        Convert MCQA format to chat format matching the evaluation format and Qwen3's thinking mode.
        
        Args:
            examples: Dictionary containing MCQA fields
            
        Returns:
            Dict with chat format
        """
        questions = examples["question"]
        choices = examples["choices"]
        answers = examples["answer_text"]
        explanations = examples["explanation"]
        
        conversations = []
        for question, choice_list, answer, explanation in zip(questions, choices, answers, explanations):
            # Create the question with choices in the same format as evaluation
            question_text = f"The following are multiple choice questions (with answers) about knowledge and skills in advanced master-level STEM courses.\n\n"
            question_text += f"{question}\n"
            # Use letters A-Z for choices, matching the number of choices provided
            question_text += "".join([f"{chr(65+i)}. {choice}\n" for i, choice in enumerate(choice_list)])
            question_text += "Answer:"
            
            # Create the answer with explanation in Qwen3's thinking mode format
            answer_text = f"<think>Let me analyze this step by step:\n1. First, I'll read the question carefully\n2. Then, I'll evaluate each option\n3. Finally, I'll explain my reasoning\n\n{explanation}</think>\n{answer}"
            
            conversations.append([
                {"role": "user", "content": question_text},
                {"role": "assistant", "content": answer_text}
            ])
        
        return {"conversations": conversations}
    
    def prepare_datasets(
        self,
        reasoning_dataset_name: str = "Open-Orca/OpenMathReasoning-10k",
        non_reasoning_dataset_name: str = "mlabonne/FineTome-100k",
        chat_percentage: float = 0.75,
        seed: int = 3407,
        is_mcqa: bool = False,
    ):
        """
        Prepare training datasets.
        
        Args:
            reasoning_dataset_name (str): Name of the reasoning dataset
            non_reasoning_dataset_name (str): Name of the non-reasoning dataset
            chat_percentage (float): Percentage of chat data to use
            seed (int): Random seed
            is_mcqa (bool): Whether the dataset is in MCQA format
        """
        # Load reasoning dataset
        reasoning_dataset = load_dataset(reasoning_dataset_name, split="train")
        
        # Convert MCQA format to chat if needed
        if is_mcqa:
            reasoning_dataset = reasoning_dataset.map(
                self._convert_mcqa_to_chat,
                batched=True,
                remove_columns=reasoning_dataset.column_names
            )
            reasoning_conversations = reasoning_dataset["conversations"]
        else:
            # For non-MCQA datasets, assume they're already in chat format
            reasoning_conversations = reasoning_dataset["conversations"]
        
        # Load non-reasoning dataset
        non_reasoning_dataset = load_dataset(non_reasoning_dataset_name, split="train")
        
        # Get conversations from non-reasoning dataset
        if "conversations" in non_reasoning_dataset.features:
            non_reasoning_conversations = non_reasoning_dataset["conversations"]
        else:
            # Assume it's in text format and needs conversion
            non_reasoning_conversations = non_reasoning_dataset["text"]
        
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
        elif save_method.startswith("merged"):
            # Merge LoRA weights with base model and save
            base_model = self.model.merge_and_unload()
            base_model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            if push_to_hub and hub_model_id and hub_token:
                base_model.push_to_hub(hub_model_id, token=hub_token)
                self.tokenizer.push_to_hub(hub_model_id, token=hub_token)
    
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
        enable_thinking: bool = True,
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
        # Use Qwen3's chat template with thinking mode
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
        # Adjust generation parameters based on thinking mode
        if enable_thinking:
            temperature = 0.6
            top_p = 0.95
            max_new_tokens = min(max_new_tokens, 32768)  # Qwen3's max context length
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            streamer=TextStreamer(self.tokenizer, skip_prompt=True),
        )
        
        # Parse thinking content if enabled
        if enable_thinking:
            output_ids = output[0][len(inputs.input_ids[0]):].tolist()
            try:
                # Find the end of thinking content (</think> token)
                index = len(output_ids) - output_ids[::-1].index(151668)
                thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                return {"thinking": thinking_content, "response": content}
            except ValueError:
                # If no thinking content found, return just the response
                return {"thinking": "", "response": self.tokenizer.decode(output[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()}
        else:
            return {"thinking": "", "response": self.tokenizer.decode(output[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()} 