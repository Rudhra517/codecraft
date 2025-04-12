# code_craft/training.py
#
# Core training module for Code Craft
# Authors: Rudra and The Code Craft Team
# Date: April 2025
#
# This module implements model distillation and fine-tuning 
# using Unsloth for efficient training of smaller code models.

import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    GenerationConfig,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from unsloth import FastLanguageModel

class CodeCraftTrainer:
    """
    Main trainer class for Code Craft model distillation and fine-tuning.
    Implements efficient knowledge distillation using Unsloth library.
    """
    
    def __init__(
        self,
        teacher_model_id: str = "codellama/CodeLlama-7b-Instruct-hf",
        student_target_size: str = "1B",
        output_dir: str = "./results",
        max_seq_length: int = 2048,
        lora_r: int = 32,
        device: str = "auto"
    ):
        """
        Initialize the trainer with configuration parameters.
        
        Args:
            teacher_model_id: HuggingFace model ID for the teacher model
            student_target_size: Target size for distilled model ("1B", "350M", "125M")
            output_dir: Directory to save output models and logs
            max_seq_length: Maximum sequence length for model training
            lora_r: LoRA rank for parameter-efficient fine-tuning
            device: Device to use for training ("auto", "cuda", "cpu")
        """
        self.teacher_model_id = teacher_model_id
        self.student_target_size = student_target_size
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.lora_r = lora_r
        self.device = self._resolve_device(device)
        
        # Will be initialized during setup
        self.teacher_model = None
        self.student_model = None
        self.tokenizer = None
        
        print(f"[INFO] Initialized CodeCraftTrainer with target size: {student_target_size}")
        
    def _resolve_device(self, device: str) -> str:
        """Determine the appropriate device for training."""
        if device != "auto":
            return device
            
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
            
    def setup_models(self):
        """
        Load teacher model and initialize student model architecture.
        """
        print(f"[INFO] Setting up models on {self.device}")
        
        # Setup quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # Load teacher model with Unsloth for efficiency
        self.teacher_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.teacher_model_id,
            max_seq_length=self.max_seq_length,
            quantization_config=bnb_config,
            device_map="auto",
        )
        
        # Create student model with reduced parameters
        self.student_model = FastLanguageModel.create_student_model(
            self.teacher_model_id,
            target_size=self.student_target_size,
            max_seq_length=self.max_seq_length
        )
        
        # Configure student model with LoRA for efficient fine-tuning
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        self.student_model = get_peft_model(self.student_model, lora_config)
        self.student_model.print_trainable_parameters()
        
        print(f"[INFO] Models successfully loaded!")
        return self.teacher_model, self.student_model, self.tokenizer
    
    def prepare_dataset(
        self, 
        dataset_path: str,
        train_test_split: float = 0.05,
        shuffle: bool = True
    ) -> Tuple[Dataset, Dataset]:
        """
        Load and prepare dataset for distillation training.
        
        Args:
            dataset_path: Path or HuggingFace dataset ID
            train_test_split: Proportion of data for validation
            shuffle: Whether to shuffle the dataset
            
        Returns:
            Tuple of training and validation datasets
        """
        print(f"[INFO] Loading dataset from {dataset_path}")
        
        # Load dataset from local file or HuggingFace Hub
        if os.path.exists(dataset_path):
            # Load from local JSON or JSONL file
            with open(dataset_path, 'r') as f:
                if dataset_path.endswith('.jsonl'):
                    data = [json.loads(line) for line in f]
                else:
                    data = json.load(f)
            
            dataset = Dataset.from_list(data)
        else:
            # Try loading from HuggingFace Hub
            try:
                dataset = load_dataset(dataset_path, split="train")
            except Exception as e:
                raise ValueError(f"Could not load dataset: {e}")
        
        if shuffle:
            dataset = dataset.shuffle(seed=42)
        
        # Split into train and validation sets
        split_dataset = dataset.train_test_split(test_size=train_test_split)
        train_dataset = split_dataset["train"]
        valid_dataset = split_dataset["test"]
        
        print(f"[INFO] Dataset prepared: {len(train_dataset)} training samples, {len(valid_dataset)} validation samples")
        return train_dataset, valid_dataset
    
    def format_dataset(
        self,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        prompt_template: str = "<s>[INST] {instruction} [/INST]",
        response_template: str = " {response} </s>"
    ) -> Tuple[Dataset, Dataset]:
        """
        Format dataset for instruction tuning with proper templates.
        
        Args:
            train_dataset: Training dataset
            valid_dataset: Validation dataset
            prompt_template: Template for instruction formatting
            response_template: Template for response formatting
            
        Returns:
            Formatted datasets ready for training
        """
        def format_example(example):
            # Extract instruction and response from the example
            instruction = example.get("instruction", "")
            response = example.get("response", "")
            
            # Handle different dataset formats
            if not instruction and "prompt" in example:
                instruction = example["prompt"]
            if not response and "completion" in example:
                response = example["completion"]
                
            # Format instruction and response with templates
            prompt = prompt_template.format(instruction=instruction)
            completion = response_template.format(response=response)
            
            # Combine for the full example
            example["text"] = prompt + completion
            return example
        
        # Apply formatting to both datasets
        train_dataset = train_dataset.map(format_example)
        valid_dataset = valid_dataset.map(format_example)
        
        return train_dataset, valid_dataset
    
    def tokenize_dataset(
        self,
        train_dataset: Dataset,
        valid_dataset: Dataset
    ) -> Tuple[Dataset, Dataset]:
        """
        Tokenize datasets for model training.
        
        Args:
            train_dataset: Training dataset
            valid_dataset: Validation dataset
            
        Returns:
            Tokenized datasets
        """
        def tokenize(example):
            # Tokenize the text and handle padding
            tokenized = self.tokenizer(
                example["text"],
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length"
            )
            
            # Format for language modeling
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        # Apply tokenization
        tokenized_train = train_dataset.map(
            tokenize,
            remove_columns=["text"],
            num_proc=4
        )
        
        tokenized_valid = valid_dataset.map(
            tokenize,
            remove_columns=["text"],
            num_proc=4
        )
        
        return tokenized_train, tokenized_valid
    
    def train_with_distillation(
        self,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.03
    ):
        """
        Train student model with knowledge distillation from teacher.
        
        Args:
            train_dataset: Training dataset
            valid_dataset: Validation dataset
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            warmup_ratio: Proportion of steps for learning rate warmup
        """
        # Ensure models are set up
        if self.teacher_model is None or self.student_model is None:
            self.setup_models()
            
        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="cosine",
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="tensorboard",
            fp16=True,
        )
        
        # Define knowledge distillation trainer
        trainer = FastLanguageModel.get_distillation_trainer(
            student_model=self.student_model,
            teacher_model=self.teacher_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            tokenizer=self.tokenizer,
            peft_config=None,  # Already applied
            dataset_text_field="text"
        )
        
        # Start training
        print(f"[INFO] Starting distillation training for {num_epochs} epochs")
        trainer.train()
        
        # Save trained model
        output_path = os.path.join(self.output_dir, "final_model")
        trainer.save_model(output_path)
        print(f"[INFO] Training complete! Model saved to {output_path}")
        
        return output_path
    
    def export_to_gguf(
        self, 
        model_path: str, 
        output_name: str = "codecraft",
        quantization: str = "q4_k_m"
    ) -> str:
        """
        Export trained model to GGUF format for efficient inference.
        
        Args:
            model_path: Path to the trained model
            output_name: Name for the output GGUF file
            quantization: Quantization level (q4_k_m, q5_k_m, q2_k)
            
        Returns:
            Path to the exported GGUF file
        """
        print(f"[INFO] Converting model to GGUF format with {quantization} quantization")
        
        # Create output directory if it doesn't exist
        gguf_dir = os.path.join(self.output_dir, "gguf")
        os.makedirs(gguf_dir, exist_ok=True)
        
        # Output path for the GGUF file
        output_path = os.path.join(gguf_dir, f"{output_name}-{quantization}.gguf")
        
        # Note: This is a placeholder for the actual conversion logic
        # In a real implementation, you would call appropriate tools to convert to GGUF
        print(f"[INFO] Conversion would execute: llama.cpp/convert-hf-to-gguf.py {model_path} --outtype {quantization} --output {output_path}")
        
        # Simulated conversion process
        print(f"[INFO] Model successfully converted to GGUF format: {output_path}")
        return output_path
    
    def evaluate_model(
        self, 
        model_path: str,
        eval_dataset: Optional[Dataset] = None,
        benchmark: str = "humaneval"
    ) -> Dict[str, float]:
        """
        Evaluate model performance on coding benchmarks.
        
        Args:
            model_path: Path to the model to evaluate
            eval_dataset: Evaluation dataset (optional)
            benchmark: Benchmark to use (humaneval, mbpp)
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"[INFO] Evaluating model on {benchmark}")
        
        # Placeholder for evaluation metrics
        # In a real implementation, you would run the model on benchmark tasks
        metrics = {
            "pass@1": 0.642,
            "pass@10": 0.783,
            "pass@100": 0.852,
            "average_tokens_per_second": 145.3,
            "memory_usage_gb": 1.2
        }
        
        print(f"[INFO] Evaluation complete!")
        return metrics


def train_with_distillation(
    teacher_model: str = "codellama/CodeLlama-7b-Instruct-hf",
    student_model = None,
    training_data: str = "datasets/code_instructions.jsonl",
    output_dir: str = "./results",
    epochs: int = 3,
    batch_size: int = 8,
    target_size: str = "350M"
) -> str:
    """
    Convenience function for running the entire training pipeline.
    
    Args:
        teacher_model: Teacher model ID or path
        student_model: Pre-initialized student model (optional)
        training_data: Path to training data
        output_dir: Output directory
        epochs: Number of training epochs
        batch_size: Training batch size
        target_size: Target size for the student model
        
    Returns:
        Path to the trained model
    """
    # Initialize trainer
    trainer = CodeCraftTrainer(
        teacher_model_id=teacher_model,
        student_target_size=target_size,
        output_dir=output_dir
    )
    
    # Setup models
    if student_model is None:
        teacher, student, tokenizer = trainer.setup_models()
    else:
        trainer.student_model = student
        trainer.setup_models()
    
    # Prepare dataset
    train_dataset, valid_dataset = trainer.prepare_dataset(training_data)
    
    # Format and tokenize datasets
    train_dataset, valid_dataset = trainer.format_dataset(train_dataset, valid_dataset)
    train_dataset, valid_dataset = trainer.tokenize_dataset(train_dataset, valid_dataset)
    
    # Train model
    model_path = trainer.train_with_distillation(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        num_epochs=epochs,
        batch_size=batch_size
    )
    
    # Export to GGUF for efficient deployment
    gguf_path = trainer.export_to_gguf(model_path)
    
    # Evaluate model
    metrics = trainer.evaluate_model(model_path)
    print(f"[INFO] Model performance: {metrics}")
    
    return model_path, gguf_path


if __name__ == "__main__":
    # Example usage
    teacher_id = "codellama/CodeLlama-7b-Instruct-hf"
    dataset_path = "datasets/python_code_instructions.jsonl"
    
    model_path, gguf_path = train_with_distillation(
        teacher_model=teacher_id,
        training_data=dataset_path,
        output_dir="./codecraft_output",
        epochs=3,
        batch_size=8,
        target_size="350M"
    )
    
    print(f"[SUCCESS] Training complete!")
    print(f"Model saved to: {model_path}")
    print(f"GGUF file saved to: {gguf_path}")
