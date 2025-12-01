"""
Fine-tuning script for MediMind using LoRA (Low-Rank Adaptation).
This script fine-tunes microsoft/phi-2 on the medical QA dataset.

Usage:
    python -m app.training.train --output_dir ./models/medimind-phi2-lora
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb


def load_jsonl_dataset(file_path: str) -> List[Dict]:
    """Load dataset from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_prompt(instruction: str, response: str = None) -> str:
    """
    Format instruction-response pair for phi-2.
    Uses the same format as inference.
    """
    if response:
        return f"Human: {instruction}\nAssistant: {response}"
    else:
        return f"Human: {instruction}\nAssistant:"


def preprocess_function(examples, tokenizer, max_length: int = 512):
    """Preprocess dataset for training."""
    # Format prompts
    prompts = [
        format_prompt(inst, resp) 
        for inst, resp in zip(examples["instruction"], examples["response"])
    ]
    
    # Tokenize
    model_inputs = tokenizer(
        prompts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    
    # Create labels (same as input_ids for causal LM)
    labels = model_inputs["input_ids"].copy()
    model_inputs["labels"] = labels
    
    return model_inputs


def create_dataset(data_path: str, tokenizer, max_length: int = 512):
    """Create HuggingFace Dataset from JSONL file."""
    print(f"Loading dataset from {data_path}...")
    data = load_jsonl_dataset(data_path)
    print(f"Loaded {len(data)} examples")
    
    # Convert to HuggingFace Dataset format
    dataset_dict = {
        "instruction": [item["instruction"] for item in data],
        "response": [item["response"] for item in data],
    }
    dataset = Dataset.from_dict(dataset_dict)
    
    # Preprocess
    print("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    return tokenized_dataset


def setup_model_and_tokenizer(
    model_name: str = "microsoft/phi-2",
    use_4bit: bool = True,
    use_8bit: bool = False,
):
    """Setup model and tokenizer with LoRA configuration."""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Configure quantization (4-bit for memory efficiency)
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif use_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    # Prepare model for k-bit training if using quantization
    if use_4bit or use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # LoRA rank
        lora_alpha=16,  # LoRA alpha
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "dense"],  # phi-2 attention modules
        bias="none",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def train(
    dataset_path: str,
    output_dir: str,
    model_name: str = "microsoft/phi-2",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 512,
    use_4bit: bool = True,
    save_steps: int = 100,
    logging_steps: int = 10,
):
    """Main training function."""
    print("=" * 50)
    print("MediMind Fine-tuning with LoRA")
    print("=" * 50)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name, use_4bit=use_4bit)
    
    # Load and preprocess dataset
    dataset = create_dataset(dataset_path, tokenizer, max_length)
    
    # Split into train/validation (90/10)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Effective batch size = batch_size * 4
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),  # Use mixed precision on GPU
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=save_steps,
        evaluation_strategy="steps",
        save_total_limit=3,  # Keep only last 3 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=100,
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("\nTraining complete!")
    print(f"Model saved to: {output_dir}")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune phi-2 for MediMind")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="app/data/dataset.jsonl",
        help="Path to dataset JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/medimind-phi2-lora",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/phi-2",
        help="Base model name",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization (use full precision)",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps",
    )
    
    args = parser.parse_args()
    
    # Resolve dataset path relative to backend directory
    dataset_path = Path(__file__).parent.parent.parent / args.dataset_path
    if not dataset_path.exists():
        # Try alternative path
        alt_path = Path(__file__).parent.parent / "data" / "dataset.jsonl"
        if alt_path.exists():
            dataset_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {dataset_path} or {alt_path}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train(
        dataset_path=str(dataset_path),
        output_dir=str(output_dir),
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        use_4bit=not args.no_4bit,
        save_steps=args.save_steps,
    )


if __name__ == "__main__":
    main()

