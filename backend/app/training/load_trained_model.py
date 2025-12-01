"""
Helper script to load and test a trained LoRA model.
Usage: python -m app.training.load_trained_model --model_path ./models/medimind-phi2-lora
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_trained_model(base_model_name: str, lora_path: str):
    """Load base model and apply LoRA adapter."""
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    
    print(f"Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def test_model(model, tokenizer, test_prompt: str = "Human: What is a headache?\nAssistant:"):
    """Test the trained model with a sample prompt."""
    print(f"\nTesting with prompt: {test_prompt}")
    
    inputs = tokenizer(test_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    
    print(f"Response: {response}\n")
    return response


def main():
    parser = argparse.ArgumentParser(description="Load and test trained LoRA model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained LoRA model",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/phi-2",
        help="Base model name",
    )
    parser.add_argument(
        "--test_prompt",
        type=str,
        default="Human: What is a headache?\nAssistant:",
        help="Test prompt",
    )
    
    args = parser.parse_args()
    
    model, tokenizer = load_trained_model(args.base_model, args.model_path)
    test_model(model, tokenizer, args.test_prompt)


if __name__ == "__main__":
    main()

