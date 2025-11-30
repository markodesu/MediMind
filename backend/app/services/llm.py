import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def generate_response(message):
    inputs = tokenizer(message, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150)
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Simple confidence scoring: average softmax probability of last token
    with torch.no_grad():
        logits = model(**inputs).logits
        last_token_logits = logits[0, -1]
        probs = torch.softmax(last_token_logits, dim=-1)
        confidence = float(torch.max(probs))
    return text, confidence

