from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"  # lightweight for development

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def generate_response(question: str):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=60)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # dummy confidence: longer outputs mean higher confidence
    confidence = min(len(text) / 50, 1)

    return text, round(confidence, 2)
