from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# -----------------------------
# Load AI Model
# -----------------------------
MODEL_NAME = "davanstrien/MedRoBERTa-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


# -----------------------------
# Request Schema
# -----------------------------
class SymptomRequest(BaseModel):
    symptoms: str


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_symptoms(req: SymptomRequest):
    inputs = tokenizer(req.symptoms, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)

    confidence, prediction = torch.max(probs, dim=1)

    confidence_score = float(confidence.item())
    label_index = prediction.item()

    # -----------------------------
    # Intelligent Behaviour
    # -----------------------------
    # If confidence is low, recommend professional help
    if confidence_score < 0.45:
        return {
            "prediction": "Uncertain",
            "confidence": confidence_score,
            "advice": "Your symptoms are unclear. Please visit the UCA nurse or nearest hospital."
        }

    return {
        "prediction": label_index,
        "confidence": confidence_score,
        "advice": "MediMind suggests this is a common condition. If symptoms worsen, seek medical help."
    }

