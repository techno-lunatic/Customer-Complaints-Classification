
from pathlib import Path
import json
import torch

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# Project directories
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Consumer Complaint Classifier")

# Model variables
tokenizer = None
model = None
labels = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load trained model when application starts
@app.on_event("startup")
def load_trained_model():
    global tokenizer, model, labels

    if not (MODEL_DIR / "config.json").exists():
        print("WARNING: Trained model not found.")
        return

    print("Loading trained model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    model.to(device)
    model.eval()

    with open(MODEL_DIR / "labels.json", encoding="utf-8") as file:
        labels = json.load(file)

    print("Model loaded successfully!")


# Request format
class ComplaintRequest(BaseModel):
    text: str


# Homepage
@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


# Check model status
@app.get("/api/health")
def health():
    return {
        "model_loaded": model is not None,
        "device": str(device)
    }


# Predict complaint category
@app.post("/api/predict")
def predict(request: ComplaintRequest):

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Trained model not found."
        )

    complaint = request.text.strip()

    if len(complaint.split()) < 5:
        raise HTTPException(
            status_code=400,
            detail="Please enter at least 5 words."
        )

    inputs = tokenizer(
    complaint,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128
)

    # Remove token_type_ids if the tokenizer generates them
    inputs.pop("token_type_ids", None)

    # Move inputs to the model's device
    inputs = inputs.to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)[0]

    top_scores, top_indices = torch.topk(
        probabilities,
        k=min(3, len(labels))
    )

    results = []

    for score, index in zip(
        top_scores.tolist(),
        top_indices.tolist()
    ):
        results.append({
            "category": labels[index],
            "confidence": round(score * 100, 2)
        })

    return {"predictions": results}


# Serve frontend files
app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static"
)
