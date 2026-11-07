import json
import traceback
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_FOLDER = Path(__file__).resolve().parent
MODEL_FOLDER = PROJECT_FOLDER / "model"
FRONTEND_FOLDER = PROJECT_FOLDER / "frontend"

app = FastAPI()
app.mount("/static", StaticFiles(directory=FRONTEND_FOLDER), name="static")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_FOLDER,
    local_files_only=True,
)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_FOLDER,
    local_files_only=True,
)
model.to(device)
model.eval()

with open(MODEL_FOLDER / "labels.json", encoding="utf-8") as file:
    labels = json.load(file)


class Complaint(BaseModel):
    text: str


@app.get("/")
def home():
    return FileResponse(FRONTEND_FOLDER / "index.html")


@app.post("/api/predict")
def predict(complaint: Complaint):
    try:
        text = complaint.text.strip()

        if not text:
            raise HTTPException(
                status_code=400,
                detail="Please enter a complaint.",
            )

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(device)

        # DistilBERT does not use token_type_ids.
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            scores = model(**inputs).logits[0].softmax(dim=0)

        best_scores, best_indexes = scores.topk(min(3, len(labels)))

        predictions = []
        for score, index in zip(
            best_scores.tolist(),
            best_indexes.tolist(),
        ):
            predictions.append({
                "issue": labels[index],
                "confidence": round(score, 4),
            })

        return {"predictions": predictions}

    except HTTPException:
        raise
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error