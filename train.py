import re
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# -------------------- Configuration --------------------
SEED = 42
DATA_PATH = "dataset/complaints.csv"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "model"
TEXT_COL = "Consumer complaint narrative"
LABEL_COL = "Issue"

MIN_WORDS = 5
MIN_CLASS_COUNT = 100  # Increase to 300 to match the reference notebook more closely
MAX_LENGTH = 128       # 256 is more accurate for long complaints, but uses more memory
EPOCHS = 2             # Start with 2; increase to 3-4 if time/GPU permits
BATCH_SIZE = 8         # Use 16 if your GPU has enough memory
LEARNING_RATE = 3e-5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalise(text):
    """Used only to identify near-duplicate complaints."""
    text = str(text).lower()
    text = re.sub(r"x{2,}", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class ComplaintDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: torch.tensor(value[index], dtype=torch.long)
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return item


def main():
    set_seed(SEED)
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, usecols=[TEXT_COL, LABEL_COL])
    df = df.dropna().copy()
    df["text"] = df[TEXT_COL].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["issue"] = df[LABEL_COL].astype(str).str.strip()
    df["normalised"] = df["text"].map(normalise)

    # Remove short complaints.
    df = df[df["normalised"].str.split().str.len() >= MIN_WORDS]

    # Remove identical text assigned to conflicting labels.
    label_counts = df.groupby("normalised")["issue"].nunique()
    conflicting_texts = label_counts[label_counts > 1].index
    df = df[~df["normalised"].isin(conflicting_texts)]

    # Keep one copy of each near-duplicate complaint.
    df = df.drop_duplicates("normalised")

    # Keep categories with enough examples for a useful first model.
    counts = df["issue"].value_counts()
    selected_classes = counts[counts >= MIN_CLASS_COUNT].index
    df = df[df["issue"].isin(selected_classes)].reset_index(drop=True)

    print(f"Usable complaints: {len(df)}")
    print(f"Number of categories: {df['issue'].nunique()}")
    print(df["issue"].value_counts())

    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df["issue"])
    class_names = encoder.classes_.tolist()

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(class_names),
        id2label={i: name for i, name in enumerate(class_names)},
        label2id={name: i for i, name in enumerate(class_names)},
    )

    train_dataset = ComplaintDataset(train_df["text"], train_df["label"], tokenizer)
    test_dataset = ComplaintDataset(test_df["text"], test_df["label"], tokenizer)

    # Simple class weights help reduce the effect of category imbalance.
    class_counts = np.bincount(train_df["label"], minlength=len(class_names))
    weights = 1.0 / np.sqrt(class_counts)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_function = torch.nn.CrossEntropyLoss(
                weight=class_weights.to(outputs.logits.device)
            )
            loss = loss_function(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro"),
        }

    training_args = TrainingArguments(
        output_dir="training_checkpoints",
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=SEED,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\nStarting fine-tuning. A GPU is strongly recommended.")
    trainer.train()

    print("\nTest-set evaluation:")
    predictions = trainer.predict(test_dataset)
    predicted_ids = np.argmax(predictions.predictions, axis=1)
    print(classification_report(
        test_df["label"], predicted_ids,
        target_names=class_names, zero_division=0
    ))

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    with open(Path(OUTPUT_DIR) / "labels.json", "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    print(f"\nModel saved in: {OUTPUT_DIR}/")
    print("Next: run `streamlit run app.py` to use the frontend.")


if __name__ == "__main__":
    main()
