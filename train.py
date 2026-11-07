"""Train the complaint classifier using the columns in BERT_Model.ipynb."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
BASE_MODEL = "distilbert-base-uncased"
TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Issue"
MAX_LENGTH = 256
TOP_N = 9


def main():
    default_csv = ROOT / "data" / "complaints.csv"
    prompt = "Path to complaints.csv [" + str(default_csv) + "]: "
    entered_path = input(prompt).strip().strip('"')
    csv_path = Path(entered_path) if entered_path else default_csv
    if not csv_path.is_file():
        raise FileNotFoundError("CSV file not found: " + str(csv_path))

    data = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    missing = [name for name in (TEXT_COLUMN, LABEL_COLUMN) if name not in data.columns]
    if missing:
        raise ValueError("CSV is missing required column(s): " + ", ".join(missing))

    data = data[[TEXT_COLUMN, LABEL_COLUMN]].dropna().drop_duplicates().copy()
    data[TEXT_COLUMN] = data[TEXT_COLUMN].astype(str).str.strip()
    data[LABEL_COLUMN] = data[LABEL_COLUMN].astype(str).str.strip()
    data = data[(data[TEXT_COLUMN] != "") & (data[LABEL_COLUMN] != "")]
    top_labels = data[LABEL_COLUMN].value_counts().head(TOP_N).index
    data = data[data[LABEL_COLUMN].isin(top_labels)].reset_index(drop=True)

    encoder = LabelEncoder()
    data["labels"] = encoder.fit_transform(data[LABEL_COLUMN])
    train_data, test_data = train_test_split(
        data, test_size=0.2, stratify=data["labels"], random_state=42, shuffle=True
    )
    train_data, validation_data = train_test_split(
        train_data, test_size=0.1, stratify=train_data["labels"], random_state=42
    )
    label_names = [str(label) for label in encoder.classes_]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def make_dataset(frame):
        dataset = Dataset.from_pandas(
            frame[[TEXT_COLUMN, "labels"]].reset_index(drop=True), preserve_index=False
        )

        def tokenize(batch):
            return tokenizer(
                batch[TEXT_COLUMN], truncation=True, padding="max_length",
                max_length=MAX_LENGTH,
            )

        dataset = dataset.map(tokenize, batched=True)
        dataset = dataset.remove_columns([TEXT_COLUMN])
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return dataset

    train_ds = make_dataset(train_data)
    validation_ds = make_dataset(validation_data)
    test_ds = make_dataset(test_data)
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(label_names), id2label=id2label, label2id=label2id
    )

    def metrics(result):
        predicted = np.argmax(result.predictions, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            result.label_ids, predicted, average="macro", zero_division=0
        )
        return {
            "accuracy": accuracy_score(result.label_ids, predicted),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    arguments = TrainingArguments(
        output_dir=str(ROOT / "results"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        compute_metrics=metrics,
    )
    print("Training rows:", len(train_data), "Validation rows:", len(validation_data))
    print("Test rows:", len(test_data), "Issue categories:", ", ".join(label_names))
    trainer.train()
    print("Held-out test metrics:", trainer.evaluate(test_ds))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    (MODEL_DIR / "labels.json").write_text(
        json.dumps(label_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Saved model, tokenizer, and labels in", MODEL_DIR)


if __name__ == "__main__":
    main()
