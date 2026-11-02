# Consumer Complaint Classification using DistilBERT

A multi-class text classification application that fine-tunes DistilBERT to predict the issue category of consumer financial complaints. It uses Hugging Face Transformers for model training, FastAPI for inference and a simple HTML, CSS and JavaScript frontend.

## Features

- Fine-tuning of `distilbert-base-uncased` for complaint classification.
- Text preprocessing, duplicate removal and conflicting-label filtering.
- Dynamic category selection based on minimum class frequency.
- Stratified train-test splitting.
- Class-weighted cross-entropy loss to address class imbalance.
- FastAPI backend for model inference.
- Top-3 predicted complaint categories with confidence scores.
- Browser-based interface for submitting complaints and viewing predictions.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| NLP Model | DistilBERT |
| Deep Learning | PyTorch |
| Model Framework | Hugging Face Transformers |
| Data Processing | Pandas, NumPy |
| Evaluation | Scikit-learn |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |

## Project Structure

```text
Customer-Complaints-Classification/
│
├── dataset/
│   └── complaints.csv
│
├── model/                  # Generated after training
│
├── training_checkpoints/   # Training checkpoints
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── train.py
├── app.py
├── requirements.txt
└── README.md
```

## Dataset and Preprocessing

The training script expects a CSV file at `dataset/complaints.csv` containing:

- `Consumer complaint narrative`: Complaint text.
- `Issue`: Target complaint category.

The preprocessing pipeline performs:

1. Removal of missing values.
2. Text normalization for identifying duplicate complaints.
3. Filtering of complaints containing fewer than five normalized words.
4. Removal of identical normalized text assigned to conflicting categories.
5. Removal of duplicate normalized complaints.
6. Filtering of categories with fewer than 100 samples.

The number of classification categories is determined dynamically from the remaining dataset after preprocessing; it is not fixed in the code.

## Model Training

The project uses `distilbert-base-uncased` as the pretrained model and fine-tunes it for sequence classification.

**Training configuration**

| Parameter | Value |
|---|---|
| Pretrained Model | distilbert-base-uncased |
| Maximum Sequence Length | 128 |
| Batch Size | 8 |
| Epochs | 2 |
| Learning Rate | 3e-5 |
| Train-Test Split | 80:20 |
| Random Seed | 42 |
| Minimum Class Samples | 100 |

Additional implementation details:

- LabelEncoder converts complaint categories into numerical labels.
- Stratified splitting preserves category proportions.
- Inverse-square-root class weights are used in the cross-entropy loss.
- Accuracy and macro-F1 are calculated during evaluation.
- The trained model, tokenizer and category labels are saved in the `model/` directory.

**Evaluation note:** The current implementation uses the 20% test split as the Trainer evaluation dataset for checkpoint selection and also for final evaluation. Consequently, the reported test metrics are not an independent estimate of generalization performance.

## Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/techno-lunatic/Customer-Complaints-Classification.git

cd Customer-Complaints-Classification
```

**2. Create and activate a virtual environment**

```bash
python -m venv bert_venv

# Git Bash (Windows)
source bert_venv/Scripts/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Prepare the dataset**

Place `complaints.csv` inside the `dataset/` directory.

**5. Train the model**

```bash
python train.py
```

The script fine-tunes DistilBERT and saves the trained model, tokenizer and category labels in the `model/` directory. The first run downloads the pretrained model from Hugging Face.

A GPU is recommended for faster training.

## Running the Application

After training completes, start the FastAPI server from the project root:

```bash
uvicorn app:app --reload
```

Open the application in your browser:

```text
http://127.0.0.1:8000
```

The backend loads the saved model and tokenizer during application startup.

## API Endpoints

### 1. Health Check

**Endpoint:** `GET /api/health`

Returns the model loading status and the device being used.

Example response:

```json
{
    "model_loaded": true,
    "device": "cpu"
}
```

### 2. Predict Complaint Category

**Endpoint:** `POST /api/predict`

Accepts complaint text in JSON format.

**Request:**

```json
{
    "text": "I was charged an incorrect amount on my credit card statement and the bank has not resolved my complaint."
}
```

**Response:**

```json
{
    "predictions": [
        {
            "category": "Example Category",
            "confidence": 85.42
        },
        {
            "category": "Another Category",
            "confidence": 9.31
        }
    ]
}
```

The response contains up to three predicted categories, depending on the number of trained classes, with their corresponding model confidence scores.

**Input validation:**
- Complaints containing fewer than five words return HTTP 400.
- If the trained model is unavailable, the endpoint returns HTTP 503.

## Inference Workflow

1. The user enters a complaint through the browser interface.
2. JavaScript sends the complaint to the FastAPI `/api/predict` endpoint.
3. The backend tokenizes the input using the saved DistilBERT tokenizer.
4. The trained model generates class probabilities.
5. The top three predictions and their confidence scores are returned as JSON.
6. JavaScript displays the predictions in the frontend.
