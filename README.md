# Consumer Complaint Classifier

A small text classification project based on the supplied BERT_Model.ipynb and model archive. It uses DistilBERT, a simple FastAPI backend, and plain HTML, CSS, and JavaScript.

## Project files

- train.py: fine-tunes DistilBERT and saves the model and label names.
- app.py: loads the saved model locally and exposes prediction endpoints.
- model/: checkpoint, tokenizer, and labels used by the API.
- data/complaints.csv: supplied dataset for training.
- frontend/: browser interface with plain HTML, CSS, and JavaScript.
- notebooks/BERT_Model.ipynb: supplied notebook, preserved as received.

## Run the included model

Use Python 3.10 or newer. From the BERT_Project folder:

    python -m venv .venv

Activate it in Windows PowerShell:

    .venv\Scripts\Activate.ps1

Or activate it on macOS/Linux:

    source .venv/bin/activate

Then install and run:

    python -m pip install -r requirements.txt
    python -m uvicorn app:app --reload

Open http://127.0.0.1:8000. The API loads its checkpoint from model/ and uses local files for inference. The first installation downloads Python packages, but inference does not download a model. Visit /api/health to check readiness or /docs for the API interface.

## Train with complaints.csv

The CSV must include the notebook's actual columns:

- Consumer complaint narrative: complaint text
- Issue: category

Run python train.py. It defaults to data/complaints.csv; enter a different path at the prompt if needed. The script follows the notebook's preprocessing: remove missing and duplicate rows, retain the nine most frequent Issue values, fit LabelEncoder, and make stratified train/validation/test splits with random state 42. It tokenizes to 256 tokens and trains for four epochs. Training writes the new weights, tokenizer, and exact class order to model/. The API then uses those matching files.

The supplied complaints.csv has 15,062 rows and the expected two columns. Its top nine Issue categories differ from the categories recorded in the archived checkpoint. The included model/labels.json matches that checkpoint's archived LabelEncoder; running train.py replaces it with the class order learned from the supplied CSV. No results are claimed for training that has not been run.

The first training run downloads the pretrained distilbert-base-uncased model. GPU training is faster; CPU training may take a long time. Install a PyTorch build that matches your CUDA setup if you plan to train on an NVIDIA GPU.

## Prediction endpoint

POST /api/predict accepts JSON such as {"text": "I was charged a fee I do not recognize."}. It returns the three highest scoring issue categories and confidence scores. These scores are model estimates, not guaranteed conclusions.

## Model details

The notebook uses DistilBERT, a 256-token limit, the columns documented above, a nine-category selection, and top-three predictions. The uploaded archive contains model weights and its label encoder. The encoder's class names are stored as readable JSON so the API does not need to load a Python pickle.
