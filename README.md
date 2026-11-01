# BERT Consumer Complaint Classifier

A beginner-friendly multi-class text classification project. It fine-tunes DistilBERT to predict the `Issue` category of a consumer financial complaint.

## Frontend and backend

- **HTML** defines the page structure: heading, text area, button and results area.
- **CSS** adds basic spacing, colours and layout.
- **JavaScript** reads the complaint, sends it to the backend using `fetch()`, and displays the returned predictions.
- **FastAPI** is the small Python backend. It receives the complaint, runs the saved DistilBERT model and returns JSON predictions.

FastAPI is needed in this setup because a browser cannot directly execute the Python/PyTorch model. It is not the only possible backend, but it is a simple way to connect JavaScript to Python. We do not need a separate React app or a complex API design.

## Project structure

```text
BERT_Consumer_Complaint_Classifier/
├── dataset/
│   └── complaints.csv
├── model/                 # created after training
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── train.py
├── app.py                 # FastAPI backend
├── requirements.txt
└── README.md
```

## Step 1: Train the model

A GPU is recommended. You can train in Google Colab or on a computer with a supported GPU.

Install packages:

```bash
pip install -r requirements.txt
```

Run training from the project root:

```bash
python train.py
```

The script saves the model and tokenizer in `model/`. The browser application cannot make predictions until this folder has been created.

## Step 2: Start the web application

From the project root, run:

```bash
uvicorn app:app --reload
```

Open this address in your browser:

```text
http://127.0.0.1:8000
```

To stop the server, press `Ctrl + C` in the terminal.

## How one prediction works

1. The user types a complaint in `index.html`.
2. Clicking the button calls the JavaScript function `predictComplaint()`.
3. JavaScript sends the text to `POST /api/predict`.
4. FastAPI tokenizes the text and runs DistilBERT.
5. FastAPI returns the top three categories and confidence scores as JSON.
6. JavaScript displays the results on the page.

## Notes

- `train.py` uses the supplied dataset and expects the columns `Consumer complaint narrative` and `Issue`.
- The starter training configuration filters out categories with fewer than 100 examples, uses a stratified 80/20 split, and trains for 2 epochs.
- The first training run downloads the pretrained DistilBERT model from Hugging Face.
- The confidence values are model scores, not guarantees.
