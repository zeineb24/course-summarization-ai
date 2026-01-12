# course-summarization-ai
### Fine-Tuned T5 for Long-Document Academic Summarization

This project implements an **end-to-end NLP pipeline** for summarizing long course materials (lecture notes, syllabi, and PDFs) using a **fine-tuned T5 model**.  
It supports **long documents via token-level chunking**, **hierarchical summarization**, and an **interactive Gradio web interface**.

---

## 🚀 Features

- ✅ Fine-tuned **T5-base** on **SciSummNet**
- ✅ Handles **long documents** using tokenizer-aware chunking
- ✅ **Hierarchical summarization** (chunk → merge → final summary)
- ✅ Clean **ROUGE evaluation**
- ✅ Local **Gradio web app**
- ✅ Modular, production-ready codebase

---

## 🧠 Model & Training

- **Base model:** `t5-base`
- **Task:** Abstractive summarization
- **Dataset:** SciSummNet
- **Training setup:**
  - 1 epoch fine-tuning
  - AdamW optimizer
  - Max input length: 512 tokens
  - Mixed precision when available

---

## 📊 Results

Evaluation was performed on a held-out **test set** using ROUGE:

| Metric   | Score |
|--------|-------|
| ROUGE-1 | **0.86** |
| ROUGE-2 | **0.82** |
| ROUGE-L | **0.85** |

These results demonstrate strong abstractive summarization performance for a single-epoch fine-tuned T5-base model.

---

## 🏗️ Architecture Overview

User Input (Text / PDF)
│
▼
Text Extraction
│
▼
Token-Level Chunking
│
▼
Summarize Each Chunk
│
▼
Merge Partial Summaries
│
▼
Final Summary Generation

yaml
Copy code

---

## 📂 Project Structure

course-summarization-ai/
│
├── app/
│ ├── app.py # Gradio interface
│ ├── pdf_extract.py # PDF text extraction
│ └── init.py
│
├── src/
│ ├── chunking.py # Long-document chunking
│ └── infer.py # Inference + summarization logic
│
├── models/
│ └── t5_scisumm_finetuned/ # (not tracked in GitHub)
│
├── requirements.txt
├── README.md
└── .gitignore

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/zeineb24/course-summarization-ai.git
cd course-summarization-ai

```
### 2️⃣ Create a virtual environment (recommended)
```python -m venv .venv
source .venv/bin/activate  # Linux / Mac
.venv\Scripts\activate     # Windows
```
### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```
## 📦 Model Weights

The fine-tuned model (~900 MB) is not stored in this repository.

### To use the application:

Fine-tune the model yourself or

Download the pre-trained weights (provided separately)

Place them in the following directory:

models/t5_scisumm_finetuned/

## ▶️ Run the Application

From the project root:

python -m app.app
```

Then open the URL shown in your terminal (usually http://127.0.0.1:7860).


## ✨ Example Use Cases

Summarizing lecture notes

Condensing academic papers

Generating quick overviews of course materials

Pre-reading support for exams or research

## 🧪 Technologies Used

Python

Hugging Face Transformers

PyTorch

Gradio

ROUGE

SentencePiece

