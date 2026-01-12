# src/infer.py
from __future__ import annotations

import os
from transformers import pipeline, AutoTokenizer
from src.chunking import chunk_text

# Robust absolute path to your local model folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "t5_scisumm_finetuned")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
_summarizer = pipeline(
    "summarization",
    model=MODEL_DIR,
    tokenizer=tokenizer,
)

def summarize_text(text: str, max_length: int = 200, min_length: int = 60) -> str:
    text = (text or "").strip()
    if not text:
        return "Please paste some text to summarize."

    chunks = chunk_text(text, tokenizer=tokenizer, max_tokens=700, overlap=50)

    if len(chunks) <= 1:
        prompt = "summarize: " + text
        out = _summarizer(prompt, max_length=max_length, min_length=min_length, do_sample=False)
        return out[0]["summary_text"]

    partial_summaries = []
    for c in chunks:
        prompt = "summarize: " + c
        out = _summarizer(prompt, max_length=180, min_length=50, do_sample=False)
        partial_summaries.append(out[0]["summary_text"])

    merged = " ".join(partial_summaries)
    final_prompt = "summarize: " + merged
    final = _summarizer(final_prompt, max_length=max_length, min_length=min_length, do_sample=False)
    return final[0]["summary_text"]
