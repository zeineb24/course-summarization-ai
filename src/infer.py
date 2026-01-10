# src/infer.py
from __future__ import annotations

from transformers import pipeline

MODEL_NAME = "t5-base"
_summarizer = pipeline("summarization", model=MODEL_NAME)

def summarize_text(text: str, max_length: int = 200, min_length: int = 60) -> str:
    text = (text or "").strip()
    if not text:
        return "Please paste some text to summarize."

    # T5 works better with a task prefix
    prompt = "summarize: " + text

    out = _summarizer(prompt, max_length=max_length, min_length=min_length, do_sample=False)
    return out[0]["summary_text"]
