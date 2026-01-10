# src/infer.py
from __future__ import annotations

from transformers import pipeline
from src.chunking import chunk_text

MODEL_NAME = "t5-base"
_summarizer = pipeline("summarization", model=MODEL_NAME)

def summarize_text(text: str, max_length: int = 200, min_length: int = 60) -> str:
    text = (text or "").strip()
    if not text:
        return "Please paste some text to summarize."

    # Split long text into chunks
    chunks = chunk_text(text, max_tokens=700, overlap=50)

    # If it’s short enough, summarize directly
    if len(chunks) <= 1:
        prompt = "summarize: " + text
        out = _summarizer(prompt, max_length=max_length, min_length=min_length, do_sample=False)
        return out[0]["summary_text"]

    # Summarize each chunk
    partial_summaries = []
    for c in chunks:
        prompt = "summarize: " + c
        out = _summarizer(prompt, max_length=180, min_length=50, do_sample=False)
        partial_summaries.append(out[0]["summary_text"])

    # Merge chunk summaries and summarize again (hierarchical)
    merged = " ".join(partial_summaries)
    final_prompt = "summarize: " + merged
    final = _summarizer(final_prompt, max_length=max_length, min_length=min_length, do_sample=False)
    return final[0]["summary_text"]
