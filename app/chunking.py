# app/chunking.py
from typing import List
from transformers import AutoTokenizer

MODEL_NAME = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def chunk_text(text: str, max_tokens: int = 700, overlap: int = 50) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    start = 0
    step = max_tokens - overlap
    while start < len(token_ids):
        end = start + max_tokens
        chunk_ids = token_ids[start:end]
        chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk)
        start += step

    return chunks
