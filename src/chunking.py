# src/chunking.py
from __future__ import annotations
from typing import List
from transformers import PreTrainedTokenizerBase

def chunk_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int = 700,
    overlap: int = 50,
) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    step = max_tokens - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than max_tokens")

    start = 0
    while start < len(token_ids):
        end = start + max_tokens
        chunk_ids = token_ids[start:end]
        chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk)
        start += step

    return chunks
