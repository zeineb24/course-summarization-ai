# app/pdf_extract.py
from __future__ import annotations

from typing import Optional, Tuple
import fitz  # PyMuPDF


def extract_text_from_pdf(
    pdf_path: str,
    max_pages: Optional[int] = None,
) -> Tuple[str, int]:
    """
    Extracts text from a PDF file.

    Returns:
      (full_text, pages_read)

    Beginner note:
    - A PDF is made of pages.
    - We read each page and grab the text.
    - Some PDFs are scanned images, so they may return little/no text (OCR is a later upgrade).
    """
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    pages_to_read = total_pages if max_pages is None else min(max_pages, total_pages)

    all_text = []
    for i in range(pages_to_read):
        page = doc.load_page(i)
        page_text = page.get_text("text")  # simple extraction
        if page_text:
            all_text.append(page_text)

    doc.close()
    return "\n".join(all_text).strip(), pages_to_read
