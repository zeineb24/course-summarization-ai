# app/app.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr

from src.infer import summarize_text
from app.pdf_extract import extract_text_from_pdf



def summarize_from_text(user_text: str, max_length: int, min_length: int) -> str:
    return summarize_text(user_text, max_length=max_length, min_length=min_length)


def summarize_from_pdf(pdf_file, max_pages: int, max_length: int, min_length: int) -> str:
    """
    pdf_file comes from Gradio upload.
    It can be:
      - a temporary file object
      - or a path string
    We'll handle both safely.
    """
    if pdf_file is None:
        return "Please upload a PDF first."

    pdf_path = pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file)

    if not os.path.exists(pdf_path):
        return f"Could not find the uploaded PDF file at: {pdf_path}"

    text, pages_read = extract_text_from_pdf(pdf_path, max_pages=max_pages)

    if not text.strip():
        return (
            f"No text was extracted from the PDF (pages read: {pages_read}).\n\n"
            "This usually means the PDF is scanned images. OCR support can be added as an upgrade."
        )

    return summarize_text(text, max_length=max_length, min_length=min_length)


with gr.Blocks(title="Course Summarization AI") as demo:
    gr.Markdown(
        "# 📚 Course Summarization AI\n"
        "Summarize **course notes**, **research papers**, or **PDF documents** using a T5 transformer.\n\n"
        "✅ Paste text OR upload a PDF.\n"
    )

    with gr.Tab("Summarize Text"):
        user_text = gr.Textbox(
            label="Paste your text here",
            lines=12,
            placeholder="Paste a long section of notes, an article, or a paper..."
        )

        with gr.Row():
            min_len = gr.Slider(20, 200, value=60, step=5, label="Min summary length")
            max_len = gr.Slider(50, 400, value=200, step=10, label="Max summary length")

        text_btn = gr.Button("Summarize Text")
        text_out = gr.Textbox(label="Summary", lines=10)

        text_btn.click(
            fn=summarize_from_text,
            inputs=[user_text, max_len, min_len],
            outputs=[text_out],
        )

    with gr.Tab("Summarize PDF"):
        pdf_file = gr.File(label="Upload a PDF", file_types=[".pdf"])

        with gr.Row():
            max_pages = gr.Slider(1, 40, value=6, step=1, label="Max pages to read (for speed)")
        with gr.Row():
            min_len2 = gr.Slider(20, 200, value=60, step=5, label="Min summary length")
            max_len2 = gr.Slider(50, 400, value=200, step=10, label="Max summary length")

        pdf_btn = gr.Button("Summarize PDF")
        pdf_out = gr.Textbox(label="Summary", lines=12)

        pdf_btn.click(
            fn=summarize_from_pdf,
            inputs=[pdf_file, max_pages, max_len2, min_len2],
            outputs=[pdf_out],
        )

    gr.Markdown(
        "### Notes\n"
        "- This version extracts **text only** from PDFs.\n"
        "- If a PDF is scanned images, you’ll need OCR .\n"
    )

if __name__ == "__main__":
    demo.launch()
