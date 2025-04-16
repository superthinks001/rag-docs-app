import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import docx2txt
import base64
from pathlib import Path
from typing import List

# OCR-related imports
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract

st.set_page_config(page_title="📄 RAG Docs App", layout="wide")
st.title("📄 RAG Docs App – Summarize, Ask & Understand Your Files")

# --- Upload Section ---
uploaded_files = st.file_uploader("📂 Upload PDFs, DOCX, or Excel files", type=["pdf", "docx", "xlsx", "csv"], accept_multiple_files=True)

# --- OCR Helper for Scanned PDFs ---
def extract_text_from_scanned_pdf(file):
    try:
        images = convert_from_bytes(file.read())
        text = ""
        for i, img in enumerate(images):
            page_text = pytesseract.image_to_string(img)
            text += f"\n\n--- Page {i + 1} ---\n\n{page_text}"
        return text.strip()
    except Exception as e:
        return f"❌ OCR failed: {str(e)}"

# --- Display File Previews ---
def preview_file(file, filetype):
    try:
        if filetype == "pdf":
            # Try normal extraction
            file.seek(0)
            with fitz.open(stream=file.read(), filetype="pdf") as doc:
                text = "\n".join(page.get_text() for page in doc)
            if len(text.strip()) < 50:
                # If too little text, try OCR
                file.seek(0)
                text = extract_text_from_scanned_pdf(file)
            return text[:800] + "..." if len(text) > 800 else text

        elif filetype == "docx":
            return docx2txt.process(file)[:800]

        elif filetype == "csv":
            return pd.read_csv(file).head()

        elif filetype == "xlsx":
            return pd.read_excel(file).head()

    except Exception as e:
        return f"❌ Failed to preview: {str(e)}"

# --- Show Previews ---
if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"🗂 File: {file.name}")
        ext = Path(file.name).suffix.lower()[1:]
        preview = preview_file(file, ext)

        if isinstance(preview, pd.DataFrame):
            st.dataframe(preview)
        else:
            st.text(preview)
