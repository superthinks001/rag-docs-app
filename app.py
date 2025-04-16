import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import docx2txt
import base64
from typing import List
from pathlib import Path

st.set_page_config(page_title="📄 RAG Docs App", layout="wide")
st.title("📄 RAG Docs App – Summarize, Ask & Understand Your Files")

# --- Upload Section ---
uploaded_files = st.file_uploader("📂 Upload PDFs, DOCX, or Excel files", type=["pdf", "docx", "xlsx", "csv"], accept_multiple_files=True)

# --- Display File Previews ---
def preview_file(file, filetype):
    if filetype == "pdf":
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
            return text[:800] + "..." if len(text) > 800 else text
    elif filetype == "docx":
        return docx2txt.process(file)[:800]
    elif filetype in ["csv", "xlsx"]:
        df = pd.read_csv(file) if filetype == "csv" else pd.read_excel(file)
        return df.head()

if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"🗂 File: {file.name}")
        ext = Path(file.name).suffix.lower()[1:]

        try:
            preview = preview_file(file, ext)
            if isinstance(preview, pd.DataFrame):
                st.dataframe(preview)
            else:
                st.text(preview)
        except Exception as e:
            st.error(f"Failed to preview {file.name}: {e}")
