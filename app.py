import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import docx2txt
from pathlib import Path
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
from groq import Groq
from sentence_transformers import SentenceTransformer

# LangChain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.llms.ollama import Ollama
from langchain.embeddings import HuggingFaceEmbeddings

# 🔑 Groq for summarization
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# 🤖 Ollama local model for Q&A
llm_for_qa = Ollama(model="llama3")

# 🤝 HuggingFace embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Streamlit config
st.set_page_config(page_title="📄 RAG Docs App", layout="wide")
st.title("📄 RAG Docs App – Summarize, Ask & Understand Your Files")

# File uploader
uploaded_files = st.file_uploader(
    "📂 Upload PDFs, DOCX, or Excel files",
    type=["pdf", "docx", "xlsx", "csv"],
    accept_multiple_files=True
)

# OCR for scanned PDFs
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

# Preview file content
def preview_file(file, filetype):
    try:
        if filetype == "pdf":
            file.seek(0)
            with fitz.open(stream=file.read(), filetype="pdf") as doc:
                text = "\n".join(page.get_text() for page in doc)
            if len(text.strip()) < 50:
                file.seek(0)
                text = extract_text_from_scanned_pdf(file)
            return text
        elif filetype == "docx":
            return docx2txt.process(file)
        elif filetype == "csv":
            return pd.read_csv(file)
        elif filetype == "xlsx":
            return pd.read_excel(file)
    except Exception as e:
        return f"❌ Failed to preview: {str(e)}"

# Summarizer using Groq
def generate_summary(text):
    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a document summarizer."},
                {"role": "user", "content": f"Summarize the following document:\n\n{text}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Failed to summarize: {e}"

# LangChain Q&A from one document
def get_qa_chain_from_text(text: str):
    docs = [Document(page_content=text)]
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    db = FAISS.from_documents(split_docs, embedding_model)
    retriever = db.as_retriever()
    chain = RetrievalQA.from_chain_type(llm=llm_for_qa, retriever=retriever)
    return chain

# LangChain Q&A from multiple documents
def get_qa_chain_from_multiple_texts(texts: list):
    docs = [Document(page_content=txt) for txt in texts]
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    db = FAISS.from_documents(split_docs, embedding_model)
    retriever = db.as_retriever()
    chain = RetrievalQA.from_chain_type(llm=llm_for_qa, retriever=retriever)
    return chain

# --- Main App Loop ---
all_texts = []

if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"🗂 File: {file.name}")
        ext = Path(file.name).suffix.lower()[1:]
        full_text = preview_file(file, ext)

        if isinstance(full_text, pd.DataFrame):
            st.dataframe(full_text)
            table_text = full_text.to_markdown(index=False)
            full_text = table_text

        else:
            short_preview = full_text[:800] + "..." if len(full_text) > 800 else full_text
            st.text(short_preview)
            if len(full_text) > 800:
                with st.expander("🔍 Show full document text"):
                    st.text(full_text)

        # Store for multi-file Q&A
        if isinstance(full_text, str) and len(full_text) > 20:
            all_texts.append(full_text)

            # Summarize individual file
            if st.button(f"Summarize {file.name}"):
                with st.spinner("Summarizing..."):
                    summary = generate_summary(full_text)
                    st.markdown("### 🧠 Summary")
                    st.write(summary)

            # Single-file Q&A
            question = st.text_input(f"❓ Ask a question about {file.name}", key=file.name)
            if question:
                with st.spinner("Finding answer..."):
                    qa_chain = get_qa_chain_from_text(full_text)
                    answer = qa_chain.run(question)
                    st.markdown("### 🤖 Answer")
                    st.write(answer)

    # Multi-file Q&A
    if len(all_texts) > 1:
        st.markdown("---")
        st.markdown("## 📚 Ask a Question Using All Documents")
        question_all = st.text_input("❓ Your question across all files", key="multi_file_qa")
        if question_all:
            with st.spinner("Searching across all documents..."):
                qa_chain_multi = get_qa_chain_from_multiple_texts(all_texts)
                answer_multi = qa_chain_multi.run(question_all)
                st.markdown("### 🤖 Multi-Document Answer")
                st.write(answer_multi)
