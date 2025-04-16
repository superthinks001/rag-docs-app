import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import docx2txt
from pathlib import Path
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
from groq import Groq

# LangChain components
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq  # ✅ Corrected import

# 🧠 LLM and Embeddings Setup
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

llm_for_qa = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0,
    api_key=st.secrets["GROQ_API_KEY"]  # ✅ Correct key
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 🎨 UI Config
st.set_page_config(page_title="📄 RAG Docs App", layout="wide")
st.title("📄 RAG Docs App – Summarize, Ask & Understand Your Files")

uploaded_files = st.file_uploader(
    "Upload PDFs, DOCX, Excel, or CSV files",
    type=["pdf", "docx", "xlsx", "csv"],
    accept_multiple_files=True
)

# 📜 Extract text from scanned PDFs
def extract_text_from_scanned_pdf(file):
    try:
        images = convert_from_bytes(file.read())
        return "\n".join([pytesseract.image_to_string(img) for img in images])
    except Exception as e:
        return f"❌ OCR failed: {e}"

# 📂 Preview file content
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
        return f"❌ Preview error: {e}"

# 🧠 Generate summary with Groq LLM
def generate_summary(text):
    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful document summarizer."},
                {"role": "user", "content": f"Summarize the following:\n\n{text}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Summarization error: {e}"

# 💬 Get Q&A chain with FAISS + LangChain
def get_qa_chain(text):
    documents = [Document(page_content=text)]
    chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
    vector_store = FAISS.from_documents(chunks, embedding_model)
    retriever = vector_store.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm_for_qa, retriever=retriever)

# 🚀 Main app logic
if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"📄 File: {file.name}")
        ext = Path(file.name).suffix.lower()[1:]
        full_text = preview_file(file, ext)

        # If it's a table, convert to markdown for Q&A
        if isinstance(full_text, pd.DataFrame):
            st.dataframe(full_text)
            full_text = full_text.to_markdown(index=False)
        else:
            st.text(full_text[:800] + "..." if len(full_text) > 800 else full_text)
            if len(full_text) > 800:
                with st.expander("🔍 Full Text"):
                    st.text(full_text)

        # Ask & Summarize
        if isinstance(full_text, str) and len(full_text) > 20:
            if st.button(f"🧠 Summarize {file.name}"):
                with st.spinner("Summarizing..."):
                    summary = generate_summary(full_text)
                    st.markdown("### 📌 Summary")
                    st.write(summary)

            question = st.text_input(f"❓ Ask a question about {file.name}", key=file.name)
            if question:
                with st.spinner("Searching..."):
                    chain = get_qa_chain(full_text)
                    answer = chain.run(question)
                    st.markdown("### 🤖 Answer")
                    st.write(answer)
