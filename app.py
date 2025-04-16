import streamlit as st
import pandas as pd
import PyMuPDF
fitz = PyMuPDF
import docx2txt
from pathlib import Path
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI

# Use Groq for summarization
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Use HuggingFace for embedding + OpenAI-compatible LangChain model for Q&A
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm_for_qa = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

st.set_page_config(page_title="📄 RAG Docs App", layout="wide")
st.title("📄 RAG Docs App – Summarize, Ask & Understand Your Files")

uploaded_files = st.file_uploader(
    "📂 Upload PDFs, DOCX, or Excel files",
    type=["pdf", "docx", "xlsx", "csv"],
    accept_multiple_files=True
)

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

def get_qa_chain_from_text(text: str):
    docs = [Document(page_content=text)]
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    db = FAISS.from_documents(split_docs, embedding_model)
    retriever = db.as_retriever()
    chain = RetrievalQA.from_chain_type(llm=llm_for_qa, retriever=retriever)
    return chain

# --- App Logic ---
if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"🗂 File: {file.name}")
        ext = Path(file.name).suffix.lower()[1:]
        full_text = preview_file(file, ext)

        if isinstance(full_text, pd.DataFrame):
            st.dataframe(full_text)
            full_text = full_text.to_markdown(index=False)

        else:
            st.text(full_text[:800] + "..." if len(full_text) > 800 else full_text)
            if len(full_text) > 800:
                with st.expander("🔍 Show full document text"):
                    st.text(full_text)

        if isinstance(full_text, str) and len(full_text) > 20:
            if st.button(f"Summarize {file.name}"):
                with st.spinner("Summarizing..."):
                    summary = generate_summary(full_text)
                    st.markdown("### 🧠 Summary")
                    st.write(summary)

            question = st.text_input(f"❓ Ask a question about {file.name}", key=file.name)
            if question:
                with st.spinner("Finding answer..."):
                    qa_chain = get_qa_chain_from_text(full_text)
                    answer = qa_chain.run(question)
                    st.markdown("### 🤖 Answer")
                    st.write(answer)
