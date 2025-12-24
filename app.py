import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -----------------------------
# Streamlit UI Config
# -----------------------------
st.set_page_config(page_title="PDF Q&A Bot", page_icon="📄")
st.title("📄 PDF Q&A Chatbot")

# -----------------------------
# Helper Functions
# -----------------------------
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def split_text(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)


def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(chunks, embedding=embeddings)


# -----------------------------
# File Upload
# -----------------------------
pdf = st.file_uploader("Upload a PDF file", type="pdf")

if pdf:
    text = load_pdf(pdf)
    chunks = split_text(text)
    vectorstore = create_vector_store(chunks)

    query = st.text_input("Ask a question about the PDF")

    if query:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        answer = qa_chain.run(query)

        st.markdown("### ✅ Answer")
        st.write(answer)
