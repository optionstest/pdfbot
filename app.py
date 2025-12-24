import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


st.set_page_config(page_title="PDF Q&A Bot", page_icon="📄")
st.title("📄 PDF Q&A Chatbot")

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
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.from_texts(chunks, embedding=embeddings)

pdf = st.file_uploader("Upload a PDF file", type="pdf")

if pdf:
    text = load_pdf(pdf)
    chunks = split_text(text)
    vectorstore = create_vector_store(chunks)

    query = st.text_input("Ask a question about the PDF")

    if query:
        llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )
        answer = qa_chain.run(query)
        st.write("### Answer")
        st.write(answer)
