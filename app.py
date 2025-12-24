import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="PDF Q&A Bot", page_icon="📄")
st.title("📄 PDF Q&A Chatbot (Modern LangChain)")

# Helper functions
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

# File uploader
pdf = st.file_uploader("Upload a PDF file", type="pdf")

if pdf:
    with st.spinner("📖 Reading PDF..."):
        text = load_pdf(pdf)
        chunks = split_text(text)
    with st.spinner("🔎 Creating FAISS vector store..."):
        vectorstore = create_vector_store(chunks)
    st.success("PDF indexed successfully!")

    query = st.text_input("Ask a question about the PDF")

    if query:
        with st.spinner("🤖 Generating answer..."):
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                openai_api_key=OPENAI_API_KEY
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )
            answer = qa_chain.run(query)
        st.markdown("### ✅ Answer")
        st.write(answer)
