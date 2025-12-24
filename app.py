import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from helper import build_vectorstore_from_pdf, answer_question

load_dotenv()

st.set_page_config(page_title="PDF Q&A Bot", page_icon="📄", layout="centered")
st.title("Ask Questions from your PDF 📄🤖")
st.caption("Upload a PDF, then ask questions. The app retrieves relevant passages and answers using OpenAI.")

api_key = os.getenv("OPENAI_API_KEY", "").strip()

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    k = st.slider("How many chunks to retrieve (k)", 2, 10, 4, 1)
    st.divider()
    st.write("API Key")
    if api_key:
        st.success("OPENAI_API_KEY found in environment ✅")
    else:
        st.warning("No OPENAI_API_KEY found. Add it to a .env file or your environment variables.")

pdf = st.file_uploader("Upload a PDF", type=["pdf"])

# Session state: keep vectorstore for the current uploaded PDF
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_fingerprint" not in st.session_state:
    st.session_state.pdf_fingerprint = None
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of (role, text)

def fingerprint(file_bytes: bytes) -> str:
    # small fingerprint for caching within a session
    import hashlib
    return hashlib.sha256(file_bytes).hexdigest()

if pdf is not None:
    file_bytes = pdf.getvalue()
    fp = fingerprint(file_bytes)

    if st.session_state.pdf_fingerprint != fp:
        # New PDF uploaded: rebuild index
        st.session_state.pdf_fingerprint = fp
        st.session_state.vectorstore = None
        st.session_state.chat = []

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        with st.spinner("Reading PDF and building search index..."):
            try:
                st.session_state.vectorstore = build_vectorstore_from_pdf(tmp_path, api_key=api_key)
                st.success("PDF indexed. Ask away!")
            except Exception as e:
                st.error(f"Failed to process PDF: {e}")
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

if st.session_state.vectorstore is None:
    st.info("Upload a PDF to start.")
    st.stop()

st.subheader("Chat")

# Render chat history
for role, content in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(content)

user_q = st.chat_input("Ask a question about the PDF...")
if user_q:
    st.session_state.chat.append(("user", user_q))
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = answer_question(
                    query=user_q,
                    vectorstore=st.session_state.vectorstore,
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    k=k,
                )
                st.markdown(answer)

                if sources:
                    with st.expander("Sources used"):
                        for i, s in enumerate(sources, start=1):
                            st.markdown(f"**{i}.** {s}")
            except Exception as e:
                st.error(f"Error: {e}")
                answer = None

    if answer is not None:
        st.session_state.chat.append(("assistant", answer))
