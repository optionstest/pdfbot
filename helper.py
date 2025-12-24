from __future__ import annotations

from typing import List, Tuple

from pypdf import PdfReader

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_pdf_text(file_path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)
    parts: List[str] = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt:
            parts.append(txt)
    return "\n".join(parts).strip()


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def build_vectorstore_from_pdf(file_path: str, api_key: str) -> FAISS:
    """Build a FAISS vector store from the contents of a PDF."""
    text = load_pdf_text(file_path)
    if not text:
        raise ValueError("No text could be extracted from this PDF. If it's scanned, try a text-based PDF or add OCR.")

    chunks = chunk_text(text)
    docs = [Document(page_content=c) for c in chunks]

    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore


def answer_question(
    query: str,
    vectorstore: FAISS,
    api_key: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    k: int = 4,
) -> Tuple[str, List[str]]:
    """Retrieve relevant chunks and answer using a chat model. Returns (answer, sources)."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([d.page_content for d in docs]).strip()

    prompt = (
        "You are a helpful assistant answering questions about a PDF.\n\n"
        "Rules:\n"
        "1) Use ONLY the provided context to answer.\n"
        "2) If the answer is not in the context, say you don't know.\n"
        "3) Keep the answer clear and concise.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    llm = ChatOpenAI(api_key=api_key, model=model, temperature=temperature)
    resp = llm.invoke(prompt)
    answer = getattr(resp, "content", str(resp)).strip()

    sources = [d.page_content[:300].replace("\n", " ").strip() + ("..." if len(d.page_content) > 300 else "") for d in docs]
    return answer, sources
