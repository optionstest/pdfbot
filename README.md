# PDF Q&A Bot (Streamlit)

Upload a PDF, ask questions, and get answers grounded in the document using OpenAI + vector search (FAISS).

## What’s included
- Streamlit UI with chat experience
- PDF text extraction (pypdf)
- Chunking and embeddings
- FAISS vector search
- OpenAI chat model answers using retrieved context

## Setup

### 1) Create environment + install deps
```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
# venv\Scripts\activate

pip install -r requirements.txt
```

### 2) Add your OpenAI API key
Create a file named `.env` in the project root:
```bash
OPENAI_API_KEY=your_key_here
```

### 3) Run the app
```bash
streamlit run app.py
```

## Notes
- If your PDF is scanned (images only), text extraction may return empty. You’d need OCR for that.
- Default model is `gpt-4o-mini` but you can switch from the sidebar.

## Git tips
This repo includes a `.gitignore` that prevents committing your `.env`.
