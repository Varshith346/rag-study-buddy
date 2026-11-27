# RAG Study Buddy

Ask questions about your own PDF notes and generate quizzes from them.

Notes are split into chunks, embedded with `all-MiniLM-L6-v2`, and stored in ChromaDB. Each question retrieves the most relevant chunks and the LLM answers from them, citing file and page. Quizzes use Pydantic structured output.

Supports Groq, OpenAI, Gemini, Claude and Ollama via LangChain's `init_chat_model`.

## Setup

```
pip install -r requirements.txt
cp .env.example .env    # set LLM_PROVIDER and your API key
streamlit run app.py
```

## Files

- `rag.py` – indexing, retrieval, answering, quiz generation
- `app.py` – Streamlit UI
- `llm.py` – model selection
