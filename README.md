# 📚 RAG Study Buddy

Chat with your own lecture notes and quiz yourself on them.
Upload PDFs → ask questions → get answers **grounded in your notes, with page citations** → generate multiple-choice quizzes.

Built with **LangChain**, **ChromaDB**, **Hugging Face embeddings**, **Pydantic** and **Streamlit**.
Works with **any LLM provider**: Groq, OpenAI, Google Gemini, Anthropic Claude, or local models via Ollama.

## ✨ Features
- **Retrieval-Augmented Generation (RAG):** answers come only from your uploaded notes, so less hallucination.
- **Source citations:** every answer shows which file and page it came from.
- **Quiz generator:** structured MCQs created with Pydantic + LLM structured output.
- **Provider-agnostic:** switch LLMs with one line in `.env`.
- **Local embeddings:** `all-MiniLM-L6-v2` runs on your machine, free and with no API key.

## 🏗️ How it works
```
PDFs ──► PyPDFLoader ──► RecursiveCharacterTextSplitter (500 chars, 50 overlap)
                                   │
                                   ▼
                    HuggingFace embeddings ──► ChromaDB (vector store)
                                                     │
Question ──► embed ──► similarity search (top-4) ────┘
                              │
                              ▼
              Prompt (context + question) ──► LLM ──► Answer + sources
```

## 🚀 Run it
```bash
git clone https://github.com/varshith346/rag-study-buddy.git
cd rag-study-buddy
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env         # Windows  (Mac/Linux: cp .env.example .env)
# open .env, set LLM_PROVIDER and paste your API key
streamlit run app.py
```

## 📁 Project structure
| File | What it does |
|---|---|
| `app.py` | Streamlit UI: upload, chat tab, quiz tab |
| `rag.py` | The RAG pipeline: indexing, retrieval, answering, quiz |
| `llm.py` | Picks the LLM provider from `.env` using `init_chat_model` |

## 🔮 Next ideas
- Hybrid search (BM25 + vectors)
- Re-ranking retrieved chunks
- Chat memory for follow-up questions
