"""The RAG pipeline: load -> split -> embed -> store -> retrieve -> generate.

Every function here is small on purpose. Read it top to bottom once and you
understand the whole project.
"""
import os
from functools import lru_cache

from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from llm import get_llm

DB_DIR = "db"                 # Chroma saves the vectors here (on disk)
COLLECTION = "study_notes"
CHUNK_SIZE = 500              # characters per chunk
CHUNK_OVERLAP = 50            # characters shared between neighbouring chunks


# ---------- 1. Embeddings + vector store ----------

@lru_cache  # load the embedding model only once (it's ~90 MB)
def get_embeddings():
    # Runs locally on your laptop: free and no API key.
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=DB_DIR,
    )


# ---------- 2. Indexing (load -> split -> embed -> store) ----------

def build_index(pdf_paths: list[str]) -> tuple[int, int]:
    """Read PDFs, split them into chunks and store their embeddings.
    Returns (number_of_pages, number_of_chunks)."""
    pages = []
    for path in pdf_paths:
        pages.extend(PyPDFLoader(path).load())  # one Document per page

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)

    store = get_store()
    store.reset_collection()      # start fresh so old notes don't mix in
    store.add_documents(chunks)   # embeds every chunk and saves it
    return len(pages), len(chunks)


def has_index() -> bool:
    return os.path.isdir(DB_DIR) and get_store()._collection.count() > 0


# ---------- 3. Retrieval helpers ----------

def format_docs(docs) -> str:
    """Turn retrieved chunks into one context string, each tagged with its source."""
    parts = []
    for d in docs:
        src = os.path.basename(d.metadata.get("source", "notes"))
        page = d.metadata.get("page", 0) + 1  # PyPDF pages start at 0
        parts.append(f"[{src} p.{page}]\n{d.page_content}")
    return "\n\n".join(parts)


def sources_of(docs) -> list[str]:
    seen = []
    for d in docs:
        tag = f"{os.path.basename(d.metadata.get('source', 'notes'))} p.{d.metadata.get('page', 0) + 1}"
        if tag not in seen:
            seen.append(tag)
    return seen


# ---------- 4. Question answering (retrieve -> generate) ----------

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a friendly study assistant. Answer the student's question using ONLY the context "
     "from their notes below. If the answer is not in the context, say: "
     "\"I couldn't find that in your notes.\" Keep answers clear and simple.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def ask(question: str, k: int = 4) -> tuple[str, list[str]]:
    docs = get_store().similarity_search(question, k=k)   # RETRIEVE
    chain = ANSWER_PROMPT | get_llm()                      # prompt -> LLM (LCEL pipe)
    reply = chain.invoke({"context": format_docs(docs), "question": question})  # GENERATE
    return reply.content, sources_of(docs)


# ---------- 5. Quiz generation (structured output with Pydantic) ----------

class QuizQuestion(BaseModel):
    question: str = Field(description="The question text")
    options: list[str] = Field(description="Exactly 4 answer options")
    answer_index: int = Field(description="Index (0-3) of the correct option")
    explanation: str = Field(description="One sentence on why the answer is correct")


class Quiz(BaseModel):
    questions: list[QuizQuestion]


def make_quiz(topic: str, n: int = 5) -> Quiz:
    docs = get_store().similarity_search(topic, k=6)
    quiz_llm = get_llm(temperature=0.3).with_structured_output(Quiz)
    prompt = (
        f"Create {n} multiple-choice questions about '{topic}' using ONLY the study material below. "
        "Each question must have exactly 4 options and exactly one correct answer.\n\n"
        f"{format_docs(docs)}"
    )
    quiz = quiz_llm.invoke(prompt)
    # Never trust the LLM blindly: drop malformed questions.
    quiz.questions = [q for q in quiz.questions if len(q.options) == 4 and 0 <= q.answer_index < 4]
    return quiz
