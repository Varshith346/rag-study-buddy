# 🧠 LEARN: RAG Study Buddy, explained simply

Read this once top to bottom, then open the code next to it.

---

## 1. The problem RAG solves
An LLM only knows what it was trained on. It has **never seen your notes**.
If you paste a 200-page PDF into the prompt it's too long, slow and expensive.

**RAG = Retrieval-Augmented Generation**
1. **Retrieve:** find the few pieces of your notes that are relevant to the question
2. **Augment:** paste only those pieces into the prompt
3. **Generate:** the LLM answers using them

## 2. The pipeline, step by step (`rag.py`)

### Step A: Load (`PyPDFLoader`)
Reads a PDF and gives back a list of `Document` objects, one per page.
A `Document` has `page_content` (the text) and `metadata` (like `source` and `page`).

### Step B: Split (`RecursiveCharacterTextSplitter`)
Cuts the pages into **chunks of ~500 characters**.
- *Why small chunks?* Search becomes precise, and we only send relevant text to the LLM.
- *Why `chunk_overlap=50`?* If an important sentence is cut at a chunk boundary, the overlap keeps it whole in at least one chunk.
- *Why "recursive"?* It tries to split on paragraphs first, then lines, then sentences, then words, so chunks stay natural.

### Step C: Embed (`HuggingFaceEmbeddings`)
An **embedding** turns text into a list of numbers (a vector, here 384 numbers) that captures its **meaning**.
Texts with similar meaning get vectors that are close together.
`"dog is barking"` and `"puppy making noise"` → close. `"stock market"` → far away.

### Step D: Store (`Chroma`)
A **vector database** stores the vectors and can quickly find the ones closest to a new vector.
We save it to the `db/` folder so we don't re-embed every time.

### Step E: Retrieve (`similarity_search`)
Your question is embedded too, and Chroma returns the **top-k (4) closest chunks**.
"Closest" is usually measured with **cosine similarity** (the angle between vectors).

### Step F: Generate (`ANSWER_PROMPT | get_llm()`)
The `|` pipe is **LCEL (LangChain Expression Language)**: the prompt's output flows into the LLM.
The system prompt says *"answer ONLY from the context"*. That's how we reduce **hallucinations**.

## 3. Structured output with Pydantic (quiz)
Normally an LLM returns free text, which code can't rely on.
`llm.with_structured_output(Quiz)` makes it return a **`Quiz` object** with exactly the fields we defined.
Pydantic **validates** it: wrong types raise errors. We also filter out any question that doesn't have 4 options.

## 4. Provider-agnostic LLM (`llm.py`)
`init_chat_model(model, model_provider=...)` gives every provider the **same interface**.
So `.invoke()` and `.with_structured_output()` work the same on Groq, OpenAI, Gemini, Claude and Ollama.
Changing provider = changing `LLM_PROVIDER` in `.env`. No code changes.

## 5. Why secrets go in `.env`
API keys must never be pushed to GitHub, because bots scan GitHub and steal them within minutes.
`.env` is listed in `.gitignore`, so git ignores it. `.env.example` shows the format without real keys.

---

## 🎤 Interview questions: practise answering these out loud

1. **Explain your project in 30 seconds.**
   *"It's a RAG app for students. You upload notes as PDFs; I split them into chunks, embed them with a sentence-transformer model, and store them in ChromaDB. When you ask a question, I retrieve the 4 most similar chunks and pass them to an LLM with a prompt that forces it to answer only from that context, with page citations. It can also generate quizzes using Pydantic structured output, and it works with any LLM provider through LangChain's init_chat_model."*

2. **What is RAG, and why not just fine-tune?**
   RAG adds knowledge at query time: it's cheap, easy to update (just re-index), and gives citations. Fine-tuning changes the model's weights: it's expensive and better for teaching *style or behaviour*, not facts.

3. **What is an embedding?** A vector of numbers representing the meaning of text; similar meaning → nearby vectors.

4. **Why chunk? How did you pick 500/50?** Context limits and precision. 500 characters is about one paragraph, a good balance. Too small loses context; too big adds noise. I'd tune it by testing answers.

5. **What is a vector database? Name some.** A store optimised for nearest-neighbour search over vectors: ChromaDB, FAISS, Pinecone, Weaviate, Qdrant, pgvector.

6. **How do you reduce hallucinations?** Strict prompt ("only use context, else say not found"), retrieval of relevant chunks, showing sources, temperature 0.

7. **What if retrieval returns the wrong chunks?** Improvements: better chunking, hybrid search (keyword BM25 + vectors), re-ranking with a cross-encoder, more/fewer chunks (tune k).

8. **What does `with_structured_output` do?** Uses the model's tool/JSON mode so the response matches a Pydantic schema, then parses and validates it.

9. **Cosine similarity?** Measures the angle between two vectors: 1 = same direction (same meaning), 0 = unrelated.

10. **How would you scale this to 10,000 PDFs?** Batch embedding, a hosted vector DB (Pinecone/Qdrant), metadata filters (per subject), caching, async ingestion.

## ✅ Exercises (do these to really learn it)
- [ ] Change `CHUNK_SIZE` to 200 and 1000. How do answers change?
- [ ] Change `k=4` to `k=1` and `k=8` in `ask()`.
- [ ] Switch `LLM_PROVIDER` to another provider and compare answers.
- [ ] Add a `difficulty` field (`easy`/`medium`/`hard`) to `QuizQuestion` and show it in the UI.
