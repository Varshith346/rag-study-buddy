"""Streamlit web app. Run with:  streamlit run app.py"""
import os
import tempfile

import streamlit as st

from rag import build_index, ask, make_quiz, has_index

st.set_page_config(page_title="RAG Study Buddy", page_icon="📚")
st.title("📚 RAG Study Buddy")
st.caption("Upload your notes, ask questions, and quiz yourself.")

# ---------- Sidebar: upload + index ----------
with st.sidebar:
    st.header("1. Upload your notes")
    files = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Build knowledge base", disabled=not files, use_container_width=True):
        with st.spinner("Reading, chunking and embedding your notes..."):
            tmp_dir = tempfile.mkdtemp()
            paths = []
            for f in files:
                path = os.path.join(tmp_dir, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                paths.append(path)
            pages, chunks = build_index(paths)
        st.success(f"Indexed {pages} pages into {chunks} chunks ✅")
    st.divider()
    st.caption(f"LLM provider: **{os.getenv('LLM_PROVIDER', 'groq')}** (change it in .env)")

if not has_index():
    st.info("👈 Upload a PDF and click **Build knowledge base** to start.")
    st.stop()

ask_tab, quiz_tab = st.tabs(["💬 Ask your notes", "📝 Quiz me"])

# ---------- Tab 1: chat ----------
with ask_tab:
    if "history" not in st.session_state:
        st.session_state.history = []
    for role, msg in st.session_state.history:
        st.chat_message(role).markdown(msg)

    question = st.chat_input("Ask something about your notes...")
    if question:
        st.chat_message("user").markdown(question)
        with st.spinner("Searching your notes..."):
            answer, sources = ask(question)
        reply = f"{answer}\n\n*Sources: {', '.join(sources)}*"
        st.chat_message("assistant").markdown(reply)
        st.session_state.history += [("user", question), ("assistant", reply)]

# ---------- Tab 2: quiz ----------
with quiz_tab:
    topic = st.text_input("Quiz topic", placeholder="e.g. binary trees")
    n = st.slider("Number of questions", 3, 10, 5)
    if st.button("Generate quiz", disabled=not topic):
        with st.spinner("Writing your quiz..."):
            st.session_state.quiz = make_quiz(topic, n)

    quiz = st.session_state.get("quiz")
    if quiz:
        score = 0
        for i, q in enumerate(quiz.questions):
            st.markdown(f"**Q{i + 1}. {q.question}**")
            choice = st.radio("Pick one", q.options, index=None, key=f"q{i}", label_visibility="collapsed")
            if choice is not None:
                if q.options.index(choice) == q.answer_index:
                    score += 1
                    st.success(f"Correct! {q.explanation}")
                else:
                    st.error(f"Answer: {q.options[q.answer_index]}. {q.explanation}")
        st.metric("Score", f"{score} / {len(quiz.questions)}")
