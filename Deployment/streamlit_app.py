"""
streamlit_app.py

Streamlit frontend for the support chatbot. Reuses the existing pipeline
(chunking_main / loader / retrieval_v2 / generate / conversation /
app_config) directly, in-process -- no separate API server needed for an
internal tool like this.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss

from Chatbot.Deployment.chunking_main import MODEL_NAME, NOT_FOUND_MESSAGE
from Chatbot.Deployment.loader import load_documents_blocks
from Chatbot.Deployment.retrieval import chunk_blocks, build_hybrid_index, retrieve_for_question
from Chatbot.Deployment.generate import generate_answer_from_context
from Chatbot.Deployment.answer_helpers import answer_meta_question
from Chatbot.Deployment.conversation import condense_query
from Chatbot.Deployment.app_config import APP_DOCUMENTS

st.set_page_config(page_title="Support Chatbot", page_icon="💬", layout="centered")


@st.cache_resource(show_spinner="Loading documents and building index...")
def load_pipeline():
    """Cached across the whole Streamlit process, not per-session -- building
    the index is expensive (embedding every chunk), so it should happen once
    on first load, not on every user interaction or every new browser tab."""
    model = SentenceTransformer(MODEL_NAME)
    all_chunks = []
    next_id = 0

    for app_name, document_paths in APP_DOCUMENTS.items():
        blocks = load_documents_blocks(document_paths)
        app_chunks = chunk_blocks(blocks, app=app_name)
        for chunk in app_chunks:
            chunk.chunk_id = next_id
            next_id += 1
        all_chunks.extend(app_chunks)

    if not all_chunks:
        raise RuntimeError("No documents loaded -- check app_config.APP_DOCUMENTS paths.")

    texts = [c.text for c in all_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True).astype("float32")
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)

    hybrid_index = build_hybrid_index(all_chunks, model, faiss_index)
    return hybrid_index


def answer_question_scoped(query, hybrid_index, app, history, top_k=5):
    if not query:
        return NOT_FOUND_MESSAGE
    meta_answer = answer_meta_question(query)
    if meta_answer:
        return meta_answer
    standalone_query = condense_query(history, query)
    results = retrieve_for_question(standalone_query, hybrid_index, top_k=top_k, app=app)
    recent_history = history[-4:]
    return generate_answer_from_context(
        standalone_query, results, app_name=app, conversation_history=recent_history
    )


# --- Session state ---
if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, answer) tuples, oldest first
if "selected_app" not in st.session_state:
    st.session_state.selected_app = list(APP_DOCUMENTS.keys())[0]

try:
    hybrid_index = load_pipeline()
except Exception as e:
    st.error(f"Failed to load documents/index: {e}")
    st.stop()

# --- Sidebar: app picker + conversation controls ---
with st.sidebar:
    st.title("Support Chatbot")
    app_names = list(APP_DOCUMENTS.keys())
    selected = st.selectbox(
        "Application", app_names, index=app_names.index(st.session_state.selected_app)
    )
    if selected != st.session_state.selected_app:
        st.session_state.selected_app = selected
        st.session_state.history = []  # switching apps resets conversation memory
        st.rerun()

    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()

st.subheader(f"Ask about: {st.session_state.selected_app}")

# --- Render existing conversation ---
for question, answer in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        st.markdown(answer)

# --- New question ---
question = st.chat_input("Ask a question...")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = answer_question_scoped(
                question, hybrid_index, st.session_state.selected_app, st.session_state.history, top_k=5
            )
        st.markdown(answer)
    st.session_state.history.append((question, answer))