"""
chatbot_app.py

Entry point. Nothing in this file is specific to any one application --
onboarding app #3, #4, #5 is purely an app_config.py edit.

Flow:
  1. Load every app's documents via loader.py (heading-aware for docx,
     sheet-aware for xlsx, native Word tables included).
  2. Chunk generically (retrieval_v2.chunk_blocks) -- tables kept whole
     where possible, narrative text windowed by word count.
  3. Build one combined hybrid (BM25 + embedding) index, tagged per app.
  4. User picks an app from app_config.APP_DOCUMENTS.
  5. Every question is answered scoped to that app; diagnostic-sounding
     questions get MMR-diversified retrieval so a rule and its supporting
     table both make it into context even if they're in different sections.
"""

from sentence_transformers import SentenceTransformer
import faiss

from chunking_main import MODEL_NAME, NOT_FOUND_MESSAGE
from loader import load_documents_blocks
from retrieval import chunk_blocks, build_hybrid_index, retrieve_for_question
from generate import generate_answer_from_context
from answer_helpers import answer_meta_question
from conversation import condense_query
from app_config import APP_DOCUMENTS


def build_combined_index(model):
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
    return all_chunks, hybrid_index


def select_app():
    apps = list(APP_DOCUMENTS.keys())
    print("\nWhich application is your question about?")
    for i, name in enumerate(apps, start=1):
        print(f"  {i}. {name}")

    while True:
        choice = input("Select a number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(apps):
            return apps[int(choice) - 1]
        print("Please enter a valid number.")


def answer_question_scoped(query, hybrid_index, app, history, top_k=5):
    if not query:
        return NOT_FOUND_MESSAGE

    meta_answer = answer_meta_question(query)
    if meta_answer:
        return meta_answer

    # Rewrite follow-ups ("how do I resolve this issue") into standalone
    # queries ("how do I resolve a supplier number mismatch") using recent
    # history, so retrieval has an actual error/entity name to search for.
    standalone_query = condense_query(history, query)

    results = retrieve_for_question(standalone_query, hybrid_index, top_k=top_k, app=app)
    recent_history = history[-4:]
    return generate_answer_from_context(standalone_query, results, app_name=app, conversation_history=recent_history)


def main():
    print("Loading documents and building index for all applications...")
    model = SentenceTransformer(MODEL_NAME)
    chunks, hybrid_index = build_combined_index(model)
    print(f"Ready. {len(chunks)} chunks indexed across {len(APP_DOCUMENTS)} application(s).")

    selected_app = select_app()
    print(f"\nSupport Chatbot -- scoped to: {selected_app}")
    print("Type 'switch' to change application, 'new' to clear conversation memory, 'exit/quit/bye' to quit.\n")

    history = []  # list of (question, answer) tuples, oldest first -- reset on app switch

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit", "bye"}:
            print("Bot: Thank you. Have a great day.")
            break
        if question.lower() == "switch":
            selected_app = select_app()
            history = []
            print(f"Bot: Switched to {selected_app}. Conversation memory cleared.\n")
            continue
        if question.lower() == "new":
            history = []
            print("Bot: Conversation memory cleared.\n")
            continue

        answer = answer_question_scoped(question, hybrid_index, selected_app, history, top_k=5)
        print(f"Bot: {answer}\n")
        history.append((question, answer))


if __name__ == "__main__":
    main()