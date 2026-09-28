"""
answer_helpers.py

Deliberately tiny. The only special-cased questions here are ones that are
true for every app ("who are you") -- anything that depends on document
content is left entirely to retrieval + the LLM, so this file never needs
per-app edits.
"""

def answer_meta_question(query):
    lowered_query = query.lower().strip()
    if lowered_query in {"who are you", "who are you?"}:
        return "I am a support assistant for the selected application's documentation."
    if "what can you help" in lowered_query:
        return "I can help answer questions from the documentation for the application you selected."
    return None