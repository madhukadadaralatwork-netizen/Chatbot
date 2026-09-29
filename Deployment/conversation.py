"""
conversation.py

Handles multi-turn context. Two responsibilities:

  1. condense_query: rewrites a follow-up question ("how do I resolve this
     issue") into a standalone one ("how do I resolve a supplier number
     mismatch") using recent conversation history, BEFORE retrieval. Without
     this, retrieval embeds "this issue" literally and finds nothing,
     because the actual error name never appears in the follow-up question's
     own text.

  2. Conversation history is also passed into the final generation prompt
     (see prompt.py) so the answer itself stays coherent with what was
     already discussed.

Only triggers the rewrite (an extra LLM call) when the question actually
contains a referring/anaphoric word -- a generic, domain-agnostic signal
that a question depends on prior context. A fully self-contained question
("what does error code 12 mean") skips this step entirely and costs nothing
extra.
"""

import re

from chunking_main import _generate_content_with_retry

ANAPHORA_TERMS = {
    "this", "that", "it", "these", "those", "issue", "problem", "error",
    "same", "again", "there", "them", "he", "she", "they",
}


def _needs_context(query):
    tokens = re.findall(r"[a-z']+", query.lower())
    return any(token in ANAPHORA_TERMS for token in tokens)


def condense_query(history, current_query, max_turns=4):
    """history: list of (user_question, bot_answer) tuples, oldest first."""
    if not history or not _needs_context(current_query):
        return current_query

    recent = history[-max_turns:]
    convo_text = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in recent)

    prompt = f"""Given the conversation so far, rewrite the user's latest message as a
standalone question that includes any necessary context (specific values,
error names, entities) from earlier in the conversation, so it can be
understood without the conversation history. Do not answer the question.
Return only the rewritten question, nothing else.

Conversation so far:
{convo_text}

Latest message: {current_query}

Standalone question:"""

    try:
        response = _generate_content_with_retry(prompt)
        rewritten = (response.text or "").strip()
        return rewritten or current_query
    except Exception:
        return current_query