"""
generate.py

Reuses chunking_main.py's Gemini client + retry logic (no need to duplicate
API setup), but builds context/prompt generically via prompt.py instead of
chunking_main's 450T-specific build_prompt.
"""

from chunking_main import _generate_content_with_retry, NOT_FOUND_MESSAGE
from prompt import build_prompt, NOT_FOUND_TOKEN


def build_context(results):
    if not results:
        return "No relevant documentation found."

    parts = []
    for result in results:
        label = f"[{result['source']} | {result['section']}]"
        parts.append(f"{label}\n{result['text']}")
    return "\n\n".join(parts)


def generate_answer_from_context(query, results, app_name=None, conversation_history=None):
    # --- TEMPORARY DEBUG: remove once the fallback issue is confirmed fixed ---
    print(f"\n[DEBUG] query={query!r}")
    print(f"[DEBUG] num results retrieved: {len(results)}")
    for r in results:
        print(f"[DEBUG]   source={r['source']!r} section={r['section']!r} score={r['score']:.4f}")
    # ---------------------------------------------------------------------

    if not results:
        print("[DEBUG] -> returning NOT_FOUND_MESSAGE because results is EMPTY (retrieval problem, model never called)")
        return NOT_FOUND_MESSAGE

    context = build_context(results)
    prompt = build_prompt(query, context, app_name=app_name, conversation_history=conversation_history)

    try:
        response = _generate_content_with_retry(prompt)
    except Exception as error:
        print(f"[DEBUG] -> Gemini API call raised an exception: {error!r}")
        return f"Sorry, the answer could not be generated right now ({error})."

    answer = (response.text or "").strip()
    print(f"[DEBUG] raw model output: {answer!r}")

    # The model only ever emits the short sentinel token when it doesn't
    # know the answer -- the actual user-facing wording (including your
    # "contact support" line) lives entirely in NOT_FOUND_MESSAGE below,
    # never in the prompt itself. Strip trailing punctuation the model
    # sometimes adds around the token (quotes, periods) before comparing.
    normalized = answer.strip(' \n"\'.`')
    if not answer or normalized == NOT_FOUND_TOKEN:
        print("[DEBUG] -> model emitted the NOT_FOUND token (or empty text): retrieval worked, model chose not to answer")
        return NOT_FOUND_MESSAGE

    print("[DEBUG] -> returning model's real answer")
    return answer