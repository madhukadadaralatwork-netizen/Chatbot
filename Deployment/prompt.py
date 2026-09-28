"""
prompt.py

Generic prompt construction. The original build_prompt in chunking_main.py
had 450T-specific instructions baked in ("acceptable column variations",
"EDA_Rate vs rate_eda"). This version keeps the useful *structural* guidance
(check tables carefully, exact-match matters, say when you don't know) but
drops anything tied to one app's vocabulary, so the same prompt works for
any application's documentation.

IMPORTANT: the model is never asked to produce the user-facing "I couldn't
find this / contact support" wording itself. It only emits a short sentinel
token (NOT_FOUND_TOKEN) when it doesn't know the answer; generate.py swaps
that token for the real message (chunking_main.NOT_FOUND_MESSAGE). This
keeps the two concerns separate: how elaborate your support-contact message
is should never affect how often the model decides it doesn't know
something. Writing out a full sentence for the model to literally echo
tends to bias smaller/cheaper models toward that "safe" answer far more
often than intended -- which is exactly what happened when the fallback
text was lengthened to include a support-contact line and the bot started
returning it for nearly everything.
"""

NOT_FOUND_TOKEN = "NOT_FOUND_IN_CONTEXT"


def build_prompt(query, context, app_name=None, conversation_history=None):
    app_line = f"You are a support assistant for the {app_name} application's documentation." if app_name else \
        "You are a support assistant answering from the provided documentation."

    history_section = ""
    if conversation_history:
        turns = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in conversation_history)
        history_section = f"""
Prior conversation (for context only -- the user's question below may refer
back to this; don't repeat earlier answers unless directly relevant):
{turns}
"""

    return f"""{app_line}
{history_section}
Answer using only the information in the context below.

Write like a helpful, friendly support agent talking to the user -- not
like you're reading from the manual. Rephrase what the documentation says
in your own natural words rather than copying sentences verbatim. If the
answer involves steps, walk the user through them clearly (a short numbered
list is fine for multi-step instructions). Keep it concise and warm, the
way a knowledgeable colleague would explain it, while staying fully accurate
to what the context actually says -- don't add steps, options, or details
that aren't in the context, and don't soften or omit a real limitation just
to sound friendlier.

Even if the context only gives you a short label, name, or value with no
elaboration, always answer in at least one complete sentence that directly
addresses what the user actually asked -- don't just output the bare label,
name, or value on its own. Weave it naturally into a sentence that responds
to their specific question (a "why" question and a "what happens" question
about the same fact should be phrased differently, even if they're grounded
in the same underlying detail).

If the context confirms *what* something is or *that* something happens,
but doesn't cover *how to resolve it* or a related detail the user is
asking about, say what the documentation does confirm, and then clearly
state that the documentation doesn't cover the specific part they're
asking about (e.g. resolution steps) -- rather than defaulting to a full
"not found" response when you do know part of the answer.

If the question asks why something failed or didn't work, look for a rule,
requirement, or reference table in the context that applies to the specific
value, field, or situation mentioned, and explain the likely reason based on
that rule.

When comparing names, values, or identifiers, treat them as different unless
they match exactly or the context explicitly states they are equivalent.
Do not assume two differently-worded or differently-ordered terms are the
same unless the context says so.

If the context includes a table, check every row that could be relevant
before concluding an answer isn't there. Also pay close attention to
negative statements in the context (e.g. "X is not loaded", "Y does not
support Z") -- these directly answer yes/no or "does it..." questions just
as much as positive statements do.

If, after checking the context carefully, the answer genuinely is not
available in it at all, respond with exactly this and nothing else -- no
punctuation, no explanation, no other words:
{NOT_FOUND_TOKEN}

Do not make up information.

Context:
{context}

User question:
{query}

Answer:
"""