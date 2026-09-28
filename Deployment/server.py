"""
server.py

FastAPI service exposing the support chatbot pipeline as a stateful HTTP
API, so other applications can integrate with it directly instead of using
the CLI/Streamlit interfaces.

No authentication is applied -- this assumes the service
runs on a trusted internal network. See the note at the bottom of this file
for where to add API-key auth if this ever needs to be exposed more widely.

Conversation state is stateful and server-side: POST /sessions creates a
session_id scoped to one app; every /sessions/{id}/messages call appends to
that session's history in memory -- the same (question, answer) tuple list
that chatbot_app.py's CLI loop tracked locally, just held per-session here
instead. This is a single-process, in-memory store: sessions are lost on
restart, and won't be shared if you ever run multiple worker processes.
That's fine for getting other apps integrated now; if usage grows to where
that matters, the fix is swapping the in-memory `_sessions` dict for an
external store (Redis, a database) without changing any endpoint's
contract.

Run locally:
    uvicorn server:app --host 0.0.0.0 --port 8000

Run in production (single worker -- see the note near the bottom about why
multiple workers need extra care with this pipeline):
    uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
"""

import time
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss

from Chatbot.Deployment.chunking_main import MODEL_NAME
from Chatbot.Deployment.loader import load_documents_blocks
from Chatbot.Deployment.retrieval import chunk_blocks, build_hybrid_index, retrieve_for_question
from Chatbot.Deployment.generate import generate_answer_from_context
from Chatbot.Deployment.answer_helpers import answer_meta_question
from Chatbot.Deployment.conversation import condense_query
from Chatbot.Deployment.app_config import APP_DOCUMENTS

app = FastAPI(title="Support Chatbot API", version="1.0")

# Tighten allow_origins to specific origins before this is reachable from
# outside a trusted network.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_hybrid_index = None
_sessions = {}  # session_id -> {"app": str, "history": [(q, a), ...], "last_active": float}

SESSION_TTL_SECONDS = 60 * 60 * 4  # sessions idle longer than this are dropped on next cleanup pass


def _build_index():
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
    return build_hybrid_index(all_chunks, model, faiss_index)


@app.on_event("startup")
def startup():
    global _hybrid_index
    print("Loading documents and building index...")
    _hybrid_index = _build_index()
    print(f"Ready. Apps available: {list(APP_DOCUMENTS.keys())}")


def _cleanup_stale_sessions():
    now = time.time()
    stale = [sid for sid, s in _sessions.items() if now - s["last_active"] > SESSION_TTL_SECONDS]
    for sid in stale:
        del _sessions[sid]


def _create_session(app_name: str) -> str:
    if app_name not in APP_DOCUMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown app '{app_name}'. Available: {list(APP_DOCUMENTS.keys())}",
        )
    _cleanup_stale_sessions()
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {"app": app_name, "history": [], "last_active": time.time()}
    return session_id


def _answer_in_session(session_id: str, question: str, top_k: int = 5) -> str:
    """Shared by both the explicit /sessions/{id}/messages endpoint and the
    widget-facing /chat/messages endpoint, so the two entry points can never
    drift apart in behavior."""
    session = _sessions[session_id]
    app_name = session["app"]
    history = session["history"]

    meta_answer = answer_meta_question(question)
    if meta_answer:
        answer = meta_answer
    else:
        standalone_query = condense_query(history, question)
        results = retrieve_for_question(standalone_query, _hybrid_index, top_k=top_k, app=app_name)
        recent_history = history[-4:]
        answer = generate_answer_from_context(
            standalone_query, results, app_name=app_name, conversation_history=recent_history
        )

    history.append((question, answer))
    session["last_active"] = time.time()
    return answer


# --- Request/response models ---
class CreateSessionRequest(BaseModel):
    app: str


class CreateSessionResponse(BaseModel):
    session_id: str
    app: str


class MessageRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class MessageResponse(BaseModel):
    session_id: str
    app: str
    question: str
    answer: str


class AppsResponse(BaseModel):
    apps: List[str]


class HistoryTurn(BaseModel):
    question: str
    answer: str


class HistoryResponse(BaseModel):
    session_id: str
    app: str
    history: List[HistoryTurn]


class ChatMessageRequest(BaseModel):
    app: str
    session_id: Optional[str] = None
    question: str
    top_k: Optional[int] = 5


# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok", "indexed": _hybrid_index is not None}


@app.get("/apps", response_model=AppsResponse)
def list_apps():
    return {"apps": list(APP_DOCUMENTS.keys())}


@app.post("/sessions", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest):
    session_id = _create_session(req.app)
    return {"session_id": session_id, "app": req.app}


@app.post("/sessions/{session_id}/messages", response_model=MessageResponse)
def send_message(session_id: str, req: MessageRequest):
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Create a new session via POST /sessions.",
        )
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty.")

    question = req.question.strip()
    answer = _answer_in_session(session_id, question, top_k=req.top_k or 5)
    return {"session_id": session_id, "app": session["app"], "question": question, "answer": answer}


@app.get("/sessions/{session_id}/history", response_model=HistoryResponse)
def get_history(session_id: str):
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    return {
        "session_id": session_id,
        "app": session["app"],
        "history": [{"question": q, "answer": a} for q, a in session["history"]],
    }


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
    return {"deleted": True}


# ---------------------------------------------------------------------------
# Widget-facing endpoints: what powers the embeddable "Ask me" panel.
#
# 450T and MIT already have their own "Ask me" buttons, so integration is:
#     <script src="https://your-host/embed.js" data-app="450T"></script>
#     <button onclick="SupportChat.open()">Ask me</button>
# The script does NOT create its own button -- it only builds the hidden
# chat panel and exposes window.SupportChat.open()/.close()/.toggle() for
# their existing button to call. No API calls, no session management on
# their side beyond that one onclick. Everything below is what makes that
# possible.
#
# IMPORTANT for whoever adds the <script> tag: place it as a plain,
# synchronous <script> (no async/defer) somewhere before their button can be
# clicked -- e.g. anywhere in <head> or right before the button in the body.
# Without async/defer, the browser runs it immediately when parsed, so
# window.SupportChat.open is guaranteed to exist by the time a person can
# actually click the button. If they do need async/defer (e.g. loading it
# from a tag manager), the onclick should guard for it:
#     onclick="window.SupportChat && window.SupportChat.open()"
# ---------------------------------------------------------------------------

@app.post("/chat/messages", response_model=MessageResponse)
def chat_message(req: ChatMessageRequest):
    """Like /sessions/{id}/messages, but session_id is optional: pass None
    on the first call and a session is created automatically; the response
    always includes session_id so the caller (the widget's JS) can reuse it
    for every subsequent message in that conversation, entirely in memory
    on the browser side -- no separate POST /sessions call needed first."""
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty.")

    session_id = req.session_id
    if not session_id or session_id not in _sessions:
        session_id = _create_session(req.app)
    elif _sessions[session_id]["app"] != req.app:
        # A session is permanently scoped to the app it was created for --
        # refuse rather than silently answer under the wrong app's docs.
        raise HTTPException(
            status_code=400,
            detail="This session_id belongs to a different app. Start a new conversation.",
        )

    question = req.question.strip()
    answer = _answer_in_session(session_id, question, top_k=req.top_k or 5)
    return {"session_id": session_id, "app": req.app, "question": question, "answer": answer}


@app.get("/chat", response_class=HTMLResponse)
def chat_page(app: str):
    """The actual chat UI, meant to be loaded inside an iframe by embed.js.
    Self-contained (no external assets) so it works the same whether it's
    opened directly or embedded in any host page, on an internal network
    with no guaranteed internet access for CDN assets."""
    if app not in APP_DOCUMENTS:
        return HTMLResponse(
            f"<p style='font-family:sans-serif;padding:16px;'>Unknown app '{app}'.</p>",
            status_code=400,
        )
    return HTMLResponse(_render_chat_page(app))


@app.get("/embed.js")
def embed_script():
    return Response(content=_EMBED_JS, media_type="application/javascript")


def _render_chat_page(app_name: str) -> str:
    # Minimal, dependency-free chat UI. app_name is validated against
    # APP_DOCUMENTS by the caller before this is rendered, and is only ever
    # inserted into a JS string literal below with basic escaping -- avoid
    # passing anything here that wasn't already checked against that list.
    safe_app = app_name.replace("\\", "\\\\").replace('"', '\\"')
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{app_name} Support Chat</title>
<style>
  html, body {{ height: 100%; margin: 0; font-family: -apple-system, Segoe UI, Roboto, sans-serif; }}
  body {{ display: flex; flex-direction: column; background: #f7f7f8; }}
  #header {{ padding: 12px 16px; background: #1f2937; color: #fff; font-weight: 600; font-size: 14px; }}
  #messages {{ flex: 1; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 10px; }}
  .msg {{ max-width: 85%; padding: 8px 12px; border-radius: 12px; font-size: 14px; line-height: 1.4; white-space: pre-wrap; }}
  .msg.user {{ align-self: flex-end; background: #2563eb; color: #fff; border-bottom-right-radius: 2px; }}
  .msg.bot {{ align-self: flex-start; background: #fff; color: #111; border: 1px solid #e5e7eb; border-bottom-left-radius: 2px; }}
  .msg.pending {{ opacity: 0.6; font-style: italic; }}
  #inputRow {{ display: flex; gap: 8px; padding: 10px; border-top: 1px solid #e5e7eb; background: #fff; }}
  #questionInput {{ flex: 1; padding: 8px 10px; border: 1px solid #d1d5db; border-radius: 8px; font-size: 14px; }}
  #sendBtn {{ padding: 8px 14px; border: none; border-radius: 8px; background: #2563eb; color: #fff; font-size: 14px; cursor: pointer; }}
  #sendBtn:disabled {{ opacity: 0.5; cursor: default; }}
</style>
</head>
<body>
  <div id="header">{app_name} Support</div>
  <div id="messages"></div>
  <div id="inputRow">
    <input id="questionInput" type="text" placeholder="Ask a question..." autocomplete="off">
    <button id="sendBtn">Send</button>
  </div>

<script>
(function() {{
  var APP = "{safe_app}";
  var sessionId = null;
  var messagesEl = document.getElementById("messages");
  var input = document.getElementById("questionInput");
  var sendBtn = document.getElementById("sendBtn");

  function addMessage(text, cls) {{
    var div = document.createElement("div");
    div.className = "msg " + cls;
    div.textContent = text;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }}

  function send() {{
    var question = input.value.trim();
    if (!question) return;
    input.value = "";
    sendBtn.disabled = true;
    addMessage(question, "user");
    var pending = addMessage("Thinking...", "bot pending");

    fetch("/chat/messages", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ app: APP, session_id: sessionId, question: question }})
    }})
      .then(function(r) {{ return r.json(); }})
      .then(function(data) {{
        pending.remove();
        if (data.answer) {{
          sessionId = data.session_id;
          addMessage(data.answer, "bot");
        }} else {{
          addMessage("Sorry, something went wrong: " + (data.detail || "unknown error"), "bot");
        }}
      }})
      .catch(function(err) {{
        pending.remove();
        addMessage("Sorry, something went wrong reaching the server.", "bot");
      }})
      .finally(function() {{ sendBtn.disabled = false; input.focus(); }});
  }}

  sendBtn.addEventListener("click", send);
  input.addEventListener("keydown", function(e) {{
    if (e.key === "Enter") send();
  }});
  input.focus();
}})();
</script>
</body>
</html>"""


_EMBED_JS = r"""
(function() {
  var thisScript = document.currentScript;
  var appName = thisScript.getAttribute("data-app");
  if (!appName) {
    console.error("[support-chat-widget] Missing required data-app attribute on the embed.js script tag.");
    return;
  }

  // Derive the API/chat host from this script's own src, so no host app
  // needs to hardcode our domain anywhere -- one tag is the whole contract.
  var scriptUrl = new URL(thisScript.src);
  var origin = scriptUrl.origin;

  var panel = document.createElement("div");
  Object.assign(panel.style, {
    position: "fixed", bottom: "20px", right: "20px", zIndex: 999999,
    width: "360px", height: "520px", maxWidth: "92vw", maxHeight: "75vh",
    borderRadius: "12px", overflow: "hidden", boxShadow: "0 8px 30px rgba(0,0,0,0.25)",
    display: "none", background: "#fff"
  });

  var iframe = document.createElement("iframe");
  iframe.src = origin + "/chat?app=" + encodeURIComponent(appName);
  Object.assign(iframe.style, { width: "100%", height: "100%", border: "none", display: "block" });
  panel.appendChild(iframe);

  // Small close control inside the panel itself, so the person isn't stuck
  // once it's open if the host page's own button only ever calls open()
  // rather than toggling.
  var closeBtn = document.createElement("button");
  closeBtn.textContent = "\u00D7";
  closeBtn.setAttribute("aria-label", "Close support chat");
  Object.assign(closeBtn.style, {
    position: "absolute", top: "6px", right: "8px", zIndex: 1,
    border: "none", background: "rgba(0,0,0,0.55)", color: "#fff",
    width: "24px", height: "24px", borderRadius: "50%", fontSize: "16px",
    lineHeight: "24px", textAlign: "center", cursor: "pointer", padding: "0"
  });
  panel.appendChild(closeBtn);

  document.body.appendChild(panel);

  var isOpen = false;
  function open() { panel.style.display = "block"; isOpen = true; }
  function close() { panel.style.display = "none"; isOpen = false; }
  function toggle() { isOpen ? close() : open(); }

  closeBtn.addEventListener("click", close);

  // Host apps that already have their own "Ask me" button call these
  // directly, e.g.: <button onclick="SupportChat.open()">Ask me</button>
  // No floating button is created by this script -- the host page's
  // existing button is the only trigger.
  window.SupportChat = window.SupportChat || {};
  window.SupportChat.open = open;
  window.SupportChat.close = close;
  window.SupportChat.toggle = toggle;
})();
"""


# --- Adding auth later ---
# If this API needs to be exposed beyond a trusted internal network, add an
# API-key dependency, e.g.:
#
#   from fastapi import Depends, Header
#   def require_api_key(x_api_key: str = Header(...)):
#       if x_api_key != EXPECTED_KEY:
#           raise HTTPException(status_code=401, detail="Invalid API key")
#
# then add `dependencies=[Depends(require_api_key)]` to @app routes.

# --- A note on scaling with multiple workers ---
# _hybrid_index and _sessions are plain in-process Python objects. Running
# `uvicorn server:app --workers 4` spins up 4 separate processes, each
# building its OWN copy of the index at startup (4x the embedding cost) and
# each holding its OWN _sessions dict -- a session created via one worker
# won't be found if a later request lands on a different worker. Fine at
# low request volume with 1 worker; if you need to scale beyond that, move
# _sessions to Redis/a database first, and consider a shared/external
# vector index (or a reverse-proxy with sticky sessions) before adding
# workers.