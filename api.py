from __future__ import annotations

import json
import uuid
from typing import Iterator, Optional

import chromadb
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    LLM_MODEL,
    LLM_TIMEOUT,
    OLLAMA_LLM_URL,
    TEMPERATURE,
    TITLES,
    TOP_K,
)
from rag_pipeline import (
    NO_RESULTS_MESSAGE,
    RELIABILITY_WARNING,
    build_prompt,
    build_sources_block,
    classify_intent,
    contextualize_query,
    detect_hallucination,
    handle_direct,
    retrieve_context,
    rewrite_query,
    run_query,
)

MAX_SESSIONS = 500   # oldest sessions are evicted beyond this

# APP

app = FastAPI(
    title="TAMU Newspaper RAG API",
    description="Query the Texas A&M historical newspaper collection via natural language.",
    version="1.0.0",
)

# IN-MEMORY SESSION STORE

conversations: dict[str, list[dict]] = {}


def _save_history(session_id: str, history: list[dict]) -> None:
    conversations[session_id] = history
    while len(conversations) > MAX_SESSIONS:
        conversations.pop(next(iter(conversations)))

# PYDANTIC MODELS 

class ChatRequest(BaseModel):
    question:   str
    session_id: Optional[str] = None
    top_k:      int = TOP_K
    model:      str = LLM_MODEL


class ChatResponse(BaseModel):
    session_id:           str
    answer:               str
    query_used:           str
    intent:               str
    status:               str
    grounded:             bool
    hallucination_reason: str


class RetrieveRequest(BaseModel):
    query:  str
    top_k:  int  = TOP_K
    rerank: bool = True  # accepted for compatibility; retrieval always reranks


class RetrieveResponse(BaseModel):
    query:  str
    chunks: list[dict]


class StatsResponse(BaseModel):
    total_chunks: int
    titles:       dict[str, str]


# HELPERS

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)


def _stream_direct(text: str) -> Iterator[str]:
    """
    Yield a pre-written direct response (greeting, capability, etc.)
    word by word so it still feels streamed.
    Sends a final SSE event with an empty sources block.
    """
    words = text.split(" ")
    for i, word in enumerate(words):
        chunk = word if i == 0 else " " + word
        yield _event({"type": "token", "content": chunk})

    # No sources for direct responses
    yield _event({"type": "sources", "content": ""})
    yield _event({"type": "done"})


def _event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _stream_rag(question: str, history: list[dict]) -> Iterator[str]:
    """
    Run the same retrieval as run_query() (filters, hybrid search, rerank), then stream
    the LLM response token by token via Ollama's streaming API.
    After streaming: a "warning" event if the grounding check fails, then the sources.
    """
    clean_query     = rewrite_query(question)
    search_query    = contextualize_query(clean_query, history)
    reranked_chunks = retrieve_context(search_query)

    if not reranked_chunks:
        yield _event({"type": "token", "content": NO_RESULTS_MESSAGE})
        yield _event({"type": "sources", "content": ""})
        yield _event({"type": "done"})
        return

    messages = build_prompt(
        query   = clean_query,
        chunks  = reranked_chunks,
        history = history
    )

    answer_parts: list[str] = []
    try:
        resp = requests.post(
            OLLAMA_LLM_URL,
            json={
                "model":    LLM_MODEL,
                "messages": messages,
                "stream":   True,
                "options":  {"temperature": TEMPERATURE}
            },
            stream=True,
            timeout=LLM_TIMEOUT
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            token = data.get("message", {}).get("content", "")
            if token:
                answer_parts.append(token)
                yield _event({"type": "token", "content": token})

            if data.get("done"):
                break

    except Exception as e:
        print(f"[stream] LLM stream failed: {e!r}")
        yield _event({"type": "error", "content": "The language model request failed. Please try again."})
        yield _event({"type": "done"})
        return

    verdict = detect_hallucination("".join(answer_parts), reranked_chunks)
    if not verdict["grounded"]:
        yield _event({"type": "warning", "content": RELIABILITY_WARNING.strip()})

    yield _event({"type": "sources", "content": build_sources_block(reranked_chunks)})
    yield _event({"type": "done"})


# ROUTES

@app.get("/health")
def health():
    try:
        col    = get_collection()
        count  = col.count()
        status = "ok"
    except Exception as e:
        count  = 0
        print(f"[health] chroma error: {e!r}")
        status = "chroma_unavailable"
    return {"status": status, "chunks_indexed": count}


@app.get("/titles")
def list_titles():
    return {
        "titles": [
            {"lccn": lccn, "name": name}
            for lccn, name in TITLES.items()
        ]
    }


@app.get("/stats", response_model=StatsResponse)
def collection_stats():
    try:
        col   = get_collection()
        total = col.count()
    except Exception as e:
        raise HTTPException(status_code=503, detail="ChromaDB unavailable") from e
    return StatsResponse(total_chunks=total, titles=TITLES)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Non-streaming chat endpoint — kept for testing and fallback."""
    session_id = req.session_id or str(uuid.uuid4())
    history    = conversations.get(session_id, [])

    result = run_query(raw_query=req.question, history=history)

    if result["status"] == "error":
        raise HTTPException(status_code=502, detail=result["response"])

    if result["status"] != "direct" or result["intent"] in ("out_of_scope", "capability"):
        history.append({"role": "user",      "content": req.question})
        history.append({"role": "assistant", "content": result["response"]})
        _save_history(session_id, history)

    return ChatResponse(
        session_id           = session_id,
        answer               = result["response"],
        query_used            = result["query"],
        intent               = result["intent"],
        status               = result["status"],
        grounded             = result["grounded"],
        hallucination_reason = result["hallucination_reason"],
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """
    Streaming chat endpoint — Server-Sent Events (SSE).

    Event types:
      {"type": "token",   "content": "<token text>"}   — LLM token
      {"type": "sources", "content": "<markdown>"}     — sources block after stream
      {"type": "error",   "content": "<message>"}      — error during streaming
      {"type": "done"}                                  — stream complete

    The frontend accumulates tokens, then appends the sources block
    when the "done" event arrives.
    """
    session_id = req.session_id or str(uuid.uuid4())
    history    = conversations.get(session_id, [])

    clean_query = rewrite_query(req.question)
    route       = classify_intent(clean_query, history)
    intent      = route["intent"]

    if intent != "archive_query":
        # Direct response — stream word by word
        direct_text = handle_direct(intent, clean_query)

        # Persist only meaningful exchanges
        if intent in ("out_of_scope", "capability"):
            history.append({"role": "user",      "content": req.question})
            history.append({"role": "assistant", "content": direct_text})
            _save_history(session_id, history)

        return StreamingResponse(
            _stream_direct(direct_text),
            media_type="text/event-stream",
            headers={
                "X-Session-Id": session_id,
                "X-Intent":     intent,
                "Cache-Control": "no-cache",
            }
        )

    # RAG path — stream tokens from Ollama
    # We persist history after the stream using a wrapper generator
    full_response_parts: list[str] = []

    def rag_stream_with_history() -> Iterator[str]:
        for event in _stream_rag(req.question, history):
            payload = json.loads(event[len("data: "):])
            if payload.get("type") == "token":
                full_response_parts.append(payload["content"])
            elif payload.get("type") == "sources":
                # Persist the answer text only; sources/warnings are re-derived each turn
                history.append({"role": "user",      "content": req.question})
                history.append({"role": "assistant", "content": "".join(full_response_parts)})
                _save_history(session_id, history)
            yield event

    return StreamingResponse(
        rag_stream_with_history(),
        media_type="text/event-stream",
        headers={
            "X-Session-Id":  session_id,
            "X-Intent":      intent,
            "Cache-Control": "no-cache",
        }
    )


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_raw(req: RetrieveRequest):
    """Raw retrieval without LLM generation — for debugging."""
    clean_query = rewrite_query(req.query)
    try:
        chunks = retrieve_context(clean_query)
    except requests.exceptions.RequestException:
        raise HTTPException(
            status_code=503,
            detail="Embedding failed — is Ollama running?"
        )

    return RetrieveResponse(query=clean_query, chunks=chunks[: req.top_k])


@app.get("/history/{session_id}")
def get_history(session_id: str):
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session_id": session_id, "history": conversations[session_id]}


@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Session not found.")
    del conversations[session_id]
    return {"deleted": session_id}