import re
import json
import requests
import chromadb
from datetime import datetime

from config import (
    OLLAMA_URL, EMBED_MODEL, OLLAMA_HOST,
    CHROMA_DIR, COLLECTION_NAME,
    MAX_CONTEXT_CHUNKS, MAX_SOURCES,
    MIN_SCORE_THRESHOLD, MAX_HISTORY_TURNS,
    OLLAMA_LLM_URL, LLM_MODEL, TEMPERATURE,
    EMBED_TIMEOUT, LLM_TIMEOUT,
    BASE_URL, TITLES,
    HYBRID_ENABLED, VECTOR_TOP_K, BM25_TOP_K, RRF_K,
    MIN_FILTERED_RESULTS, HALLUCINATION_CHECK_ENABLED,
)
from reranker import rerank_chunks
from filters import extract_filters
from bm25_search import bm25_search

#CHROMA CLIENT (opened lazily so importing this module never needs the index)
_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client      = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


def _chunk_key(chunk: dict) -> str:
    """Stable per-chunk id. Dense (Chroma) metadata has no chunk_id, so build it."""
    meta = chunk["metadata"]
    if meta.get("chunk_id"):
        return meta["chunk_id"]
    return f"{meta.get('doc_id', '')}_chunk{meta.get('chunk_index', 0)}"


def _post_llm(messages: list[dict], temperature: float, timeout: int = LLM_TIMEOUT) -> str:
    """One non-streaming chat call to Ollama; returns the message text."""
    resp = requests.post(
        OLLAMA_LLM_URL,
        json={
            "model":    LLM_MODEL,
            "messages": messages,
            "stream":   False,
            "options":  {"temperature": temperature},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _parse_json(raw: str) -> dict:
    """Parse a JSON object from an LLM reply, tolerating ``` fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)

#RRF FUSION
def reciprocal_rank_fusion(
    dense_chunks: list[dict],
    sparse_chunks: list[dict],
    k: int = RRF_K
) -> list[dict]:
    """
    Fuse dense (vector) and sparse (BM25) results using Reciprocal Rank Fusion.

    Each chunk gets a score: sum of 1/(k + rank) across branches it appears in.
    Chunks are matched by chunk_id from metadata. A chunk appearing in both
    branches scores higher than one appearing in only one.

    Returns a deduplicated list sorted by RRF score descending.
    RRF score is added as 'rrf_score' for transparency.
    """
    scores  = {}   # chunk key → rrf_score
    chunks  = {}   # chunk key → chunk dict (dense preferred for similarity field)

    for rank, chunk in enumerate(dense_chunks, start=1):
        cid = _chunk_key(chunk)
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
        chunks[cid] = chunk   # dense chunk has similarity score — keep it

    for rank, chunk in enumerate(sparse_chunks, start=1):
        cid = _chunk_key(chunk)
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
        if cid in chunks:
            chunks[cid]["in_sparse"] = True    # confirmed by both branches
        else:
            chunks[cid] = chunk                # sparse-only

    fused = []
    for cid, rrf_score in scores.items():
        chunk = chunks[cid]
        chunk["rrf_score"] = round(rrf_score, 6)
        fused.append(chunk)

    fused.sort(key=lambda c: c["rrf_score"], reverse=True)
    return fused


#SYSTEM PROMPT
SYSTEM_PROMPT = """You are a research assistant for the Texas A&M University Libraries, \
helping staff explore digitized historical newspaper archives. The archives include \
The Battalion, The Texas Aggie, and The Daily Bulletin/Reveille.

You answer questions strictly based on the newspaper excerpts provided to you. \
Follow these rules without exception:

1. Only use information explicitly present in the provided excerpts.
2. If the excerpts do not contain enough information to answer the question, say so clearly. \
   Do not speculate or fill gaps with general knowledge.
3. Never fabricate dates, names, events, or quotes.
4. Do not use any citation markers, numbers, brackets, or footnotes of any kind \
   in your response. Do not write things like [1], [2], (1), or similar. \
   A reference list is appended automatically — your job is only to write \
   the answer in plain prose.
5. OCR quality in historical documents can be imperfect — if a passage seems garbled, \
   note that the source text may have transcription errors rather than treating it as fact.
6. Keep your answers factual, concise, and grounded in the provided excerpts."""

#ROUTER PROMPT & DIRECT RESPONSES

ROUTER_SYSTEM_PROMPT = """You are a query classifier for a library research assistant. \
Classify the user's message into exactly one of these categories:

- greeting: a hello, hi, hey, good morning, or similar greeting
- smalltalk: casual conversation (how are you, what's up, tell me about yourself)
- closer: goodbye, thanks, thank you, see you, bye, or similar closing
- capability: asking what the assistant can do, how it works, what archives it covers
- out_of_scope: a genuine question but unrelated to Texas A&M history or its newspaper archives
- archive_query: any question about Texas A&M, its history, people, events, or the newspaper archives

Respond with ONLY a JSON object in this exact format:
{
    "intent": "<one of the six categories above>",
    "reason": "<one short sentence explaining your classification>"
}

No preamble, no markdown, no explanation outside the JSON."""

DIRECT_RESPONSES = {
    "greeting": (
        "Hello! I'm the Texas A&M University Libraries archive assistant. "
        "I can help you explore historical newspapers from The Battalion, "
        "The Texas Aggie, and The Daily Bulletin/Reveille. What would you like to research?"
    ),
    "smalltalk": (
        "I'm doing well, thanks for asking! I'm here to help you explore the Texas A&M "
        "historical newspaper archives. Got a research question I can help with?"
    ),
    "closer": (
        "You're welcome — happy to help anytime. Good luck with your research!"
    ),
    "capability": (
        "I can help you search and explore three digitized historical newspaper archives "
        "from Texas A&M University:\n\n"
        "- **The Battalion** — the student newspaper, covering campus life, events, and news\n"
        "- **The Texas Aggie** — alumni publication\n"
        "- **The Daily Bulletin / Reveille** — early campus bulletins\n\n"
        "You can ask me about people, events, campus life, sports, or anything covered "
        "in these archives. I'll find the most relevant excerpts and answer based on what "
        "the newspapers actually reported. What would you like to explore?"
    ),
}

#HALLUCINATION JUDGE PROMPT
HALLUCINATION_CHECK_PROMPT = """You are a fact-checking assistant. You will be given:
1. A set of source excerpts from historical newspapers
2. A response generated from those excerpts

Your job is to check whether every factual claim in the response is explicitly supported
by the source excerpts.

Respond with ONLY a JSON object in this exact format:
{
    "grounded": true or false,
    "reason": "one sentence explanation if not grounded, empty string if grounded"
}

Do not include anything else in your response. No preamble, no markdown, no explanation."""

#REWRITE QUERY
def rewrite_query(raw_query: str) -> str:
    """
    Normalize and clean the raw user query before embedding.
    - Strip whitespace
    - Collapse internal whitespace runs
    - Remove noisy characters (keep letters, digits, spaces, apostrophes, hyphens, &)
    - Lowercase
    """
    query = raw_query.strip()
    query = re.sub(r'\s+', ' ', query)
    query = re.sub(r"[^\w\s'\-&]", '', query)
    query = query.lower()
    return query

#EMBED QUERY
def embed_query(query: str) -> list[float]:
    """
    Convert cleaned query string into an embedding vector via Ollama.
    Uses the same model and endpoint as the indexer for vector space consistency.
    """
    response = requests.post(
        OLLAMA_URL,
        json={
            "model":  EMBED_MODEL,
            "prompt": query
        },
        timeout=EMBED_TIMEOUT
    )
    response.raise_for_status()
    return response.json()["embedding"]

#RETRIEVE CHUNKS
def retrieve_chunks(
    query_embedding: list[float],
    top_k: int = VECTOR_TOP_K,
    where: dict | None = None
) -> list[dict]:
    """
    Search ChromaDB for the top_k most similar chunks to the query embedding.
    Returns a list of dicts with text, metadata, and similarity score.
    Default top_k is intentionally wider than what reaches the LLM (RETRIEVE_TOP_K,
    e.g. 20) -- the reranker stage narrows this back down to RERANK_TOP_N candidates.

    `where` is an optional ChromaDB metadata filter dict produced by extract_filters().
    When None, retrieval is unfiltered (original behaviour).
    """
    query_kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    if where:
        query_kwargs["where"] = where

    results = _get_collection().query(**query_kwargs)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        similarity = 1 - dist   # collection uses cosine distance = 1 - cosine similarity
        chunks.append({
            "text":       doc,
            "metadata":   meta,
            "similarity": round(similarity, 4)
        })

    return chunks

#FILTER CHUNKS
def filter_chunks(chunks: list[dict], threshold: float = MIN_SCORE_THRESHOLD) -> list[dict]:
    """
    Drop weak dense matches.
    A chunk is kept if any of these hold:
      - it has no cosine score (BM25-only hit; RRF already ranked it),
      - its cosine similarity >= threshold,
      - the BM25 branch also found it (keyword evidence overrides a low cosine score).
    Sorted by rrf_score when present (hybrid), otherwise by cosine similarity.
    """
    filtered = [
        c for c in chunks
        if c.get("similarity") is None
        or c["similarity"] >= threshold
        or c.get("in_sparse")
    ]

    def sort_key(c: dict) -> float:
        if c.get("rrf_score") is not None:
            return c["rrf_score"]
        return c["similarity"] if c.get("similarity") is not None else -1.0

    filtered.sort(key=sort_key, reverse=True)
    return filtered

#SOURCES BLOCK
def build_sources_block(chunks: list[dict], max_sources: int = MAX_SOURCES) -> str:
    """
    Build a deduplicated sources list from chunk metadata.
    Deduplication at page level: same lccn + date + page = one entry.
    Capped at max_sources entries — chunks are already sorted best-first
    so the most relevant pages naturally win.
    """
    seen    = set()
    sources = []

    for chunk in chunks:
        if len(sources) >= max_sources:
            break

        meta = chunk["metadata"]
        lccn = meta.get("lccn", "")
        date = meta.get("date", "")
        page = meta.get("page", "1")

        key = (lccn, date, page)
        if key in seen:
            continue
        seen.add(key)

        title = TITLES.get(lccn, "Unknown Publication")

        try:
            dt       = datetime.strptime(date, "%Y-%m-%d")
            date_fmt = f"{dt.strftime('%B')} {dt.day}, {dt.year}"
        except Exception:
            date_fmt = date

        url = meta.get("url") or f"{BASE_URL}/lccn/{lccn}/{date}/ed-1/seq-{page}/"
        sources.append(
            f"- {title}, {date_fmt}, Page {page} — [View Page]({url})"
        )

    if not sources:
        return ""

    return "\n\n**Sources**\n" + "\n".join(sources)

#BUILD PROMPT
def _strip_appendix(text: str) -> str:
    """Remove the appended reliability warning and **Sources** block from a stored answer."""
    text = text.split("\n\n> ⚠️")[0]
    return text.split("\n\n**Sources**")[0]


def build_prompt(
    query:      str,
    chunks:     list[dict],
    history:    list[dict],
    max_chunks: int = MAX_CONTEXT_CHUNKS
) -> list[dict]:
    """
    Assemble the full message list to send to the LLM.
    Format: system prompt → trimmed history → current query with context.
    Only the top max_chunks are passed as context.
    """
    messages = []

    messages.append({
        "role":    "system",
        "content": SYSTEM_PROMPT
    })

    for msg in history[-(MAX_HISTORY_TURNS):]:
        content = msg["content"]
        if msg["role"] == "assistant":
            content = _strip_appendix(content)   # keep sources/warnings out of the prompt
        messages.append({"role": msg["role"], "content": content})

    context_chunks = chunks[:max_chunks]

    if context_chunks:
        context_blocks = []
        for chunk in context_chunks:
            meta = chunk["metadata"]
            context_blocks.append(
                f"--- {meta.get('title', 'Unknown')} "
                f"| {meta.get('date', 'Unknown date')} "
                f"| Page {meta.get('page', '?')} ---\n"
                f"{chunk['text']}"
            )
        context_str = "\n\n".join(context_blocks)

        user_content = (
            f"Using the following excerpts from the archive, answer the question below.\n\n"
            f"EXCERPTS:\n{context_str}\n\n"
            f"QUESTION: {query}"
        )
    else:
        user_content = (
            f"No relevant excerpts were found in the archive for the following question.\n\n"
            f"QUESTION: {query}\n\n"
            f"Inform the user that no relevant information was found in the digitized "
            f"archives for their query. Do not attempt to answer from general knowledge."
        )

    messages.append({
        "role":    "user",
        "content": user_content
    })

    return messages

#GENERATE RESPONSE
def generate_response(
    messages: list[dict],
    chunks:   list[dict]
) -> str:
    """
    Send assembled messages to Ollama and return the response with sources appended.
    """
    llm_text = _post_llm(messages, TEMPERATURE)
    sources  = build_sources_block(chunks)

    return llm_text + sources

#HALLUCINATION DETECTOR
def detect_hallucination(response_text: str, chunks: list[dict]) -> dict:
    """
    Check whether the LLM response is grounded in the retrieved chunks.
    Makes a second lightweight LLM call acting as a judge, shown the full chunk text.
    Fails open — if the judge call fails, returns grounded=True.
    With no chunks there is nothing to ground against and the answer is the
    "nothing found" message, so it is reported as grounded.
    """
    if not HALLUCINATION_CHECK_ENABLED or not chunks:
        return {"grounded": True, "reason": ""}

    source_text = ""
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        source_text += (
            f"[Source {i}] {meta.get('title', '')} | {meta.get('date', '')} "
            f"| Page {meta.get('page', '')}\n"
            f"{chunk['text']}\n\n"
        )

    judge_messages = [
        {"role": "system", "content": HALLUCINATION_CHECK_PROMPT},
        {
            "role":    "user",
            "content": (
                f"SOURCE EXCERPTS:\n{source_text}\n"
                f"RESPONSE TO CHECK:\n{response_text}"
            )
        }
    ]

    try:
        result = _parse_json(_post_llm(judge_messages, 0.0))
        return {
            "grounded": bool(result.get("grounded", True)),
            "reason":   result.get("reason", "")
        }

    except Exception as e:
        print(f"[hallucination detector] failed: {e}")
        return {"grounded": True, "reason": ""}

#CONVERSATIONAL ROUTER
_QUICK_INTENTS = [
    ("greeting", re.compile(r"^(hi|hello|hey|howdy|good (morning|afternoon|evening))( there)?$")),
    ("closer",   re.compile(r"^(thanks|thank you|thank you so much|thanks a lot|bye|goodbye|see you|see ya)$")),
]


def classify_intent(query: str, history: list[dict]) -> dict:
    """
    Classify the user query into one of six intents.
    Bare greetings/thanks are matched by rule (no LLM call); everything else uses a
    lightweight LLM call with the last 2 exchanges for context.
    Fails open to archive_query if classification fails.

    Returns:
        {
            "intent": str,   # greeting | smalltalk | closer | capability |
                             # out_of_scope | archive_query
            "reason": str
        }
    """
    bare = re.sub(r"[^\w\s]", "", query.lower()).strip()
    for intent, pattern in _QUICK_INTENTS:
        if pattern.match(bare):
            return {"intent": intent, "reason": "matched rule"}

    recent = history[-4:]
    history_text = ""
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content'][:150]}\n"

    user_content = (
        f"Conversation so far:\n{history_text}\nNew message: {query}"
        if history_text else
        f"Message: {query}"
    )

    judge_messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content}
    ]

    try:
        result = _parse_json(_post_llm(judge_messages, 0.0))
        intent = result.get("intent", "archive_query")

        valid_intents = {
            "greeting", "smalltalk", "closer",
            "capability", "out_of_scope", "archive_query"
        }
        if intent not in valid_intents:
            intent = "archive_query"

        return {
            "intent": intent,
            "reason": result.get("reason", "")
        }

    except Exception as e:
        print(f"[router] classification failed: {e}")
        return {
            "intent": "archive_query",
            "reason": "classification failed, defaulting to archive_query"
        }


def handle_direct(intent: str, query: str) -> str:
    """
    Generate a direct response for non-RAG intents.
    For out_of_scope, makes a general knowledge LLM call.
    For all others, returns a pre-written response.
    """
    if intent in DIRECT_RESPONSES:
        return DIRECT_RESPONSES[intent]

    if intent == "out_of_scope":
        messages = [
            {
                "role":    "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question directly "
                    "and concisely from your general knowledge. After answering, "
                    "gently note that your primary role is helping research the Texas A&M "
                    "historical newspaper archives, and invite them to ask archive-related questions."
                )
            },
            {
                "role":    "user",
                "content": query
            }
        ]

        try:
            return _post_llm(messages, 0.7)

        except Exception:
            return (
                "I wasn't able to answer that right now. My primary role is helping you "
                "explore the Texas A&M historical newspaper archives — feel free to ask "
                "me anything about The Battalion, The Texas Aggie, or The Daily Bulletin."
            )

    return DIRECT_RESPONSES["greeting"]

#QUERY CONTEXTUALIZATION
_FOLLOW_UP_CUES = re.compile(
    r"\b(he|she|it|they|them|his|her|their|that|those|this|these|there|then|"
    r"also|more|another|else|same|what about|how about)\b"
)


def contextualize_query(query: str, history: list[dict]) -> str:
    """
    Turn a follow-up ("what about 1975?") into a standalone search query using recent
    history. Only calls the LLM when there is history and the query looks like a
    follow-up (short or containing referring words). Fails open to the original query.
    """
    if not history:
        return query
    if len(query.split()) >= 8 and not _FOLLOW_UP_CUES.search(query):
        return query

    convo = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {_strip_appendix(m['content'])[:300]}"
        for m in history[-4:]
    )
    messages = [
        {"role": "system", "content": (
            "Rewrite the user's last question as one standalone search query for a "
            "newspaper archive, resolving pronouns and references using the conversation. "
            "Keep names, dates and places. Output ONLY the query, nothing else."
        )},
        {"role": "user", "content": f"{convo}\n\nLast question: {query}"},
    ]
    try:
        rewritten = _post_llm(messages, 0.0, timeout=EMBED_TIMEOUT * 2).strip().strip('"')
        if rewritten and len(rewritten) < 300:
            return rewrite_query(rewritten)
    except Exception as e:
        print(f"[contextualize] failed: {e}")
    return query


#RETRIEVAL ORCHESTRATION (shared by run_query and the API)
def retrieve_context(search_query: str) -> list[dict]:
    """
    Filters → dense (+ BM25) retrieval → RRF → score filter → rerank.
    If a metadata filter leaves fewer than MIN_FILTERED_RESULTS candidates, the search is
    repeated without it. Returns the reranked chunks to give the LLM (possibly empty).
    """
    def candidates(where: dict | None) -> list[dict]:
        vector = embed_query(search_query)
        dense  = retrieve_chunks(vector, where=where)
        if not HYBRID_ENABLED:
            return dense
        sparse = bm25_search(search_query, top_k=BM25_TOP_K, where=where)
        print(f"[hybrid] dense: {len(dense)} | sparse: {len(sparse)}")
        return reciprocal_rank_fusion(dense, sparse)

    metadata_filter = extract_filters(search_query)
    print(f"[filters] applied: {metadata_filter}" if metadata_filter
          else "[filters] no filter — unfiltered retrieval")

    raw_chunks = candidates(metadata_filter)
    if metadata_filter and len(raw_chunks) < MIN_FILTERED_RESULTS:
        print("[filters] too few results with filter — retrying unfiltered")
        raw_chunks = candidates(None)

    return rerank_chunks(search_query, filter_chunks(raw_chunks))


NO_RESULTS_MESSAGE = (
    "I couldn't find anything relevant to that in the digitized archives. "
    "Try rephrasing, or add a publication name or a year to narrow the search."
)


#ORCHESTRATOR
def run_query(
    raw_query: str,
    history:   list[dict] = None
) -> dict:
    """
    Full RAG pipeline with conversational routing.
    The only public function — call this from FastAPI.

    Returns:
        {
            "response":             str,
            "chunks":               list,
            "query":                str,
            "status":               str,   # "ok" | "no_results" | "direct" | "error"
            "grounded":             bool,
            "hallucination_reason": str,
            "intent":               str
        }
    """
    if history is None:
        history = []

    try:
        clean_query = rewrite_query(raw_query)

        #Step 1: Route
        route  = classify_intent(clean_query, history)
        intent = route["intent"]

        #Step 2: Handle non-RAG intents directly
        if intent != "archive_query":
            return {
                "response":             handle_direct(intent, clean_query),
                "chunks":               [],
                "query":                clean_query,
                "status":               "direct",
                "grounded":             True,
                "hallucination_reason": "",
                "intent":               intent
            }

        #Step 3: Full RAG pipeline
        search_query    = contextualize_query(clean_query, history)
        reranked_chunks = retrieve_context(search_query)

        if not reranked_chunks:
            return {
                "response":             NO_RESULTS_MESSAGE,
                "chunks":               [],
                "query":                search_query,
                "status":               "no_results",
                "grounded":             True,
                "hallucination_reason": "",
                "intent":               intent
            }

        messages = build_prompt(
            query   = clean_query,
            chunks  = reranked_chunks,
            history = history
        )

        response = generate_response(
            messages = messages,
            chunks   = reranked_chunks
        )

        #Step 4: Hallucination check
        prose_only    = response.split("\n\n**Sources**")[0]
        hallucination = detect_hallucination(prose_only, reranked_chunks[:MAX_CONTEXT_CHUNKS])

        if not hallucination["grounded"]:
            response = prose_only + RELIABILITY_WARNING + build_sources_block(reranked_chunks)

        return {
            "response":             response,
            "chunks":               reranked_chunks,
            "query":                search_query,
            "status":               "ok",
            "grounded":             hallucination["grounded"],
            "hallucination_reason": hallucination["reason"],
            "intent":               intent
        }

    except requests.exceptions.ConnectionError:
        return _error_result(raw_query, f"Could not connect to Ollama. Make sure it is running at {OLLAMA_HOST}.")

    except requests.exceptions.Timeout:
        return _error_result(raw_query, "The language model took too long to respond. Please try again.")

    except Exception as e:
        print(f"[run_query] unexpected error: {e!r}")
        return _error_result(raw_query, "Something went wrong while processing your question. Please try again.")


RELIABILITY_WARNING = (
    "\n\n> ⚠️ **Reliability warning:** This response may contain information "
    "not directly supported by the archive excerpts. "
    "Please verify against the linked sources."
)


def _error_result(raw_query: str, message: str) -> dict:
    return {
        "response":             message,
        "chunks":               [],
        "query":                raw_query,
        "status":               "error",
        "grounded":             False,
        "hallucination_reason": "",
        "intent":               "unknown"
    }
