"""
reranker.py — Local cross-encoder reranking for ChatONI.

Replaces the NVIDIA NIM hosted reranker (deprecated) with a local
cross-encoder model from sentence-transformers. Runs on CPU on the
M4 Mac — no GPU, no API key, no network call required.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 80MB, downloads once from HuggingFace on first run
  - Purpose-built for passage reranking (query, passage) pairs
  - Fast enough on CPU for RERANK_TOP_N candidates (typically 5-10)

Sits between filter_chunks() and build_prompt() in the pipeline:
    retrieve (wide) -> filter (score threshold) -> RERANK -> build_prompt

Fails open: if the cross-encoder fails for any reason, falls back to
incoming chunk order truncated to top_n (no score cutoff) — a reranker
outage never breaks the pipeline.
"""

from __future__ import annotations

from config import CROSS_ENCODER_MODEL, MIN_RERANK_SCORE, RERANK_TOP_N

# Load model once at module level — avoids reloading on every query
_cross_encoder = None

def _get_model():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)
            print(f"[reranker] loaded {CROSS_ENCODER_MODEL}")
        except Exception as e:
            print(f"[reranker] failed to load model: {e}")
    return _cross_encoder


def rerank_chunks(
    query:  str,
    chunks: list[dict],
    top_n:  int = RERANK_TOP_N
) -> list[dict]:
    """
    Re-score chunks using a local cross-encoder and return the top_n
    sorted by relevance descending.

    Chunks scoring below MIN_RERANK_SCORE are dropped, so an irrelevant candidate set
    yields [] (the pipeline then reports "no results" instead of answering from noise).

    Adds a 'rerank_score' key to each chunk dict. The original
    'similarity' (cosine) key is left untouched so both scores are
    available downstream.

    Fails open: if the model errors for any reason, returns the
    incoming chunks truncated to top_n in their original order.
    """
    if not chunks:
        return chunks

    model = _get_model()
    if model is None:
        print("[reranker] model unavailable — skipping rerank")
        return chunks[:top_n]

    try:
        # Cross-encoder expects (query, passage) pairs
        pairs  = [(query, c["text"]) for c in chunks]
        scores = model.predict(pairs)

        # Attach scores and sort descending
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        return [c for c in reranked if c["rerank_score"] >= MIN_RERANK_SCORE][:top_n]

    except Exception as e:
        print(f"[reranker] reranking failed, falling back to pre-rerank order: {e}")
        return chunks[:top_n]