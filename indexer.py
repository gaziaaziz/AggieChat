import argparse
import json
import sys
from pathlib import Path

import chromadb
import requests

from config import (
    BATCH_SIZE,
    CHROMA_DIR,
    CHUNKS_DIR,
    COLLECTION_NAME,
    EMBED_DIMENSIONS,
    EMBED_MODEL,
    OLLAMA_URL,
)


def test_ollama() -> bool:
    try:
        r = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": "test"}, timeout=10)
        vec = r.json().get("embedding", [])
        if len(vec) == EMBED_DIMENSIONS:
            print(f"Ollama OK — {EMBED_MODEL} ready ({EMBED_DIMENSIONS}d)")
            return True
        print(f"Ollama responded but vector length {len(vec)} != {EMBED_DIMENSIONS}")
        return False
    except Exception as e:
        print(f"Ollama not running — start with: ollama serve\n  ({e})")
        return False


def embed(text: str, _counter: list[int] | None = None) -> list[float] | None:
    try:
        r = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=30)
        r.raise_for_status()
        vec = r.json().get("embedding")
        if _counter is not None:
            _counter[0] += 1
            if _counter[0] % 50 == 0:
                print(f"  ... {_counter[0]} embeddings done", flush=True)
        return vec
    except Exception as e:
        print(f"\n  WARNING: embed failed — {e}")
        return None


def get_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def get_indexed_ids(collection) -> set[str]:
    result = collection.get(include=[])
    return set(result["ids"])


def index_chunks(collection, reset: bool = False) -> None:
    indexed_ids = set() if reset else get_indexed_ids(collection)
    chunk_files = sorted(Path(CHUNKS_DIR).glob("*.json"))
    print(f"Chunks: {len(chunk_files)} | Already indexed: {len(indexed_ids)}\n")

    skipped = 0
    failed  = 0
    indexed = 0
    counter = [0]

    for path in chunk_files:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)

        chunk_id = rec["chunk_id"]
        if chunk_id in indexed_ids:
            skipped += 1
            continue

        vector = embed(rec["text"], counter)
        if vector is None:
            failed += 1
            continue

        collection.add(
            ids=[chunk_id],
            embeddings=[vector],
            documents=[rec["text"]],
            metadatas=[{
                "doc_id":      rec["doc_id"],
                "lccn":        rec["lccn"],
                "title":       rec["title"],
                "date":        rec["date"],
                "page":        int(rec["page"]),
                "url":         rec["url"],
                "source":      rec["source"],
                "chunk_index": int(rec["chunk_index"]),
                "token_count": int(rec["token_count"]),
            }],
        )
        indexed += 1

    print(f"\nIndexed: {indexed} | Skipped: {skipped} | Failed: {failed}")
    print(f"ChromaDB total: {collection.count()}")


def validate(collection) -> None:
    query = "student life at Texas A&M"
    print(f"\n--- Validation query: '{query}' ---")

    vector = embed(query)
    if vector is None:
        print("Embedding failed — cannot validate.")
        return

    results = collection.query(
        query_embeddings=[vector],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    ids       = results["ids"][0]
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (chunk_id, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances), 1):
        print(f"\n  {i}. {meta['title']} | {meta['date']} | score: {1 - dist:.3f}")
        print(f"     {doc[:150]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed and index chunk files into ChromaDB")
    parser.add_argument("--validate", action="store_true", help="Run test query only, no indexing")
    parser.add_argument("--reset",    action="store_true", help="Wipe collection and re-index everything")
    args = parser.parse_args()

    if not test_ollama():
        sys.exit(1)

    collection = get_collection(reset=args.reset)

    if args.validate:
        if collection.count() == 0:
            print("Collection is empty — run indexing first.")
        else:
            validate(collection)
        return

    index_chunks(collection, reset=args.reset)
    validate(collection)


if __name__ == "__main__":
    main()
