"""
fts_index.py — Build and manage the SQLite FTS5 index for ChatONI hybrid search.

Run once to build the index from chunk files. Safe to re-run — already-indexed
chunks are skipped. The resulting fts_index.db sits alongside chroma_db/ in
the repo root.

Usage:
    python fts_index.py              # index all chunks
    python fts_index.py --reset      # wipe and rebuild from scratch
    python fts_index.py --validate   # run a test query and show top 5 results
"""

import os
import json
import sqlite3
import argparse
from pathlib import Path

from config import CHUNKS_DIR

FTS_DB_PATH = "fts_index.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(FTS_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")   # safe concurrent reads
    return conn


def create_table(conn: sqlite3.Connection):
    """
    Two tables:
      - chunks_fts: FTS5 virtual table for full-text search on chunk text.
        content= links it to chunks_meta so FTS5 reads text from there.
      - chunks_meta: regular table storing chunk_id + all metadata + raw text.
        This is what bm25_search() reads back to build chunk dicts.
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks_meta (
            chunk_id    TEXT PRIMARY KEY,
            doc_id      TEXT,
            lccn        TEXT,
            title       TEXT,
            date        TEXT,
            date_int    INTEGER,
            page        INTEGER,
            url         TEXT,
            source      TEXT,
            chunk_index INTEGER,
            token_count INTEGER,
            text        TEXT
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            content=chunks_meta,
            content_rowid=rowid,
            tokenize='porter ascii'
        );
    """)
    conn.commit()


def get_indexed_ids(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT chunk_id FROM chunks_meta").fetchall()
    return {r[0] for r in rows}


def build_index(reset: bool = False):
    if reset and os.path.exists(FTS_DB_PATH):
        os.remove(FTS_DB_PATH)
        print("[fts_index] reset — existing index deleted")

    conn = get_connection()
    create_table(conn)

    already_indexed = get_indexed_ids(conn)
    chunk_files     = sorted(Path(CHUNKS_DIR).glob("*.json"))

    total    = 0
    skipped  = 0
    inserted = 0

    for path in chunk_files:
        total += 1
        try:
            with open(path) as f:
                chunk = json.load(f)
        except Exception as e:
            print(f"[fts_index] could not read {path.name}: {e}")
            continue

        chunk_id = chunk.get("chunk_id", path.stem)
        if chunk_id in already_indexed:
            skipped += 1
            continue

        date_str = chunk.get("date", "")
        date_int = int(date_str.replace("-", "")) if date_str else 0

        conn.execute("""
            INSERT INTO chunks_meta
                (chunk_id, doc_id, lccn, title, date, date_int, page,
                 url, source, chunk_index, token_count, text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk_id,
            chunk.get("doc_id", ""),
            chunk.get("lccn", ""),
            chunk.get("title", ""),
            date_str,
            date_int,
            chunk.get("page", 1),
            chunk.get("url", ""),
            chunk.get("source", ""),
            chunk.get("chunk_index", 0),
            chunk.get("token_count", 0),
            chunk.get("text", ""),
        ))

        # Keep FTS5 content table in sync
        conn.execute("""
            INSERT INTO chunks_fts (rowid, text)
            SELECT rowid, text FROM chunks_meta WHERE chunk_id = ?
        """, (chunk_id,))

        inserted += 1

        if inserted % 5000 == 0:
            conn.commit()
            print(f"[fts_index] {inserted} inserted, {skipped} skipped...")

    conn.commit()
    conn.close()

    print(f"\n[fts_index] done.")
    print(f"  total files : {total}")
    print(f"  inserted    : {inserted}")
    print(f"  skipped     : {skipped}")
    print(f"  db path     : {FTS_DB_PATH}")


def sanitize_fts_query(query: str) -> str:
    """
    Clean user query for FTS5 without stripping meaningful content.
    Only removes & which causes SQLite tokenizer errors, then
    collapses whitespace. Intentionally preserves hyphens, apostrophes,
    and all letters so BM25 matching on historical text is not degraded.
    """
    import re
    query = query.replace("&", " ")
    query = re.sub(r'\s+', ' ', query)
    return query.strip()


def validate():
    conn = get_connection()

    count = conn.execute("SELECT COUNT(*) FROM chunks_meta").fetchone()[0]
    print(f"[fts_index] total records: {count}")

    query  = "student life campus Texas A&M"
    safe_q = sanitize_fts_query(query)
    print(f"\n[fts_index] test query: '{query}' -> sanitized: '{safe_q}'")
    print("-" * 60)

    rows = conn.execute("""
        SELECT m.chunk_id, m.title, m.date, m.page,
               snippet(chunks_fts, 0, '[', ']', '...', 15)
        FROM chunks_fts f
        JOIN chunks_meta m ON f.rowid = m.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY bm25(chunks_fts)
        LIMIT 5
    """, (safe_q,)).fetchall()

    for i, row in enumerate(rows, 1):
        chunk_id, title, date, page, snippet = row
        print(f"\n#{i} {title} | {date} | Page {page}")
        print(f"   {snippet}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset",    action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    if args.validate:
        validate()
    else:
        build_index(reset=args.reset)