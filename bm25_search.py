"""
bm25_search.py — BM25 keyword search over the SQLite FTS5 index.

Returns results in the same chunk dict format as retrieve_chunks() so
both branches can be fused with RRF without any format conversion.

The `where` parameter accepts the same filter dict produced by
extract_filters() — lccn and date_int conditions are applied as SQL
WHERE clauses before FTS5 ranking, mirroring the ChromaDB metadata
filter on the dense branch.
"""

import os

from config import FTS_DB_PATH
from fts_index import get_connection, sanitize_fts_query


def _build_sql_where(where: dict | None) -> tuple[str, list]:
    """
    Convert a ChromaDB-style `where` dict into a SQL WHERE clause + params.

    Handles:
        {"lccn": {"$eq": "sn86088544"}}
        {"date_int": {"$gte": 19600101}}
        {"date_int": {"$lte": 19691231}}
        {"$and": [...list of the above...]}

    Returns (clause_str, params_list).
    Empty string + empty list = no filter.
    """
    if not where:
        return "", []

    def parse_condition(cond: dict) -> tuple[str, list]:
        field, ops = next(iter(cond.items()))

        if field == "$and":
            clauses, params = [], []
            for sub in ops:
                c, p = parse_condition(sub)
                clauses.append(c)
                params.extend(p)
            return " AND ".join(clauses), params

        op, val = next(iter(ops.items()))
        op_map = {"$eq": "=", "$gte": ">=", "$lte": "<="}
        sql_op = op_map.get(op, "=")
        return f"m.{field} {sql_op} ?", [val]

    clause, params = parse_condition(where)
    return f"AND {clause}", params


def bm25_search(
    query: str,
    top_k: int = 20,
    where: dict | None = None
) -> list[dict]:
    """
    Search the FTS5 index for `query` and return up to top_k results
    as chunk dicts compatible with retrieve_chunks() output.

    Each returned dict has:
        text        — chunk text
        metadata    — same fields as ChromaDB metadata
        similarity  — None (no cosine score; RRF only needs rank, not score)
        bm25_score  — raw BM25 score from SQLite (negative; lower = better)

    Fails open: returns [] on any error so a broken FTS index never
    crashes the pipeline.
    """
    conn = None
    try:
        if not os.path.exists(FTS_DB_PATH):
            print(f"[bm25] index not found at {FTS_DB_PATH} — run fts_index.py first")
            return []

        safe_query = sanitize_fts_query(query)
        if not safe_query:
            return []

        conn = get_connection()
        where_clause, where_params = _build_sql_where(where)

        sql = f"""
            SELECT
                m.chunk_id, m.doc_id, m.lccn, m.title, m.date, m.date_int,
                m.page, m.url, m.source, m.chunk_index, m.token_count, m.text,
                bm25(chunks_fts) AS score
            FROM chunks_fts f
            JOIN chunks_meta m ON f.rowid = m.rowid
            WHERE chunks_fts MATCH ?
            {where_clause}
            ORDER BY score          -- bm25() returns negatives; lower = more relevant
            LIMIT ?
        """

        params = [safe_query] + where_params + [top_k]
        rows   = conn.execute(sql, params).fetchall()

        chunks = []
        for row in rows:
            (chunk_id, doc_id, lccn, title, date, date_int,
             page, url, source, chunk_index, token_count, text, score) = row

            chunks.append({
                "text": text,
                "metadata": {
                    "chunk_id":    chunk_id,
                    "doc_id":      doc_id,
                    "lccn":        lccn,
                    "title":       title,
                    "date":        date,
                    "date_int":    date_int,
                    "page":        page,
                    "url":         url,
                    "source":      source,
                    "chunk_index": chunk_index,
                    "token_count": token_count,
                },
                "similarity":  None,     # no cosine score from BM25 branch
                "bm25_score":  score,
            })

        return chunks

    except Exception as e:
        print(f"[bm25] search failed, returning empty: {e}")
        return []

    finally:
        if conn is not None:
            conn.close()