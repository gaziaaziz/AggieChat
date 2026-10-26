# AggieChat

RAG pipeline over historical Texas A&M newspapers (The Battalion, The Texas Aggie,
Daily Bulletin/Reveille) harvested from the TAMU Libraries Open ONI collection.

## Pipeline (one branch per module)

| Branch | Module | Purpose |
|---|---|---|
| `data_prep` | `data_scraper.py`, `data_cleaner.py` | Harvest OCR pages, clean and chunk them |
| `indexer` | `indexer.py` | Embed chunks (Ollama) into a ChromaDB vector index |
| `fts_index` | `fts_index.py` | Build a SQLite FTS5 keyword index |
| `retrieval` | `bm25_search.py`, `filters.py` | BM25 search and query metadata filters |
| `rag_pipeline` | `rag_pipeline.py`, `reranker.py` | Hybrid retrieval, rerank, generation, grounding check |
| `api` | `api.py` | FastAPI service (`/chat`, `/chat/stream`, `/retrieve`, ...) |
| `app` | `app.py` | Streamlit chat UI |

Shared settings live in `config.py` (on `main`); `utils.py` holds small shared helpers.
Each branch builds on the previous one.

## Running

```
pip install -r requirements.txt
python data_scraper.py --max-pages 20   # pilot harvest -> data/raw
python data_cleaner.py                  # clean + chunk -> data/chunks
python indexer.py                       # vector index -> chroma_db/
python fts_index.py                     # keyword index -> fts_index.db
uvicorn api:app --port 8000             # API (needs `ollama serve` running)
streamlit run app.py                    # UI
```

Data, `chroma_db/` and `*.db` files are not committed; regenerate them with the steps above.
Set `OLLAMA_HOST` to point at a non-default Ollama server and `API_BASE` for the UI.

## Notes

- The corpus is heavily skewed toward The Battalion (~94% of harvested pages), so
  unfiltered searches favour it.
- Paths in `config.py` are resolved relative to the repo, so scripts work from any directory.
