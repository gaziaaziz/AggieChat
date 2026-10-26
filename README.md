# AggieChat

RAG pipeline over historical Texas A&M newspapers (The Battalion, The Texas Aggie,
Daily Bulletin/Reveille) harvested from the TAMU Libraries Open ONI collection.

## Pipeline (one branch per module)

| Branch | Module | Purpose |
|---|---|---|
| `data_prep` | `data_scraper.py`, `data_cleaner.py` | Harvest OCR pages, clean and chunk them |
| `indexer` | `indexer.py` | Embed chunks (Ollama) into a ChromaDB vector index |
| `fts_index` | `fts_index.py` | Build a SQLite FTS5 keyword index |
| `retrieval` | `bm25_search.py`, `filters.py` | BM25 search and publication filters |

Shared settings live in `config.py`. Each branch builds on the previous one.

## Running

```
pip install requests tiktoken chromadb
python data_scraper.py --max-pages 20   # pilot harvest -> data/raw
python data_cleaner.py                  # clean + chunk -> data/chunks
python indexer.py                       # vector index -> chroma_db/
python fts_index.py                     # keyword index -> fts_index.db
```

Data, `chroma_db/` and `*.db` files are not committed; regenerate them with the steps above.
Set `OLLAMA_HOST` to point at a non-default Ollama server.
