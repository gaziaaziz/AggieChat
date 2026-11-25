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

## Evaluation Process

AggieChat was evaluated using a benchmark of 60 questions, including 57 answerable questions and 3 unanswerable questions. The system uses hybrid vector/BM25 retrieval, reranking, and the local `llama3.2:1b` model to generate answers.

The evaluation measures both retrieval quality and answer quality. Retrieval performance is evaluated by comparing the retrieved source IDs against the expected source IDs at a context size of five. Answer quality is evaluated with an LLM-as-judge setup using `protected.gpt-5.4-mini`, which scores each response on correctness, faithfulness, answer relevance, and context relevance.

| Metric | Baseline |
|---|---:|
| Overall score (mean of the four judge ratings) | 3.26 / 5 |
| Correctness | 3.07 / 5 |
| Faithfulness | 3.80 / 5 |
| Answer relevance | 3.13 / 5 |
| Context relevance | 3.05 / 5 |
| Source precision / recall / F1 | 0.525 / 0.544 / 0.534 |
| Response behavior accuracy | 48.3% |

These results provide an initial baseline for the system. Since the benchmark currently covers a limited number of newspaper issues, the scores should be interpreted as an early evaluation rather than a final measurement. Source F1 is calculated at the document level, using one expected source for each answerable question. The overall score is the unweighted mean of the four LLM judge ratings.

To rerun the benchmark, open a terminal from the project root and export the TAMU API credentials:

    export TAMUS_API_KEY="your_api_key_here"
    export TAMUS_MODEL="protected.gpt-5.4-mini"

Then run the evaluation script:

    python evaluation.py

The per-question evaluation report is written to:

    eval_results.json

The exported environment variables only apply to the current terminal session. If a new terminal session is opened, the environment variables must be exported again before rerunning the benchmark.

## Notes

- The corpus is heavily skewed toward The Battalion (~94% of harvested pages), so
  unfiltered searches favour it.
- Paths in `config.py` are resolved relative to the repo, so scripts work from any directory.
