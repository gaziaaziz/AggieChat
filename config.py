import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  # paths below do not depend on the working directory

BASE_URL = "https://newspaper.library.tamu.edu"

TITLES = {
    "sn86088544": "The Battalion",
    "sn84009780": "The Texas Aggie",
    "sn37020267": "The Daily Bulletin/Reveille",
    # "sn88037063": "The Texas Polka News",
}

RESTRICTED_LCCNS = set()  # Texas Polka News LCCN goes here once confirmed

DELAY = 0.6      # seconds between HTTP requests
MAX_PAGES = 100
RAW_DIR = str(BASE_DIR / "data" / "raw")

CHUNKS_DIR      = str(BASE_DIR / "data" / "chunks")
CHUNK_TOKENS    = 450
OVERLAP_TOKENS  = 60
MIN_ALPHA_RATIO = 0.45   # drop pages/chunks where < 45% of chars are letters
MIN_CHUNK_CHARS = 150    # drop chunks shorter than this

CHROMA_DIR       = str(BASE_DIR / "chroma_db")
FTS_DB_PATH      = str(BASE_DIR / "fts_index.db")
COLLECTION_NAME  = "tamu_newspapers"
EMBED_MODEL      = "embeddinggemma:300m"
EMBED_DIMENSIONS = 768
BATCH_SIZE       = 32

OLLAMA_HOST    = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL     = f"{OLLAMA_HOST}/api/embeddings"
OLLAMA_LLM_URL = f"{OLLAMA_HOST}/api/chat"
EMBED_TIMEOUT  = 30    # seconds
LLM_TIMEOUT    = 120   # seconds
LLM_MODEL       = "llama3.1:8b"
TOP_K           = 15
TEMPERATURE     = 0.1
MAX_HISTORY_TURNS = 6   # max individual messages (3 exchanges) to keep in prompt
MIN_SCORE_THRESHOLD = 0.40  # drop dense-only chunks below this cosine similarity (1 - cosine distance)
MAX_CONTEXT_CHUNKS   = 5    # top N chunks passed to the LLM prompt
MAX_SOURCES          = 3    # top N unique pages shown in sources block

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # local reranker, no API key needed
RETRIEVE_TOP_K      = 20
RERANK_TOP_N        = MAX_CONTEXT_CHUNKS
MIN_RERANK_SCORE    = -5.0  # cross-encoder logit; chunks scoring below this are dropped (-> "no results")

# Hybrid search
HYBRID_ENABLED   = True   # set False to fall back to dense-only retrieval
VECTOR_TOP_K     = 20     # candidates from ChromaDB dense branch
BM25_TOP_K       = 20     # candidates from FTS5 sparse branch
RRF_K            = 60     # RRF constant — 60 is the standard default
MIN_FILTERED_RESULTS = 3   # if a metadata filter yields fewer candidates than this, retry unfiltered
HALLUCINATION_CHECK_ENABLED = True  # extra LLM call per answer; set False for speed
