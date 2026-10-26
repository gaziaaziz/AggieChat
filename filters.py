"""
filters.py — Pre-retrieval metadata filter extraction for ChatONI.

Extracts date range and publication filters from a query string and returns
a ChromaDB-compatible `where` clause dict, or None if nothing was detected
confidently enough to filter on.

Design principles:
- Fail open: when uncertain, return None (no filter). A wrong filter zeros
  out results; no filter just means a wider search.
- Regex only: no LLM call. Deterministic, fast, debuggable.
- Conservative: only fire on explicit, unambiguous signals in the query.
"""

import re
from config import TITLES

# Inverted map: lowercase publication name keywords → lccn
# Each entry is a list of aliases that should map to that lccn
PUBLICATION_ALIASES = {
    "sn86088544": ["battalion"],
    "sn84009780": ["texas aggie", "aggie"],
    "sn37020267": ["daily bulletin", "bulletin", "reveille"],
}

# Build a flat lookup: keyword → lccn (sorted longest-first to avoid partial matches)
_KEYWORD_TO_LCCN: list[tuple[str, str]] = sorted(
    [
        (keyword, lccn)
        for lccn, keywords in PUBLICATION_ALIASES.items()
        for keyword in keywords
    ],
    key=lambda x: len(x[0]),
    reverse=True,
)


def _extract_lccn(query: str) -> str | None:
    """
    Detect a publication mention in the query and return its LCCN.
    Returns None if no confident match found.
    """
    q = query.lower()
    for keyword, lccn in _KEYWORD_TO_LCCN:
        if keyword in q:
            return lccn
    return None


def _extract_date_range(query: str) -> tuple[int, int] | None:
    """
    Detect an explicit year or decade in the query and return (start, end) as
    integer YYYYMMDD values for ChromaDB $gte/$lte comparison on date_int.
    Returns None if no confident date signal found.

    Handled patterns:
        - "in 1963", "during 1975", "of 1920"  → full year range
        - "the 1960s", "1960s", "in the 1970s" → decade range
        - bare 4-digit year anywhere            → full year range (lower confidence,
                                                  still explicit enough to use)
    """
    q = query.lower()

    # Decade: "1960s", "the 1960s", "in the 1970s"
    decade_match = re.search(r'\b(1[89]\d)0s\b', q)
    if decade_match:
        decade_start = int(decade_match.group(1) + "0")
        return (int(f"{decade_start}0101"), int(f"{decade_start + 9}1231"))

    # Explicit year with preposition: "in 1963", "during 1975", "of 1920"
    prep_year_match = re.search(r'\b(?:in|during|of|from|around)\s+(1[89]\d{2})\b', q)
    if prep_year_match:
        year = prep_year_match.group(1)
        return (int(f"{year}0101"), int(f"{year}1231"))

    # Bare 4-digit year (e.g. "Battalion 1975 football")
    bare_year_match = re.search(r'\b(1[89]\d{2})\b', q)
    if bare_year_match:
        year = bare_year_match.group(1)
        return (int(f"{year}0101"), int(f"{year}1231"))

    return None


def extract_filters(query: str) -> dict | None:
    """
    Main entry point. Returns a ChromaDB `where` clause dict, or None.

    Examples:
        "Battalion coverage in the 1960s"
            → {"$and": [{"lccn": {"$eq": "sn86088544"}},
                        {"date_int": {"$gte": 19600101}},
                        {"date_int": {"$lte": 19691231}}]}

        "what did the texas aggie say about integration"
            → {"lccn": {"$eq": "sn84009780"}}

        "campus protests in 1975"
            → {"$and": [{"date_int": {"$gte": 19750101}},
                        {"date_int": {"$lte": 19751231}}]}

        "who was the football coach"
            → None  (no filter — wide open search)
    """
    lccn       = _extract_lccn(query)
    date_range = _extract_date_range(query)

    conditions = []

    if lccn:
        conditions.append({"lccn": {"$eq": lccn}})

    if date_range:
        start, end = date_range
        conditions.append({"date_int": {"$gte": start}})
        conditions.append({"date_int": {"$lte": end}})

    if not conditions:
        return None                          # fail open

    if len(conditions) == 1:
        return conditions[0]                 # single condition — no $and needed

    return {"$and": conditions}              # multiple conditions