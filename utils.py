def date_to_int(date: str) -> int:
    """'1916-01-29' -> 19160129. Returns 0 for missing or malformed dates."""
    try:
        return int((date or "").replace("-", ""))
    except ValueError:
        return 0
