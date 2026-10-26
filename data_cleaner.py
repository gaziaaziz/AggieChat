import argparse
import json
import re
from pathlib import Path

import tiktoken

from config import (
    CHUNK_TOKENS,
    CHUNKS_DIR,
    MIN_ALPHA_RATIO,
    MIN_CHUNK_CHARS,
    OVERLAP_TOKENS,
    RAW_DIR,
)

ENC = tiktoken.get_encoding("cl100k_base")

MASTHEAD_TRIGGERS = [
    "THE BATTALION IS PUBLISHED",
    "INDEPENDENT STUDENT VOICE",
    "SERVING TEXAS A&M SINCE",
]

_PRICE_RE    = re.compile(r"\$[\d,]+")
_PHONE_RE    = re.compile(r"\d{3}[-.\s]\d{3}[-.\s]\d{4}|\d{10}")
_NAV_RE      = re.compile(
    r"more at thebatt|^continued$|^inews$|^page \d+$",
    re.IGNORECASE | re.MULTILINE,
)


def quality_score(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isalpha() for c in text) / len(text)


def clean_ocr(text: str) -> str | None:
    # Step 1 — Rejoin hyphenated line breaks (belt-and-suspenders)
    text = re.sub(r"(\w+)-\s+(\w)", lambda m: m.group(1) + m.group(2), text)

    # Step 2 — Strip pipe separators (modern layout artifacts: | | |)
    text = re.sub(r"\s*\|\s*", " ", text)

    # Step 3 — Strip masthead / staff box (modern issues)
    upper = text.upper()
    cutoff = len(text)
    for trigger in MASTHEAD_TRIGGERS:
        idx = upper.find(trigger)
        if idx != -1 and idx < cutoff:
            cutoff = idx
    text = text[:cutoff]

    # Step 4 — Strip ad lines (short lines with price or phone)
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) < 40 and (_PRICE_RE.search(stripped) or _PHONE_RE.search(stripped)):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # Step 5 — Strip navigation artifacts and ALL-CAPS noise lines
    text = _NAV_RE.sub("", text)
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Drop short ALL-CAPS lines with no sentence-ending punctuation
        if (stripped and stripped == stripped.upper() and len(stripped) < 25 and not any(c in stripped for c in ".!?")):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # Step 6 — Fix 1s / 1n OCR artifacts (whole-word only)
    text = re.sub(r"\b1s\b", "is", text)
    text = re.sub(r"\b1n\b", "in", text)

    # Step 7 — Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()

    # Step 8 — Quality gate on full page
    if not text or quality_score(text) < MIN_ALPHA_RATIO:
        return None

    return text


def chunk_text(text: str) -> list[str]:
    tokens = ENC.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_TOKENS, len(tokens))
        chunk = ENC.decode(tokens[start:end])
        if len(chunk) >= MIN_CHUNK_CHARS and quality_score(chunk) >= MIN_ALPHA_RATIO:
            chunks.append(chunk)
        start += CHUNK_TOKENS - OVERLAP_TOKENS
    return chunks


def already_chunked(doc_id: str) -> bool:
    return (Path(CHUNKS_DIR) / f"{doc_id}_chunk0.json").exists()


def write_chunk(record: dict, chunk_index: int, chunk_text: str) -> None:
    doc_id = record["doc_id"]
    chunk_id = f"{doc_id}_chunk{chunk_index}"
    payload = {
        "chunk_id":    chunk_id,
        "doc_id":      doc_id,
        "chunk_index": chunk_index,
        "text":        chunk_text,
        "lccn":        record["lccn"],
        "title":       record["title"],
        "date":        record["date"],
        "page":        record["page"],
        "url":         record["url"],
        "source":      record["source"],
        "token_count": len(ENC.encode(chunk_text)),
    }
    path = Path(CHUNKS_DIR) / f"{chunk_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def process_file(path: Path, dry_run: bool) -> tuple[int, str]:
    """Return (chunks_written, status_label)."""
    with open(path, encoding="utf-8") as f:
        record = json.load(f)

    doc_id = record.get("doc_id", path.stem)
    record.setdefault("doc_id", doc_id)

    if not dry_run and already_chunked(doc_id):
        return 0, "skip"

    cleaned = clean_ocr(record.get("ocr_text", ""))
    if cleaned is None:
        return 0, "low-quality"

    chunks = chunk_text(cleaned)
    if not chunks:
        return 0, "no-chunks"

    if dry_run:
        print(f"\n--- {doc_id} ({len(chunks)} chunks) ---")
        print(f"[cleaned text preview]\n{cleaned[:600]}\n")
        for i, ch in enumerate(chunks):
            print(f"  chunk{i} ({len(ENC.encode(ch))} tokens, {len(ch)} chars): {ch[:120]!r}")
        return len(chunks), "dry-run"

    for i, ch in enumerate(chunks):
        write_chunk(record, i, ch)
    return len(chunks), "ok"


def run(raw_files: list[Path], dry_run: bool) -> None:
    if not dry_run:
        Path(CHUNKS_DIR).mkdir(parents=True, exist_ok=True)

    pages_processed = 0
    skipped_quality = 0
    skipped_empty   = 0
    total_chunks    = 0

    for path in raw_files:
        n, status = process_file(path, dry_run)
        if status == "skip":
            continue
        elif status == "low-quality":
            skipped_quality += 1
            print(f"  skipped (low quality): {path.name}")
        elif status == "no-chunks":
            skipped_empty += 1
            print(f"  skipped (no chunks):   {path.name}")
        else:
            pages_processed += 1
            total_chunks += n
            if status != "dry-run":
                print(f"  ✓ {path.stem} → {n} chunks")

    if not dry_run:
        print(
            f"\n{pages_processed} pages → {total_chunks} chunks"
            f" | {skipped_quality} low-quality | {skipped_empty} no-chunks"
        )
        validate_output(skipped_quality, skipped_empty)


def validate_output(skipped_quality: int = 0, skipped_empty: int = 0) -> None:
    files = sorted(Path(CHUNKS_DIR).glob("*.json"))
    if not files:
        print("\nNo chunk files found.")
        return

    lengths = []
    raw_pages = set()
    for f in files:
        with open(f, encoding="utf-8") as fh:
            rec = json.load(fh)
        if "text" not in rec or "doc_id" not in rec:
            continue
        lengths.append((len(rec["text"]), f.name))
        raw_pages.add(rec["doc_id"])

    if not lengths:
        print("\nNo valid chunk files found.")
        return

    lengths.sort()
    avg = sum(l for l, _ in lengths) / len(lengths)

    print(f"\n--- Validation ---")
    print(f"Chunk files:       {len(files)}")
    print(f"Source pages:      {len(raw_pages)}")
    print(f"Avg chars/chunk:   {avg:.0f}")
    print(f"Shortest chunk:    {lengths[0][0]} chars  ({lengths[0][1]})")
    print(f"Longest chunk:     {lengths[-1][0]} chars  ({lengths[-1][1]})")
    print(f"Skipped quality:   {skipped_quality}")
    print(f"Skipped no-chunks: {skipped_empty}")

    # Sample oldest and newest
    dated = sorted(raw_pages, key=lambda d: d.split("_")[1:])
    for label, doc_id in [("Oldest", dated[0]), ("Newest", dated[-1])]:
        chunk0 = Path(CHUNKS_DIR) / f"{doc_id}_chunk0.json"
        if chunk0.exists():
            with open(chunk0, encoding="utf-8") as fh:
                rec = json.load(fh)
            print(f"\n[{label} — {doc_id}]\n{rec['text'][:400]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and chunk raw OCR pages")
    parser.add_argument("--file", help="Process a single raw JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Print output without writing files")
    args = parser.parse_args()

    if args.file:
        raw_files = [Path(args.file)]
    else:
        raw_files = sorted(Path(RAW_DIR).glob("*.json"))
        print(f"Processing {len(raw_files)} files from {RAW_DIR}/\n")

    run(raw_files, args.dry_run)


if __name__ == "__main__":
    main()
