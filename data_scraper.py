import argparse
import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import requests

from config import BASE_URL, DELAY, MAX_PAGES, RAW_DIR, TITLES, RESTRICTED_LCCNS

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "TAMULibraryResearch/1.0 (research harvester)"})
TIMEOUT = 20

def fetch_issue_dates(lccn: str) -> list[str]:
    """Return sorted list of all issue date strings for a title."""
    url = f"{BASE_URL}/lccn/{lccn}.json"
    try:
        r = SESSION.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        manifests = r.json().get("manifests", [])
        if not manifests:
            print(f"  WARNING: no issues found for {lccn}")
            return []
        dates = sorted(m["label"] for m in manifests if m.get("label"))
        return dates
    except Exception as e:
        print(f"  WARNING: could not fetch issue list for {lccn}: {e}")
        return []


def fetch_issue_manifest(lccn: str, date: str) -> dict | None:
    """Return the IIIF manifest for one issue, or None on failure."""
    url = f"{BASE_URL}/lccn/{lccn}/{date}/ed-1.json"
    try:
        r = SESSION.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  WARNING: manifest failed {lccn}/{date}: {e}")
        return None


def fetch_ocr(ocr_url: str) -> str | None:
    """Fetch and parse ALTO XML OCR for one page. Returns plain text or None."""
    try:
        r = SESSION.get(ocr_url, timeout=TIMEOUT)
        r.raise_for_status()
        text = _parse_alto_xml(r.text)
        return text if len(text) >= 100 else None
    except Exception:
        return None


def _parse_alto_xml(xml_text: str) -> str:
    """Extract word strings from ALTO XML format."""
    try:
        root = ET.fromstring(xml_text)
        words = [
            elem.attrib["CONTENT"]
            for elem in root.iter()
            if elem.tag.endswith("String") and "CONTENT" in elem.attrib
        ]
        return " ".join(words)
    except Exception:
        return ""
    

def get_canvases(manifest: dict) -> list:
    """Return list of page canvases from a IIIF manifest."""
    return manifest.get("sequences", [{}])[0].get("canvases", [])


def _canvas_url(canvas: dict, fmt: str, key: str) -> str | None:
    """Extract a URL matching fmt from a canvas resource list."""
    for image in canvas.get("images", []):
        for item in image.get(key, []):
            if fmt in item.get("@id", "") or item.get("format") == fmt:
                return item["@id"]
    return None


def get_ocr_url(canvas: dict) -> str | None:
    """Return the OCR XML URL from a canvas seeAlso list."""
    return _canvas_url(canvas, "ocr.xml", "seeAlso")


def get_page_url(canvas: dict, lccn: str, date: str, seq: int) -> str:
    """Return the human-readable page URL for citation purposes."""
    return (
        _canvas_url(canvas, "text/html", "rendering")
        or f"{BASE_URL}/lccn/{lccn}/{date}/ed-1/seq-{seq}/"
    )


def filter_dates(
    dates: list[str],
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[str]:
    """
    Filter issue dates to a specific range.
    Accepts YYYY (expanded to YYYY-01-01 / YYYY-12-31) or YYYY-MM-DD.
    """
    if not date_from and not date_to:
        return dates

    from_str = None
    to_str   = None

    if date_from:
        from_str = f"{date_from[:4]}-01-01" if len(date_from) == 4 else date_from
    if date_to:
        to_str = f"{date_to[:4]}-12-31" if len(date_to) == 4 else date_to

    filtered = dates
    if from_str:
        filtered = [d for d in filtered if d >= from_str]
    if to_str:
        filtered = [d for d in filtered if d <= to_str]

    return filtered


def sample_dates(dates: list[str], n: int) -> list[str]:
    """
    Pick n dates spread evenly across the full date range.
    Deterministic — same input always produces same output.
    """
    if len(dates) <= n:
        return dates
    step = len(dates) / n
    return [dates[int(i * step)] for i in range(n)]


def _raw_path(lccn: str, date: str, seq: int) -> Path:
    return Path(RAW_DIR) / f"{lccn}_{date}_seq{seq}.json"


def already_saved(lccn: str, date: str, seq: int) -> bool:
    return _raw_path(lccn, date, seq).exists()


def save_page(record: dict) -> None:
    path = _raw_path(record["lccn"], record["date"], record["page"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def estimate_time(n_issues: int, avg_pages_per_issue: int, delay: float) -> str:
    """Return a human-readable time estimate for scraping n issues."""
    total_requests = n_issues * (1 + avg_pages_per_issue)  # manifest + pages
    total_seconds  = total_requests * delay
    hours   = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    if hours > 0:
        return f"~{hours}h {minutes}m"
    return f"~{minutes}m"


def harvest(
    lccn_list:   list[str],
    max_pages:   int,
    delay:       float,
    date_from:   str | None = None,
    date_to:     str | None = None,
    full:        bool = False,
) -> None:
    """
    Main harvest loop.

    full=True  → every issue, every page (no sampling, no page limit)
    full=False → sample dates evenly, up to max_pages total
    """
    issues_per_title = max(1, max_pages // len(lccn_list))
    total_saved   = 0
    total_skipped = 0

    for lccn in lccn_list:
        title_name = TITLES.get(lccn, lccn)

        if lccn in RESTRICTED_LCCNS:
            print(f"\n=== {title_name} ({lccn}) — SKIPPED (restricted, on-campus only) ===")
            continue

        print(f"\n=== {title_name} ({lccn}) ===")

        all_dates = fetch_issue_dates(lccn)
        if not all_dates:
            print("  No dates found — skipping.")
            continue

        dates = filter_dates(all_dates, date_from, date_to)
        if not dates:
            print(f"  No issues in range {date_from} → {date_to} — skipping.")
            continue

        if full:
            target_dates = dates
            print(f"  {len(all_dates)} total issues | {len(dates)} in range | FULL mode — scraping all")
        else:
            target_dates = sample_dates(dates, issues_per_title)
            print(f"  {len(all_dates)} total issues | {len(dates)} in range | sampling {len(target_dates)}")

        est = estimate_time(len(target_dates), 4, delay)
        print(f"  Estimated time: {est}")

        saved     = 0
        skipped   = 0
        issue_num = 0

        for date in target_dates:
            issue_num += 1
            time.sleep(delay)

            manifest = fetch_issue_manifest(lccn, date)
            if not manifest:
                skipped += 1
                continue

            canvases = get_canvases(manifest)
            if not canvases:
                skipped += 1
                continue

            issue_title = manifest.get("label", title_name)
            issue_saved = 0

            for seq, canvas in enumerate(canvases, start=1):
                if already_saved(lccn, date, seq):
                    skipped += 1
                    continue

                ocr_url = get_ocr_url(canvas)
                if not ocr_url:
                    skipped += 1
                    continue

                time.sleep(delay)
                ocr = fetch_ocr(ocr_url)
                if ocr is None:
                    skipped += 1
                    continue

                record = {
                    "doc_id":    f"{lccn}_{date}_seq{seq}",
                    "lccn":      lccn,
                    "title":     issue_title,
                    "date":      date,
                    "page":      seq,
                    "url":       get_page_url(canvas, lccn, date, seq),
                    "ocr_text":  ocr,
                    "source":    "open_oni",
                    "harvested": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                }

                save_page(record)
                saved       += 1
                issue_saved += 1
                total_saved += 1

                print(
                    f"  [{issue_num}/{len(target_dates)} issues | "
                    f"{saved} pages saved] "
                    f"✓ {date} seq{seq} ({len(ocr):,} chars)"
                )

        total_skipped += skipped
        print(f"\n  {title_name} complete: {saved} pages saved, {skipped} skipped")

    print(f"\n{'='*60}")
    print(f"Harvest complete.")
    print(f"  Total pages saved : {total_saved}")
    print(f"  Total skipped     : {total_skipped}")
    print(f"  Output directory  : {RAW_DIR}/")
    print(f"  Total files now   : {len(list(Path(RAW_DIR).glob('*.json')))}")


def validate_output() -> None:
    raw   = Path(RAW_DIR)
    files = list(raw.glob("*.json"))
    print(f"\n--- Validation ---")
    print(f"Files in {RAW_DIR}/: {len(files)}")

    if not files:
        print("No files to validate.")
        return

    required_keys = {
        "doc_id", "lccn", "title", "date", "page",
        "url", "ocr_text", "source", "harvested"
    }

    print("Checking 3 sample files for required keys...")
    for path in files[:3]:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        missing = required_keys - rec.keys()
        status  = "OK" if not missing else f"MISSING: {missing}"
        print(f"  {path.name}: {status}")

    ocr_lengths = []
    dates_seen  = []
    lccns_seen  = {}

    for path in files:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        ocr_lengths.append((len(rec.get("ocr_text", "")), path.name))
        dates_seen.append(rec.get("date", ""))
        lccn = rec.get("lccn", "unknown")
        lccns_seen[lccn] = lccns_seen.get(lccn, 0) + 1

    ocr_lengths.sort()
    print(f"\n  Shortest OCR : {ocr_lengths[0][0]:,} chars  ({ocr_lengths[0][1]})")
    print(f"  Longest OCR  : {ocr_lengths[-1][0]:,} chars  ({ocr_lengths[-1][1]})")

    dates_seen = sorted(d for d in dates_seen if d)
    if dates_seen:
        print(f"  Date range   : {dates_seen[0]} → {dates_seen[-1]}")

    print(f"\n  Pages per title:")
    for lccn, count in sorted(lccns_seen.items()):
        title = TITLES.get(lccn, lccn)
        print(f"    {title}: {count:,} pages")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harvest OCR pages from TAMU Open ONI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_scraper.py                                    # 100-page pilot
  python data_scraper.py --max-pages 20                    # quick smoke test
  python data_scraper.py --lccn sn86088544 --date-from 1893 --date-to 1900
  python data_scraper.py --lccn sn86088544 --full          # full Battalion (~15 hours)
  python data_scraper.py --lccn sn37020267 --full          # full Daily Bulletin (~3 hours)
  python data_scraper.py --lccn sn84009780 --full          # full Texas Aggie (~5 hours)
        """
    )

    parser.add_argument(
        "--lccn", nargs="+", default=list(TITLES.keys()),
        help="One or more LCCNs to harvest (default: all known titles)"
    )
    parser.add_argument(
        "--max-pages", type=int, default=MAX_PAGES,
        help=f"Total issues to sample across all titles (default: {MAX_PAGES}). Ignored when --full is set."
    )
    parser.add_argument(
        "--delay", type=float, default=DELAY,
        help=f"Seconds to wait between requests (default: {DELAY})"
    )
    parser.add_argument(
        "--date-from", type=str, default=None,
        help="Only harvest issues on or after this date. Accepts YYYY or YYYY-MM-DD."
    )
    parser.add_argument(
        "--date-to", type=str, default=None,
        help="Only harvest issues on or before this date. Accepts YYYY or YYYY-MM-DD."
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Scrape every issue and every page — no sampling. Use for full collection harvest."
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip the validation summary at the end (useful for long overnight runs)."
    )

    args = parser.parse_args()

    if args.full:
        print("FULL MODE — scraping every issue and every page.")
        print("This will run for several hours. Safe to stop and restart at any time.")
        print(f"Delay between requests: {args.delay}s")
        print()

    harvest(
        lccn_list = args.lccn,
        max_pages = args.max_pages,
        delay     = args.delay,
        date_from = args.date_from,
        date_to   = args.date_to,
        full      = args.full,
    )

    if not args.no_validate:
        validate_output()


if __name__ == "__main__":
    main()
