"""
enrich_with_pubmeta.py — Enrich interaction sentences with publication metadata.

For each sentence in the input CSV that has a PMID or DOI, queries the
Europe PMC REST API to retrieve:
  - pub_year: publication year
  - citation_count: number of times the article has been cited
  - journal: journal name
  - article_type: research-article | review | letter | other
  - is_peer_reviewed: bool

Results are cached in a SQLite DB to avoid repeated API calls.

Usage:
    python enrich_with_pubmeta.py \\
        --input data/training/training_data_v14.csv \\
        --output data/training/training_data_v14_enriched.csv \\
        --cache data/training/pubmeta_cache.db
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

try:
    import requests
except ImportError:
    raise SystemExit("requests not installed — run: pip install requests")


EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_ARTICLE = "https://www.ebi.ac.uk/europepmc/webservices/rest/{source}/{id}/citations/1/json"

RATE_LIMIT_DELAY = 0.35  # seconds between API calls


@dataclass
class PubMeta:
    pmid: str
    pub_year: Optional[int]
    citation_count: Optional[int]
    journal: Optional[str]
    article_type: str  # "research-article" | "review" | "letter" | "other"
    is_peer_reviewed: bool


# ---------------------------------------------------------------------------
# Cache (SQLite)
# ---------------------------------------------------------------------------

def _open_cache(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pubmeta (
            pmid TEXT PRIMARY KEY,
            pub_year INTEGER,
            citation_count INTEGER,
            journal TEXT,
            article_type TEXT,
            is_peer_reviewed INTEGER
        )
    """)
    conn.commit()
    return conn


def _cache_get(conn: sqlite3.Connection, pmid: str) -> Optional[PubMeta]:
    row = conn.execute(
        "SELECT pub_year, citation_count, journal, article_type, is_peer_reviewed "
        "FROM pubmeta WHERE pmid=?", (pmid,)
    ).fetchone()
    if row:
        return PubMeta(
            pmid=pmid,
            pub_year=row[0],
            citation_count=row[1],
            journal=row[2],
            article_type=row[3] or "other",
            is_peer_reviewed=bool(row[4]),
        )
    return None


def _cache_set(conn: sqlite3.Connection, meta: PubMeta) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO pubmeta VALUES (?,?,?,?,?,?)",
        (meta.pmid, meta.pub_year, meta.citation_count, meta.journal,
         meta.article_type, int(meta.is_peer_reviewed)),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# PMID extraction
# ---------------------------------------------------------------------------

_PMID_RE = re.compile(r"(?:pubmed\.ncbi\.nlm\.nih\.gov/|PMID:|pmid:)(\d+)", re.IGNORECASE)
_DOI_RE  = re.compile(r"10\.\d{4,}/\S+")


def extract_pmid(text: str) -> Optional[str]:
    """Extract PMID from a URL, citation string, or plain number."""
    m = _PMID_RE.search(text)
    if m:
        return m.group(1)
    if text.strip().isdigit():
        return text.strip()
    return None


# ---------------------------------------------------------------------------
# EuroPMC query
# ---------------------------------------------------------------------------

def _map_type(pub_type_list: list) -> str:
    types = [t.lower() for t in pub_type_list]
    if any("review" in t for t in types):
        return "review"
    if any("letter" in t or "comment" in t for t in types):
        return "letter"
    if any("research" in t or "article" in t or "journal" in t for t in types):
        return "research-article"
    return "other"


def fetch_pubmeta(pmid: str, session: requests.Session) -> PubMeta:
    """Query EuroPMC for a PMID and return PubMeta."""
    params = {
        "query": f"EXT_ID:{pmid} AND SRC:MED",
        "format": "json",
        "resultType": "core",
        "pageSize": "1",
    }
    try:
        resp = session.get(EPMC_SEARCH, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("resultList", {}).get("result", [])
        if not results:
            return PubMeta(pmid=pmid, pub_year=None, citation_count=None,
                           journal=None, article_type="other", is_peer_reviewed=False)
        r = results[0]
        pub_year = int(r.get("pubYear", 0)) or None
        citation_count = r.get("citedByCount")
        journal = r.get("journalTitle") or r.get("journalInfo", {}).get("journal", {}).get("title")
        pub_types = r.get("pubTypeList", {}).get("pubType", [])
        article_type = _map_type(pub_types)
        is_peer_reviewed = r.get("isOpenAccess") is not None or bool(journal)
        return PubMeta(
            pmid=pmid,
            pub_year=pub_year,
            citation_count=citation_count,
            journal=journal,
            article_type=article_type,
            is_peer_reviewed=is_peer_reviewed,
        )
    except Exception as e:
        return PubMeta(pmid=pmid, pub_year=None, citation_count=None,
                       journal=None, article_type="other", is_peer_reviewed=False)


# ---------------------------------------------------------------------------
# Main enrichment
# ---------------------------------------------------------------------------

def enrich_csv(input_path: Path, output_path: Path, cache_db: Path) -> None:
    conn = _open_cache(cache_db)
    session = requests.Session()
    session.headers["User-Agent"] = "MetaP-TrustScorer/1.0 (biotic interaction research)"

    with open(input_path, newline="", encoding="utf-8") as fin, \
         open(output_path, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        extra_fields = ["pmid", "pub_year", "citation_count", "journal",
                        "article_type", "is_peer_reviewed"]
        writer = csv.DictWriter(fout, fieldnames=(reader.fieldnames or []) + extra_fields)
        writer.writeheader()

        api_calls = 0
        hits = 0
        for i, row in enumerate(reader):
            # Try to find PMID from any field
            pmid = None
            for field_name in ("pmid", "source", "reference", "doi"):
                val = row.get(field_name, "")
                if val:
                    pmid = extract_pmid(val)
                    if pmid:
                        break

            meta: Optional[PubMeta] = None
            if pmid:
                meta = _cache_get(conn, pmid)
                if meta is None:
                    time.sleep(RATE_LIMIT_DELAY)
                    meta = fetch_pubmeta(pmid, session)
                    _cache_set(conn, meta)
                    api_calls += 1
                hits += 1

            row["pmid"] = pmid or ""
            row["pub_year"] = meta.pub_year if meta else ""
            row["citation_count"] = meta.citation_count if meta else ""
            row["journal"] = meta.journal if meta else ""
            row["article_type"] = meta.article_type if meta else "unknown"
            row["is_peer_reviewed"] = meta.is_peer_reviewed if meta else ""
            writer.writerow(row)

            if (i + 1) % 500 == 0:
                print(f"  Processed {i+1:,} rows | {hits} with PMID | {api_calls} API calls",
                      flush=True)

    print(f"Done. Output: {output_path} | API calls: {api_calls}", flush=True)
    conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich interaction CSV with publication metadata")
    parser.add_argument("--input",  required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--cache",  default=str(ROOT / "classifier/data/training/pubmeta_cache.db"),
                        help="SQLite cache path")
    args = parser.parse_args()

    enrich_csv(Path(args.input), Path(args.output), Path(args.cache))


if __name__ == "__main__":
    main()
