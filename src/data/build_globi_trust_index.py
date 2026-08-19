#!/usr/bin/env python3
"""
Build a GloBI-based trust index: extract DOIs from interactions.tsv.gz,
query EuroPMC for publication year + article type, cache results.

Output: classifier/data/globi/globi_trust_index.csv
  Columns: doi, pub_year, article_type, citation_count, source_species, target_species, interaction_type

Usage:
    python classifier/src/data/build_globi_trust_index.py
    python classifier/src/data/build_globi_trust_index.py --max-dois 5000 --interaction-types commensalistOf parasiteOf
"""
from __future__ import annotations

import argparse
import csv
import gzip

csv.field_size_limit(10_000_000)
import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GLOBI_TSV = Path("classifier/data/globi/interactions.tsv.gz")
CACHE_DB = Path("classifier/data/globi/epmc_doi_cache.db")
OUT_CSV = Path("classifier/data/globi/globi_trust_index.csv")
EPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

DOI_RE = re.compile(r"10\.\d{4,}/\S+")


def init_cache(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS doi_meta (
            doi TEXT PRIMARY KEY,
            pub_year INTEGER,
            article_type TEXT,
            citation_count INTEGER,
            title TEXT,
            queried_at TEXT
        )
    """)
    conn.commit()
    return conn


def lookup_doi(doi: str, conn: sqlite3.Connection, session: requests.Session) -> dict:
    """Look up DOI in cache or EuroPMC."""
    row = conn.execute("SELECT * FROM doi_meta WHERE doi=?", (doi,)).fetchone()
    if row:
        return {"doi": row[0], "pub_year": row[1], "article_type": row[2],
                "citation_count": row[3], "title": row[4]}

    # Query EuroPMC
    params = {
        "query": f"DOI:{doi}",
        "format": "json",
        "pageSize": 1,
        "resultType": "core",
    }
    try:
        r = session.get(EPMC_API, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        results = data.get("resultList", {}).get("result", [])
        if results:
            art = results[0]
            pub_year = int(art.get("pubYear", 0)) or None
            article_type = art.get("pubTypeList", {}).get("pubType", ["unknown"])[0] if isinstance(
                art.get("pubTypeList"), dict) else "unknown"
            if isinstance(article_type, list):
                article_type = article_type[0] if article_type else "unknown"
            citation_count = art.get("citedByCount", 0)
            title = art.get("title", "")[:200]
        else:
            pub_year, article_type, citation_count, title = None, "not_found", 0, ""
    except Exception as e:
        logger.debug(f"EuroPMC error for {doi}: {e}")
        pub_year, article_type, citation_count, title = None, "api_error", 0, ""

    conn.execute(
        "INSERT OR REPLACE INTO doi_meta VALUES (?,?,?,?,?,datetime('now'))",
        (doi, pub_year, article_type, citation_count, title),
    )
    conn.commit()
    return {"doi": doi, "pub_year": pub_year, "article_type": article_type,
            "citation_count": citation_count, "title": title}


def extract_doi(citation: str, doi_field: str) -> Optional[str]:
    """Extract DOI from GloBI fields."""
    if doi_field and doi_field.startswith("10."):
        return doi_field.strip().lower()
    # Try regex on citation string
    m = DOI_RE.search(citation or "")
    return m.group(0).lower().rstrip(".") if m else None


def build_index(max_dois: int = 10_000, interaction_filter: set[str] | None = None,
                resume: bool = True) -> None:
    conn = init_cache(CACHE_DB)
    session = requests.Session()
    session.headers["User-Agent"] = "MetaP-TrustIndex/1.0 (esteban.gaillac1@gmail.com)"

    # Count already cached DOIs
    cached = conn.execute("SELECT COUNT(*) FROM doi_meta").fetchone()[0]
    logger.info(f"Cache has {cached} DOIs already")

    doi_to_pairs: dict[str, list[dict]] = {}
    n_scanned = 0

    logger.info(f"Scanning {GLOBI_TSV} ...")
    cols_needed = {"sourceTaxonSpeciesName", "targetTaxonSpeciesName", "interactionTypeName",
                   "referenceCitation", "referenceDoi"}

    with gzip.open(GLOBI_TSV, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        while True:
            try:
                row = next(reader)
            except StopIteration:
                break
            except csv.Error:
                continue  # skip malformed rows with oversized fields
            n_scanned += 1
            if n_scanned % 1_000_000 == 0:
                logger.info(f"  Scanned {n_scanned:,} rows, {len(doi_to_pairs)} unique DOIs")

            itype = row.get("interactionTypeName", "")
            if interaction_filter and itype not in interaction_filter:
                continue

            doi = extract_doi(row.get("referenceCitation", ""), row.get("referenceDoi", ""))
            if not doi:
                continue

            if doi not in doi_to_pairs:
                if len(doi_to_pairs) >= max_dois:
                    continue
                doi_to_pairs[doi] = []

            doi_to_pairs[doi].append({
                "source_species": row.get("sourceTaxonSpeciesName", ""),
                "target_species": row.get("targetTaxonSpeciesName", ""),
                "interaction_type": itype,
            })

    logger.info(f"Found {len(doi_to_pairs)} unique DOIs from {n_scanned:,} GloBI rows")

    # Query EuroPMC for DOIs not yet cached
    uncached_dois = [d for d in doi_to_pairs if not conn.execute(
        "SELECT 1 FROM doi_meta WHERE doi=?", (d,)).fetchone()]
    logger.info(f"Querying EuroPMC for {len(uncached_dois)} uncached DOIs ...")

    for i, doi in enumerate(uncached_dois):
        lookup_doi(doi, conn, session)
        if i % 100 == 0:
            logger.info(f"  {i}/{len(uncached_dois)} queried")
        time.sleep(0.2)  # EuroPMC rate limit

    # Build output CSV
    logger.info(f"Writing trust index → {OUT_CSV}")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=[
            "doi", "pub_year", "article_type", "citation_count",
            "source_species", "target_species", "interaction_type", "title"])
        writer.writeheader()
        for doi, pairs in doi_to_pairs.items():
            meta = conn.execute("SELECT pub_year,article_type,citation_count,title FROM doi_meta WHERE doi=?",
                                (doi,)).fetchone()
            if not meta:
                continue
            pub_year, article_type, citation_count, title = meta
            # One row per unique (source, target, interaction_type) per DOI
            seen = set()
            for pair in pairs:
                key = (pair["source_species"], pair["target_species"], pair["interaction_type"])
                if key in seen:
                    continue
                seen.add(key)
                writer.writerow({
                    "doi": doi, "pub_year": pub_year, "article_type": article_type,
                    "citation_count": citation_count, "title": title,
                    **pair,
                })
                rows_written += 1

    logger.info(f"Done. {rows_written:,} rows written to {OUT_CSV}")
    conn.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Build GloBI-based EuroPMC trust index")
    p.add_argument("--max-dois", type=int, default=10_000,
                   help="Max unique DOIs to process (default: 10,000)")
    p.add_argument("--interaction-types", nargs="+", default=None,
                   help="Filter to specific GloBI interaction types")
    p.add_argument("--no-resume", action="store_true",
                   help="Re-query all DOIs even if cached")
    args = p.parse_args()

    build_index(
        max_dois=args.max_dois,
        interaction_filter=set(args.interaction_types) if args.interaction_types else None,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
