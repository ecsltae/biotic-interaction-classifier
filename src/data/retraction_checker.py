"""
retraction_checker.py — Fifth trust axis: retraction and correction detection.

Two data sources (in priority order):
  1. Local Retraction Watch CSV (classifier/data/trust/retraction_watch.csv)
     Columns: PMID, DOI, Title, RetractionDate, Reason, Paywalled
     Download: https://retractionwatch.com/retraction-watch-database-user-guide/
  2. Crossref API  (https://api.crossref.org/works/{doi})
     Checks update-to and relation.is-retraction-of fields.
     Results cached in classifier/data/trust/retraction_cache.db (SQLite).

Score mapping:
  retracted=True  → 0.0
  corrected=True  → 0.6
  clean           → 1.0
  unknown (no doi)→ 0.85  (slight penalty for inability to verify)
"""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DB = ROOT / "classifier/data/trust/retraction_cache.db"
DEFAULT_RW_CSV = ROOT / "classifier/data/trust/retraction_watch.csv"

# Crossref polite pool — include email in User-Agent
_UA = "MetaP-TrustScorer/1.0 (mailto:esteban.gaillac1@gmail.com)"

# In-memory set for Retraction Watch DOIs (loaded once)
_rw_doi_set: Optional[set] = None
_rw_pmid_set: Optional[set] = None
_rw_loaded = False

# Score constants
SCORE_RETRACTED = 0.0
SCORE_CORRECTED = 0.6
SCORE_CLEAN = 1.0
SCORE_UNKNOWN = 0.85  # no DOI — cannot verify


@dataclass
class RetractionResult:
    doi: str
    retracted: bool
    corrected: bool
    retraction_type: Optional[str]  # "retraction" | "erratum" | "correction" | None
    retraction_date: Optional[str]
    source: str  # "retraction_watch" | "crossref" | "cache" | "unknown"
    score: float  # 0.0 / 0.6 / 1.0 / 0.85

    def to_dict(self) -> dict:
        return {
            "doi": self.doi,
            "retracted": self.retracted,
            "corrected": self.corrected,
            "retraction_type": self.retraction_type,
            "retraction_date": self.retraction_date,
            "source": self.source,
            "score": self.score,
        }


# ---------------------------------------------------------------------------
# Retraction Watch local CSV
# ---------------------------------------------------------------------------

def _load_retraction_watch(csv_path: Path = DEFAULT_RW_CSV) -> None:
    global _rw_doi_set, _rw_pmid_set, _rw_loaded
    if _rw_loaded:
        return
    _rw_doi_set = set()
    _rw_pmid_set = set()
    if not csv_path.exists():
        logger.debug("Retraction Watch CSV not found at %s — skipping", csv_path)
        _rw_loaded = True
        return
    try:
        with csv_path.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                doi = (row.get("DOI") or "").strip().lower()
                pmid = (row.get("PMID") or "").strip()
                if doi:
                    _rw_doi_set.add(doi)
                if pmid:
                    _rw_pmid_set.add(pmid)
        logger.info("Retraction Watch: loaded %d DOIs, %d PMIDs", len(_rw_doi_set), len(_rw_pmid_set))
    except Exception as e:
        logger.warning("Failed to load Retraction Watch CSV: %s", e)
    _rw_loaded = True


def _in_retraction_watch(doi: str, pmid: Optional[str] = None) -> bool:
    _load_retraction_watch()
    if _rw_doi_set and doi.lower() in _rw_doi_set:
        return True
    if pmid and _rw_pmid_set and pmid in _rw_pmid_set:
        return True
    return False


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

def _init_cache(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS retraction_cache (
            doi         TEXT PRIMARY KEY,
            retracted   INTEGER,
            corrected   INTEGER,
            retraction_type TEXT,
            retraction_date TEXT,
            source      TEXT,
            queried_at  TEXT
        )
    """)
    conn.commit()
    return conn


def _cache_get(conn: sqlite3.Connection, doi: str) -> Optional[RetractionResult]:
    row = conn.execute(
        "SELECT retracted, corrected, retraction_type, retraction_date, source "
        "FROM retraction_cache WHERE doi=?",
        (doi.lower(),)
    ).fetchone()
    if row is None:
        return None
    retracted, corrected, rtype, rdate, source = row
    score = SCORE_RETRACTED if retracted else (SCORE_CORRECTED if corrected else SCORE_CLEAN)
    return RetractionResult(
        doi=doi,
        retracted=bool(retracted),
        corrected=bool(corrected),
        retraction_type=rtype,
        retraction_date=rdate,
        source="cache",
        score=score,
    )


def _cache_put(conn: sqlite3.Connection, result: RetractionResult) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO retraction_cache "
        "(doi, retracted, corrected, retraction_type, retraction_date, source, queried_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (
            result.doi.lower(),
            int(result.retracted),
            int(result.corrected),
            result.retraction_type,
            result.retraction_date,
            result.source,
            time.strftime("%Y-%m-%dT%H:%M:%S"),
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Crossref API query
# ---------------------------------------------------------------------------

def _query_crossref(doi: str, session: Optional[requests.Session] = None) -> RetractionResult:
    """Query Crossref /works/{doi} and check for retraction/correction notices.

    A retracted paper is flagged in the ``updated-by`` array (this work WAS updated
    by a retraction/correction notice) — NOT ``update-to`` (which lists works that
    THIS work retracts, i.e. it marks retraction notices themselves). Crossref
    ingests Retraction Watch, so ``updated-by`` entries carry source=retraction-watch.
    """
    url = f"https://api.crossref.org/works/{requests.utils.quote(doi, safe='')}"
    sess = session or requests.Session()
    try:
        r = sess.get(url, headers={"User-Agent": _UA}, timeout=10)
        if r.status_code == 404:
            # A DOI that does not resolve in Crossref cannot be verified — treat as
            # a red flag (possibly fabricated / non-Crossref), not as clean.
            return RetractionResult(doi=doi, retracted=False, corrected=False,
                                    retraction_type=None, retraction_date=None,
                                    source="doi_not_found", score=SCORE_UNKNOWN)
        if not r.ok:
            logger.warning("Crossref %s returned %s", doi, r.status_code)
            return RetractionResult(doi=doi, retracted=False, corrected=False,
                                    retraction_type=None, retraction_date=None,
                                    source="unknown", score=SCORE_UNKNOWN)
        work = r.json().get("message", {})
    except Exception as e:
        logger.warning("Crossref query failed for %s: %s", doi, e)
        return RetractionResult(doi=doi, retracted=False, corrected=False,
                                retraction_type=None, retraction_date=None,
                                source="unknown", score=SCORE_UNKNOWN)

    retracted = False
    corrected = False
    rtype: Optional[str] = None
    rdate: Optional[str] = None

    # ``updated-by``: notices that updated THIS work (retraction / correction).
    for upd in work.get("updated-by", []):
        utype = (upd.get("type") or "").lower()
        udate = upd.get("updated", {}).get("date-time", "")[:10]
        if utype in ("retraction", "withdrawal", "removal"):
            retracted = True
            rtype = "retraction"
            rdate = udate
            break  # retraction dominates
        elif utype in ("correction", "erratum", "expression_of_concern",
                       "expression of concern"):
            corrected = True
            rtype = utype.replace(" ", "_")
            rdate = udate

    # If the work's own type is a retraction notice, it isn't a "retracted paper"
    # to be scored — leave as clean; the retracted original is flagged via updated-by.

    score = SCORE_RETRACTED if retracted else (SCORE_CORRECTED if corrected else SCORE_CLEAN)
    return RetractionResult(
        doi=doi,
        retracted=retracted,
        corrected=corrected,
        retraction_type=rtype,
        retraction_date=rdate,
        source="crossref",
        score=score,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_retraction(
    doi: str,
    cache_db: Optional[Path] = None,
    pmid: Optional[str] = None,
    session: Optional[requests.Session] = None,
) -> RetractionResult:
    """
    Check if a paper is retracted or corrected.

    Args:
        doi:      Digital Object Identifier (required)
        cache_db: Path to SQLite cache (default: classifier/data/trust/retraction_cache.db)
        pmid:     PubMed ID (used for Retraction Watch lookup if available)
        session:  requests.Session to reuse across calls

    Returns:
        RetractionResult with retracted/corrected flags and composite score
    """
    if not doi:
        return RetractionResult(doi="", retracted=False, corrected=False,
                                retraction_type=None, retraction_date=None,
                                source="unknown", score=SCORE_UNKNOWN)

    doi = doi.strip()
    db_path = cache_db or DEFAULT_CACHE_DB
    _load_retraction_watch()

    # Priority 1: local Retraction Watch CSV (instant, no network)
    if _in_retraction_watch(doi, pmid):
        result = RetractionResult(doi=doi, retracted=True, corrected=False,
                                  retraction_type="retraction", retraction_date=None,
                                  source="retraction_watch", score=SCORE_RETRACTED)
        try:
            conn = _init_cache(db_path)
            _cache_put(conn, result)
            conn.close()
        except Exception:
            pass
        return result

    # Priority 2: SQLite cache
    try:
        conn = _init_cache(db_path)
        cached = _cache_get(conn, doi)
        if cached is not None:
            conn.close()
            return cached
    except Exception as e:
        logger.warning("Cache error: %s", e)
        conn = None

    # Priority 3: Crossref API
    result = _query_crossref(doi, session)
    try:
        if conn:
            _cache_put(conn, result)
            conn.close()
    except Exception:
        pass

    return result


def check_retraction_batch(
    dois: List[str],
    cache_db: Optional[Path] = None,
    pmids: Optional[List[Optional[str]]] = None,
) -> List[RetractionResult]:
    """
    Score multiple DOIs. Reuses a single requests.Session and cache connection.
    pmids list (optional) must be same length as dois if provided.
    """
    db_path = cache_db or DEFAULT_CACHE_DB
    _load_retraction_watch()
    results: List[RetractionResult] = []

    try:
        conn = _init_cache(db_path)
    except Exception:
        conn = None

    session = requests.Session()
    session.headers.update({"User-Agent": _UA})

    for i, doi in enumerate(dois):
        pmid = pmids[i] if pmids and i < len(pmids) else None
        if not doi:
            results.append(RetractionResult(doi="", retracted=False, corrected=False,
                                            retraction_type=None, retraction_date=None,
                                            source="unknown", score=SCORE_UNKNOWN))
            continue

        doi = doi.strip()

        if _in_retraction_watch(doi, pmid):
            r = RetractionResult(doi=doi, retracted=True, corrected=False,
                                 retraction_type="retraction", retraction_date=None,
                                 source="retraction_watch", score=SCORE_RETRACTED)
            if conn:
                try:
                    _cache_put(conn, r)
                except Exception:
                    pass
            results.append(r)
            continue

        if conn:
            try:
                cached = _cache_get(conn, doi)
                if cached is not None:
                    results.append(cached)
                    continue
            except Exception:
                pass

        r = _query_crossref(doi, session)
        if conn:
            try:
                _cache_put(conn, r)
            except Exception:
                pass
        results.append(r)
        time.sleep(0.05)  # 20 req/s — well within Crossref polite pool 50/s

    if conn:
        try:
            conn.close()
        except Exception:
            pass

    return results


def retraction_score(doi: Optional[str], pmid: Optional[str] = None) -> float:
    """Convenience function: return 0-1 retraction score for a DOI."""
    if not doi:
        return SCORE_UNKNOWN
    return check_retraction(doi, pmid=pmid).score


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    test_dois = sys.argv[1:] if len(sys.argv) > 1 else [
        "10.3354/dao002147",               # clean — GloBI
        "10.1016/j.vaccine.2011.11.001",   # Wakefield retraction (if discoverable)
        "10.1126/science.1255768",         # known correction
    ]
    print(f"{'DOI':<45} {'Retracted':>10} {'Corrected':>10} {'Score':>6}  Source")
    print("-" * 90)
    for doi in test_dois:
        r = check_retraction(doi)
        print(f"{doi:<45} {str(r.retracted):>10} {str(r.corrected):>10} {r.score:>6.2f}  {r.source}")
