"""
doc_resolver.py — Resolve any document reference to a canonical DOI.

The trust pipeline is DOI-keyed (Crossref for references, retraction checks, etc.),
but users refer to documents in many ways: DOI, PMID, PMCID, or just a title. This
module mirrors the dispatch pattern of BioMoQA-RAG's DocResolver and resolves any of
those to a canonical DOI (+ pmid/pmcid/title) via a single Europe PMC search endpoint,
which indexes all four identifier types and returns cross-mapped IDs.

    from classifier.src.data.doc_resolver import resolve_reference
    r = resolve_reference("Highly accurate protein structure prediction with AlphaFold")
    r.doi  -> "10.1038/s41586-021-03819-2"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_UA = "MetaP-TrustScorer/1.0 (mailto:esteban.gaillac1@gmail.com)"
_EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# Identifier shapes
_RE_PMID = re.compile(r"^\d{1,8}$")
_RE_PMCID = re.compile(r"^PMC\d+$", re.IGNORECASE)
_RE_DOI = re.compile(r"^(https?://(dx\.)?doi\.org/)?10\.\d{4,}/\S+$", re.IGNORECASE)


@dataclass
class ResolvedDoc:
    ref: str                    # original input
    id_type: str                # "doi" | "pmid" | "pmcid" | "title"
    doi: Optional[str]
    pmid: Optional[str]
    pmcid: Optional[str]
    title: str
    matched: bool               # a document was found
    score: float                # Europe PMC relevance (meaningful for title matches)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "ref": self.ref, "id_type": self.id_type, "doi": self.doi,
            "pmid": self.pmid, "pmcid": self.pmcid, "title": self.title,
            "matched": self.matched, "score": round(self.score, 2), "note": self.note,
        }


def _epmc_query(query: str, session: requests.Session) -> Optional[dict]:
    try:
        r = session.get(
            _EPMC_SEARCH,
            params={"query": query, "resultType": "core", "format": "json", "pageSize": 1},
            headers={"User-Agent": _UA}, timeout=15,
        )
        if r.ok:
            results = r.json().get("resultList", {}).get("result", [])
            return results[0] if results else None
    except Exception as e:
        logger.debug("EPMC query failed (%s): %s", query, e)
    return None


def _clean_doi(doi: str) -> str:
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi.strip(), flags=re.IGNORECASE)


def _from_record(ref: str, id_type: str, rec: dict, fallback_doi: str = "") -> ResolvedDoc:
    doi = (rec.get("doi") or fallback_doi) or None
    return ResolvedDoc(
        ref=ref, id_type=id_type, doi=doi,
        pmid=rec.get("pmid"), pmcid=rec.get("pmcid"),
        title=(rec.get("title") or "").strip().rstrip("."),
        matched=bool(doi),
        score=float(rec.get("score", 0) or 0),
    )


def resolve_reference(ref: str, session: Optional[requests.Session] = None) -> ResolvedDoc:
    """Resolve any document reference to a canonical DOI (+ ids/title)."""
    ref = (ref or "").strip()
    sess = session or requests.Session()
    sess.headers.update({"User-Agent": _UA})

    if not ref:
        return ResolvedDoc(ref, "empty", None, None, None, "", False, 0.0,
                           note="Empty reference.")

    # DOI (possibly a doi.org URL)
    if _RE_DOI.match(ref):
        doi = _clean_doi(ref)
        rec = _epmc_query(f'DOI:"{doi}"', sess) or {}
        # Pass the DOI through even if EPMC lacks the record — Crossref will handle it.
        out = _from_record(ref, "doi", rec, fallback_doi=doi)
        if not rec:
            out.title = out.title or ""
            out.note = "DOI not in Europe PMC; will resolve metadata via Crossref."
            out.matched = True
        return out

    # PMCID
    if _RE_PMCID.match(ref):
        rec = _epmc_query(f"PMCID:{ref.upper()}", sess)
        if rec:
            return _from_record(ref, "pmcid", rec)
        return ResolvedDoc(ref, "pmcid", None, None, ref.upper(), "", False, 0.0,
                           note="PMCID not found in Europe PMC.")

    # PMID
    if _RE_PMID.match(ref):
        rec = _epmc_query(f"EXT_ID:{ref} AND SRC:MED", sess)
        if rec:
            r = _from_record(ref, "pmid", rec)
            if not r.doi:
                r.note = "Article has no DOI; trust checks that need a DOI are limited."
            return r
        return ResolvedDoc(ref, "pmid", None, ref, None, "", False, 0.0,
                           note="PMID not found in Europe PMC.")

    # Free text → title search
    rec = _epmc_query(ref, sess)
    if rec:
        r = _from_record(ref, "title", rec)
        r.note = f"Resolved by title match to: {r.title}"
        return r
    return ResolvedDoc(ref, "title", None, None, None, "", False, 0.0,
                       note="No document matched this title.")


if __name__ == "__main__":
    import sys
    tests = sys.argv[1:] or [
        "10.1038/s41586-021-03819-2",
        "19815723",
        "PMC7398426",
        "Highly accurate protein structure prediction with AlphaFold",
    ]
    for t in tests:
        r = resolve_reference(t)
        print(f"[{r.id_type:6s}] {t[:45]:45s} -> doi={r.doi} pmid={r.pmid} matched={r.matched}")
        if r.note:
            print(f"          {r.note}")
