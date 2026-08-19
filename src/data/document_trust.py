"""
document_trust.py — Document-level trust scoring for "truth in the literature".

Where trust_scorer.py scores a single CLAIM (sentence + species pair), this module
scores a whole DOCUMENT and, crucially, the literature it rests on.

The central idea: a paper's trustworthiness depends not only on whether it is itself
retracted, but on whether its *cited foundation* is sound. A paper that cites retracted
work, or builds entirely on decades-old evidence, is a weaker foundation for downstream
claims — even if the paper itself was never retracted.

Two trust dimensions:
  A. SELF trust     — is this document retracted? how old? (retraction_checker + temporal)
  B. FOUNDATION trust — of the works it cites, how many are retracted / how old?

Data sources:
  - Crossref /works/{doi}: metadata + reference list (references carry DOIs inline)
  - retraction_checker.check_retraction_batch(): retraction status for self + references

Usage:
    from classifier.src.data.document_trust import score_document
    report = score_document("10.1016/S0140-6736(97)11096-0")
    print(report.grade, report.overall_trust, report.red_flags)
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import List, Optional

import requests

from src.data.retraction_checker import check_retraction, check_retraction_batch
from src.data.trust_scorer import temporal_score, CURRENT_YEAR
from src.data.citation_context import get_citation_contexts

logger = logging.getLogger(__name__)

_UA = "MetaP-TrustScorer/1.0 (mailto:esteban.gaillac1@gmail.com)"
_CROSSREF = "https://api.crossref.org/works/"
_OPENALEX = "https://api.openalex.org/works/doi:"

# How much each dimension contributes to the overall document trust score.
DOC_WEIGHTS = {
    "self_retraction": 0.35,   # is the paper itself retracted (hard signal)
    "self_temporal":   0.15,   # how old is the paper
    "foundation_retraction": 0.30,  # does it cite retracted work
    "foundation_temporal":   0.20,  # how current is its cited evidence base
}


@dataclass
class ReferenceTrust:
    doi: str
    year: Optional[int]
    retracted: bool
    retraction_type: Optional[str]
    title: str = ""


@dataclass
class DocumentTrustReport:
    doi: str
    title: str
    year: Optional[int]
    venue: str

    # Self dimension
    self_retracted: bool
    self_retraction_type: Optional[str]
    self_temporal_score: float

    # Foundation dimension
    n_references: int
    n_references_with_doi: int
    n_retracted_references: int
    retracted_references: List[ReferenceTrust]
    median_reference_age: Optional[float]
    pct_references_last_10yr: Optional[float]
    foundation_retraction_score: float
    foundation_temporal_score: float

    # Composite
    overall_trust: float
    grade: str                       # A / B / C / D / F
    red_flags: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "doi": self.doi,
            "title": self.title,
            "year": self.year,
            "venue": self.venue,
            "overall_trust": round(self.overall_trust, 3),
            "grade": self.grade,
            "self": {
                "retracted": self.self_retracted,
                "retraction_type": self.self_retraction_type,
                "temporal_score": round(self.self_temporal_score, 3),
            },
            "foundation": {
                "n_references": self.n_references,
                "n_references_with_doi": self.n_references_with_doi,
                "n_retracted_references": self.n_retracted_references,
                "retracted_references": [
                    {"doi": r.doi, "year": r.year, "type": r.retraction_type,
                     "title": r.title}
                    for r in self.retracted_references
                ],
                "median_reference_age": self.median_reference_age,
                "pct_references_last_10yr": self.pct_references_last_10yr,
                "retraction_score": round(self.foundation_retraction_score, 3),
                "temporal_score": round(self.foundation_temporal_score, 3),
            },
            "red_flags": self.red_flags,
            "notes": self.notes,
        }


def _fetch_crossref(doi: str, session: requests.Session) -> Optional[dict]:
    try:
        r = session.get(
            f"{_CROSSREF}{requests.utils.quote(doi, safe='')}",
            headers={"User-Agent": _UA}, timeout=15,
        )
        if r.ok:
            return r.json().get("message", {})
        logger.warning("Crossref %s → %s", doi, r.status_code)
    except Exception as e:
        logger.warning("Crossref fetch failed for %s: %s", doi, e)
    return None


def _year_from_crossref(work: dict) -> Optional[int]:
    for key in ("published-print", "published-online", "published", "issued", "created"):
        parts = (work.get(key) or {}).get("date-parts") or [[]]
        if parts and parts[0] and parts[0][0]:
            return int(parts[0][0])
    return None


def _fetch_openalex(doi: str, session: requests.Session) -> dict:
    """Fetch venue, retraction flag, citation count from OpenAlex (corroborating source)."""
    try:
        r = session.get(f"{_OPENALEX}{doi}", params={"mailto": "esteban.gaillac1@gmail.com"},
                        timeout=15)
        if r.ok:
            w = r.json()
            src = (w.get("primary_location") or {}).get("source") or {}
            return {
                "is_retracted": w.get("is_retracted"),
                "venue": src.get("display_name", ""),
                "is_in_doaj": src.get("is_in_doaj"),
                "cited_by_count": w.get("cited_by_count"),
                "type": w.get("type"),
            }
    except Exception as e:
        logger.debug("OpenAlex fetch failed for %s: %s", doi, e)
    return {}


def _abstract_from_crossref(work: dict) -> str:
    """Crossref abstracts are JATS-wrapped; strip tags."""
    raw = work.get("abstract", "") or ""
    if not raw:
        return ""
    import re as _re
    return _re.sub(r"<[^>]+>", " ", _re.sub(r"\s+", " ", raw)).strip()


def _retraction_notice_title(work: dict, session: requests.Session) -> str:
    """Fetch the title of the retraction/correction notice (often states the reason)."""
    for upd in work.get("updated-by", []):
        if (upd.get("type") or "").lower() in ("retraction", "withdrawal", "removal"):
            ndoi = upd.get("DOI")
            if ndoi:
                notice = _fetch_crossref(ndoi, session) or {}
                titles = notice.get("title") or []
                if titles:
                    return titles[0]
    return ""


def _grade(score: float) -> str:
    if score >= 0.85:
        return "A"
    if score >= 0.70:
        return "B"
    if score >= 0.55:
        return "C"
    if score >= 0.40:
        return "D"
    return "F"


def score_document(
    doi: str,
    max_references: int = 60,
    session: Optional[requests.Session] = None,
) -> DocumentTrustReport:
    """Score a document's trustworthiness including the literature it cites.

    Args:
        doi: the document's DOI
        max_references: cap on how many references to retraction-check (rate limit)
        session: optional shared requests.Session
    """
    sess = session or requests.Session()
    sess.headers.update({"User-Agent": _UA})

    work = _fetch_crossref(doi, sess) or {}
    title = (work.get("title") or [""])[0] if work.get("title") else ""
    year = _year_from_crossref(work)
    venue = (work.get("container-title") or [""])[0] if work.get("container-title") else ""

    red_flags: List[str] = []
    notes: List[str] = []

    # ---- A. SELF trust -----------------------------------------------------
    self_ret = check_retraction(doi, session=sess)
    self_retracted = self_ret.retracted
    self_temporal = temporal_score(year)
    self_retraction_score = 0.0 if self_retracted else (0.6 if self_ret.corrected else 1.0)
    if self_retracted:
        red_flags.append(f"This document is RETRACTED ({self_ret.retraction_type or 'retraction'}).")
    elif self_ret.corrected:
        notes.append(f"This document has a correction/erratum ({self_ret.retraction_type}).")
    if self_ret.source == "doi_not_found":
        red_flags.append("DOI does not resolve in Crossref — cannot verify provenance.")

    # ---- B. FOUNDATION trust (cited literature) ----------------------------
    references = work.get("reference", []) or []
    n_references = len(references)
    ref_dois = [r.get("DOI", "").strip() for r in references if r.get("DOI")]
    ref_dois = ref_dois[:max_references]
    n_with_doi = len(ref_dois)

    # reference years (from unstructured Crossref ref entries where available)
    ref_years: List[int] = []
    for r in references:
        y = r.get("year")
        if y and str(y)[:4].isdigit():
            ref_years.append(int(str(y)[:4]))

    retracted_refs: List[ReferenceTrust] = []
    if ref_dois:
        ret_results = check_retraction_batch(ref_dois)
        ref_meta = {r.get("DOI", "").strip().lower(): r for r in references if r.get("DOI")}
        for rr in ret_results:
            if rr.retracted:
                meta = ref_meta.get(rr.doi.lower(), {})
                y = meta.get("year")
                y = int(str(y)[:4]) if y and str(y)[:4].isdigit() else None
                retracted_refs.append(ReferenceTrust(
                    doi=rr.doi, year=y, retracted=True,
                    retraction_type=rr.retraction_type,
                    title=(meta.get("article-title") or meta.get("journal-title") or "")[:120],
                ))
    n_retracted_refs = len(retracted_refs)

    # foundation retraction score: 1.0 clean, penalise steeply per retracted ref
    if n_with_doi == 0:
        foundation_retraction_score = 0.5  # can't verify — neutral-low
        notes.append("No references carry DOIs — citation base could not be verified.")
    else:
        frac_retracted = n_retracted_refs / n_with_doi
        foundation_retraction_score = max(0.0, 1.0 - 6.0 * frac_retracted)
    if n_retracted_refs > 0:
        red_flags.append(
            f"Cites {n_retracted_refs} retracted work(s) out of {n_with_doi} DOI-linked references."
        )

    # foundation temporal: median age of cited works + recency
    median_age: Optional[float] = None
    pct_recent: Optional[float] = None
    if ref_years:
        ages = [max(0, CURRENT_YEAR - y) for y in ref_years]
        median_age = float(statistics.median(ages))
        pct_recent = round(100 * sum(1 for a in ages if a <= 10) / len(ages), 1)
        foundation_temporal_score = temporal_score(CURRENT_YEAR - int(median_age))
        if median_age > 25:
            notes.append(f"Cited evidence base is old (median reference age {median_age:.0f} yr).")
    else:
        foundation_temporal_score = 0.5
        notes.append("Reference years unavailable — cited-evidence recency not assessed.")

    # ---- Composite ---------------------------------------------------------
    w = DOC_WEIGHTS
    overall = (
        w["self_retraction"]       * self_retraction_score +
        w["self_temporal"]         * self_temporal +
        w["foundation_retraction"] * foundation_retraction_score +
        w["foundation_temporal"]   * foundation_temporal_score
    )
    # A retracted document is untrustworthy regardless of its citations — hard cap.
    if self_retracted:
        overall = min(overall, 0.15)
    overall = round(min(1.0, max(0.0, overall)), 3)

    return DocumentTrustReport(
        doi=doi, title=title, year=year, venue=venue,
        self_retracted=self_retracted,
        self_retraction_type=self_ret.retraction_type,
        self_temporal_score=self_temporal,
        n_references=n_references,
        n_references_with_doi=n_with_doi,
        n_retracted_references=n_retracted_refs,
        retracted_references=retracted_refs,
        median_reference_age=median_age,
        pct_references_last_10yr=pct_recent,
        foundation_retraction_score=foundation_retraction_score,
        foundation_temporal_score=foundation_temporal_score,
        overall_trust=overall,
        grade=_grade(overall),
        red_flags=red_flags,
        notes=notes,
    )


def build_dossier(doi: str, max_references: int = 60) -> dict:
    """Assemble a structured EVIDENCE DOSSIER for LLM-based trust judgment.

    Unlike ``score_document`` (which returns a number), this returns the raw evidence
    an LLM needs to *reason*: the document's own retraction status and reason, its
    abstract, venue, and — for each retracted work it cites — the actual passage where
    it is cited, so the LLM can distinguish reliance from criticism.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": _UA})

    report = score_document(doi, max_references=max_references, session=session)
    work = _fetch_crossref(doi, session) or {}
    abstract = _abstract_from_crossref(work)
    openalex = _fetch_openalex(doi, session)
    notice_title = _retraction_notice_title(work, session) if report.self_retracted else ""

    # Citation context for retracted references (only fetches OA full text once).
    # Enrich each retracted ref with its authoritative title + first author from
    # Crossref so the OA full-text matcher has something to match on.
    retracted_ref_dossier = []
    if report.retracted_references:
        enriched = {}
        for r in report.retracted_references:
            meta = _fetch_crossref(r.doi, session) or {}
            title = (meta.get("title") or [r.title or ""])[0]
            authors = meta.get("author") or []
            surname = authors[0].get("family", "") if authors else ""
            yr = _year_from_crossref(meta) or r.year
            enriched[r.doi] = {"title": title, "surname": surname, "year": yr}

        targets = [
            {"doi": r.doi, "title": enriched[r.doi]["title"],
             "author_year": (enriched[r.doi]["surname"], enriched[r.doi]["year"])}
            for r in report.retracted_references
        ]
        contexts = get_citation_contexts(doi, targets, session=session)
        for r in report.retracted_references:
            ctx = contexts.get(r.doi)
            e = enriched[r.doi]
            retracted_ref_dossier.append({
                "doi": r.doi,
                "title": e["title"],
                "year": e["year"],
                "retraction_type": r.retraction_type,
                "citation_passages": ctx.passages if ctx else [],
                "citation_context_source": ctx.source if ctx else "unavailable",
            })

    # Deterministic signals — surfaced for the LLM, not as a verdict
    signals = []
    if report.self_retracted:
        signals.append("document_is_retracted")
    if report.n_retracted_references > 0:
        signals.append(f"cites_{report.n_retracted_references}_retracted_works")
    if report.median_reference_age and report.median_reference_age > 25:
        signals.append("old_citation_base")
    if report.n_references_with_doi == 0:
        signals.append("no_verifiable_references")

    return {
        "document": {
            "doi": doi,
            "title": report.title,
            "abstract": abstract,
            "year": report.year,
            "venue": report.venue or openalex.get("venue", ""),
            "type": openalex.get("type", ""),
            "cited_by_count": openalex.get("cited_by_count"),
            "in_doaj": openalex.get("is_in_doaj"),
            "self_retracted": report.self_retracted,
            "retraction_type": report.self_retraction_type,
            "retraction_notice_title": notice_title,
            "openalex_is_retracted": openalex.get("is_retracted"),
        },
        "foundation": {
            "n_references": report.n_references,
            "n_references_with_doi": report.n_references_with_doi,
            "n_retracted_references": report.n_retracted_references,
            "retracted_references": retracted_ref_dossier,
            "median_reference_age": report.median_reference_age,
            "pct_references_last_10yr": report.pct_references_last_10yr,
        },
        "deterministic_scores": {
            "overall_trust": report.overall_trust,
            "grade": report.grade,
            "self_temporal_score": round(report.self_temporal_score, 3),
            "foundation_retraction_score": round(report.foundation_retraction_score, 3),
            "foundation_temporal_score": round(report.foundation_temporal_score, 3),
        },
        "signals": signals,
        "red_flags": report.red_flags,
        "notes": report.notes,
    }


if __name__ == "__main__":
    import sys
    dois = sys.argv[1:] or [
        "10.1016/S0140-6736(97)11096-0",  # Wakefield — retracted
        "10.1038/nature12373",            # clean high-profile Nature paper
    ]
    for d in dois:
        rep = score_document(d)
        print(f"\n{'='*70}\n{rep.doi}  [{rep.grade}]  trust={rep.overall_trust}")
        print(f"  {rep.title[:80]}")
        print(f"  {rep.venue} ({rep.year})")
        print(f"  self retracted: {rep.self_retracted}")
        print(f"  references: {rep.n_references_with_doi} DOI-linked, "
              f"{rep.n_retracted_references} retracted, median age {rep.median_reference_age}")
        for flag in rep.red_flags:
            print(f"  🚩 {flag}")
        for note in rep.notes:
            print(f"  • {note}")
