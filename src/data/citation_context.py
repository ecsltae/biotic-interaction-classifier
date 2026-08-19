"""
citation_context.py — Extract the in-text passage(s) where a paper cites a
specific reference, using Europe PMC open-access full text.

This is the "precious" evidence for LLM-based trust judgment: knowing that a paper
cites a retracted work is not enough — the LLM needs to see HOW it is cited. A
citation can be damning ("we build on the finding of X") or benign ("contrary to
the retracted claim of X, we show ..."). Only the surrounding sentence reveals intent.

Mechanism (open-access papers only):
  1. Resolve citing DOI → PMCID via Europe PMC search
  2. Fetch JATS full-text XML
  3. In the <ref-list>, find the <ref> whose DOI / title matches the target
     reference, note its id (rid)
  4. Find every <xref ref-type="bibr" rid="..."> in the body, extract the
     enclosing sentence/paragraph text

Gracefully returns [] when the citing paper is not open-access or has no matchable
reference — the caller then falls back to reference metadata alone.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from xml.etree import ElementTree as ET

import requests

logger = logging.getLogger(__name__)

_UA = "MetaP-TrustScorer/1.0 (mailto:esteban.gaillac1@gmail.com)"
_EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_EPMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"


@dataclass
class CitationContext:
    target_doi: str
    passages: List[str]        # sentences/paragraphs where the reference is cited
    ref_label: str             # e.g. "12" or "Mikovits et al."
    source: str                # "europepmc_oa" | "unavailable"


def _resolve_pmcid(doi: str, session: requests.Session) -> Optional[str]:
    try:
        r = session.get(
            _EPMC_SEARCH,
            params={"query": f"DOI:{doi}", "resultType": "core", "format": "json"},
            headers={"User-Agent": _UA}, timeout=15,
        )
        if not r.ok:
            return None
        results = r.json().get("resultList", {}).get("result", [])
        if not results:
            return None
        rec = results[0]
        if rec.get("isOpenAccess") == "Y" and rec.get("pmcid"):
            return rec["pmcid"]
    except Exception as e:
        logger.debug("EPMC resolve failed for %s: %s", doi, e)
    return None


def _fetch_fulltext(pmcid: str, session: requests.Session) -> Optional[str]:
    try:
        r = session.get(
            _EPMC_FULLTEXT.format(pmcid=pmcid),
            headers={"User-Agent": _UA}, timeout=25,
        )
        if r.ok and r.text and "<article" in r.text:
            return r.text
    except Exception as e:
        logger.debug("EPMC fulltext failed for %s: %s", pmcid, e)
    return None


def _norm_doi(doi: str) -> str:
    return (doi or "").strip().lower().rstrip(".")


def _text_of(elem) -> str:
    """Flatten an element's text content, dropping tags."""
    return re.sub(r"\s+", " ", "".join(elem.itertext())).strip()


def _ref_label(ref, rid: str) -> str:
    lbl_el = ref.find("label")
    if lbl_el is not None and lbl_el.text:
        return lbl_el.text.strip()
    return rid


def _find_ref_id(
    root, target_doi: str, target_title: str = "",
    author_year: Optional[tuple] = None,
) -> Optional[tuple]:
    """Locate the <ref> matching the target → return (rid, label).

    Match priority: JATS <pub-id type=doi> element → DOI anywhere in text →
    title substring → (first-author surname + year).
    """
    tdoi = _norm_doi(target_doi)
    ttitle = (target_title or "").lower()[:60]
    for ref in root.iter("ref"):
        rid = ref.get("id")
        if not rid:
            continue
        # 1. Structured DOI element (most reliable)
        for pid in ref.iter("pub-id"):
            if pid.get("pub-id-type") == "doi" and _norm_doi(pid.text or "") == tdoi and tdoi:
                return rid, _ref_label(ref, rid)
        ref_text = _text_of(ref).lower()
        # 2. DOI anywhere in flattened text
        if tdoi and tdoi in ref_text.replace(" ", ""):
            return rid, _ref_label(ref, rid)
        # 3. Title substring
        if ttitle and len(ttitle) > 20 and ttitle in ref_text:
            return rid, _ref_label(ref, rid)
        # 4. First-author surname + year (last resort)
        if author_year:
            surname, year = author_year
            if surname and year and surname.lower() in ref_text and str(year) in ref_text:
                return rid, _ref_label(ref, rid)
    return None


_MARKER = "\x00CITE\x00"          # sentinel inserted where the target citation sits
_BLOCK_TAGS = {"p", "sec", "td", "caption", "list-item", "abstract", "statement"}


def _flatten_with_marker(elem, target_xref) -> str:
    """Flatten element text in document order, inserting _MARKER at target_xref."""
    parts: List[str] = []

    def walk(e):
        if e is target_xref:
            parts.append(_MARKER)
            if e.tail:
                parts.append(e.tail)
            return
        if e.text:
            parts.append(e.text)
        for child in list(e):
            walk(child)
        if e.tail:
            parts.append(e.tail)

    walk(elem)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _extract_passages(root, rid: str, max_passages: int = 4) -> List[str]:
    """Find the sentence around every <xref rid=rid ref-type=bibr> in the body."""
    parent_map = {c: p for p in root.iter() for c in p}

    def enclosing_block(node):
        cur = parent_map.get(node)
        while cur is not None:
            tag = cur.tag.split("}")[-1]  # strip namespace
            if tag in _BLOCK_TAGS:
                return cur
            nxt = parent_map.get(cur)
            if nxt is None:
                return cur  # top of tree — return what we have
            cur = nxt
        return None

    passages: List[str] = []
    for xref in root.iter("xref"):
        if xref.get("ref-type") != "bibr" or xref.get("rid") != rid:
            continue
        block = enclosing_block(xref)
        if block is None:
            continue
        marked = _flatten_with_marker(block, xref)
        if _MARKER not in marked:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", marked)
        chosen = next((s for s in sentences if _MARKER in s), marked)
        snippet = chosen.replace(_MARKER, "").strip()
        # collapse leftover double-spaces from marker removal
        snippet = re.sub(r"\s+([.,;)])", r"\1", re.sub(r"\s{2,}", " ", snippet))[:500]
        if snippet and snippet not in passages:
            passages.append(snippet)
        if len(passages) >= max_passages:
            break
    return passages


def get_citation_context(
    citing_doi: str,
    target_doi: str,
    target_title: str = "",
    session: Optional[requests.Session] = None,
) -> CitationContext:
    """Extract passages in ``citing_doi`` that cite ``target_doi`` (OA only)."""
    sess = session or requests.Session()
    sess.headers.update({"User-Agent": _UA})

    pmcid = _resolve_pmcid(citing_doi, sess)
    if not pmcid:
        return CitationContext(target_doi=target_doi, passages=[], ref_label="",
                               source="unavailable")

    xml = _fetch_fulltext(pmcid, sess)
    if not xml:
        return CitationContext(target_doi=target_doi, passages=[], ref_label="",
                               source="unavailable")

    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return CitationContext(target_doi=target_doi, passages=[], ref_label="",
                               source="unavailable")

    found = _find_ref_id(root, target_doi, target_title)
    if not found:
        return CitationContext(target_doi=target_doi, passages=[], ref_label="",
                               source="unavailable")
    rid, label = found
    passages = _extract_passages(root, rid)
    return CitationContext(
        target_doi=target_doi,
        passages=passages,
        ref_label=label,
        source="europepmc_oa" if passages else "unavailable",
    )


def get_citation_contexts(
    citing_doi: str,
    targets: List[dict],
    session: Optional[requests.Session] = None,
) -> Dict[str, CitationContext]:
    """Batch: targets = [{"doi":..., "title":...}]. Fetches full text once."""
    sess = session or requests.Session()
    sess.headers.update({"User-Agent": _UA})

    pmcid = _resolve_pmcid(citing_doi, sess)
    out: Dict[str, CitationContext] = {}
    if not pmcid:
        for t in targets:
            out[t["doi"]] = CitationContext(target_doi=t["doi"], passages=[],
                                            ref_label="", source="unavailable")
        return out

    xml = _fetch_fulltext(pmcid, sess)
    if not xml:
        for t in targets:
            out[t["doi"]] = CitationContext(target_doi=t["doi"], passages=[],
                                            ref_label="", source="unavailable")
        return out

    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        for t in targets:
            out[t["doi"]] = CitationContext(target_doi=t["doi"], passages=[],
                                            ref_label="", source="unavailable")
        return out

    for t in targets:
        found = _find_ref_id(root, t["doi"], t.get("title", ""), t.get("author_year"))
        if not found:
            out[t["doi"]] = CitationContext(target_doi=t["doi"], passages=[],
                                            ref_label="", source="unavailable")
            continue
        rid, label = found
        passages = _extract_passages(root, rid)
        out[t["doi"]] = CitationContext(
            target_doi=t["doi"], passages=passages, ref_label=label,
            source="europepmc_oa" if passages else "unavailable",
        )
    return out


if __name__ == "__main__":
    import sys
    citing = sys.argv[1] if len(sys.argv) > 1 else "10.1371/journal.pone.0035150"
    target = sys.argv[2] if len(sys.argv) > 2 else ""
    ctx = get_citation_context(citing, target)
    print(f"citing:  {citing}")
    print(f"target:  {target}")
    print(f"source:  {ctx.source}   label: {ctx.ref_label}")
    for i, p in enumerate(ctx.passages, 1):
        print(f"  [{i}] {p}")
