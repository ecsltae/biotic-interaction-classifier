"""
biotxplorer_trust.py — Trust-annotate BiotXplorer interaction triplets.

Chains the public BiotXplorer API to the local trust service (port 8003):

  1. /api/search    -> interaction triplets for a species pair, ranked by text-mining score
  2. /api/passages  -> the supporting documents behind each triplet
                       (each carries doi, pubyear, publication_types + the matched passage)
  3. trust service  -> per-document credibility: retraction, recency, evidence type
  4. aggregate      -> a trust-weighted score per triplet, and a re-ranking

The point: BiotXplorer's `score` measures how strongly an interaction is *attested*
in text. It says nothing about whether the attesting documents are credible. A triplet
supported by 70 recent primary studies and one supported by a single retracted paper
are indistinguishable in the current ranking.

Usage:
    python biotxplorer_trust.py --taxon1 "Triticum aestivum" --taxon2 "Hordeum vulgare"
    python biotxplorer_trust.py --taxon1 ott:31926 --taxon2 ott:657948 --max-docs 40
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

BIOTXPLORER = "https://biotxplorer.sibils.org/api"
TRUST_SERVICE = "http://localhost:8003"


# ---------------------------------------------------------------------------
# BiotXplorer client
# ---------------------------------------------------------------------------

def normalize_taxon(term: str, timeout: int = 20) -> Optional[str]:
    """Resolve a species name to an OTT concept id via BiotXplorer."""
    if term.startswith("ott:"):
        return term
    r = requests.get(f"{BIOTXPLORER}/normalize/taxon", params={"term": term}, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    candidates = payload if isinstance(payload, list) else payload.get("results", [])
    for c in candidates:
        cid = c.get("concept_id") if isinstance(c, dict) else None
        if cid:
            return cid
    return None


def search_triplets(
    taxon1: str = "",
    taxon2: str = "",
    interaction: str = "",
    collection: str = "medline",
    timeout: int = 30,
) -> List[dict]:
    """Interaction triplets, as BiotXplorer ranks them.

    Any of taxon1 / taxon2 / interaction may be omitted to leave that slot open —
    one taxon alone returns its whole neighbourhood. Empty parameters must be
    dropped rather than sent blank: ``taxon2=""`` is treated as a filter that
    matches nothing, silently returning zero hits.
    """
    params = {"collection": collection}
    if taxon1:
        params["taxon1"] = taxon1
    if taxon2:
        params["taxon2"] = taxon2
    if interaction:
        params["interaction"] = interaction
    r = requests.get(f"{BIOTXPLORER}/search", params=params, timeout=timeout)
    r.raise_for_status()
    hits = r.json()
    return hits if isinstance(hits, list) else []


def fetch_supporting_docs(
    triplet_key: str,
    collection: str = "medline",
    timeout: int = 60,
) -> List[dict]:
    """The documents behind one triplet, with metadata and matched passages."""
    r = requests.get(
        f"{BIOTXPLORER}/passages",
        params={"triplet_key": triplet_key, "collection": collection},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("documents", [])


# ---------------------------------------------------------------------------
# Trust annotation
# ---------------------------------------------------------------------------

@dataclass
class TripletTrust:
    interaction: str
    triplet_key: str
    bx_score: float
    docs_count: int
    n_scored: int = 0
    n_retracted: int = 0
    retracted_dois: List[str] = field(default_factory=list)
    mean_trust: float = 0.0
    min_trust: float = 0.0
    median_year: Optional[int] = None
    evidence_types: Dict[str, int] = field(default_factory=dict)
    trust_weighted_score: float = 0.0


def _passage_text(doc: dict) -> str:
    """The matched passage, which is what the evidence axis should classify."""
    parts = [p.get("text", "") for p in doc.get("passages", []) if p.get("text")]
    if parts:
        return " ".join(parts)
    return (doc.get("document", {}) or {}).get("title", "")


def score_triplet(
    hit: dict,
    species1: str,
    species2: str,
    collection: str,
    max_docs: int,
    trust_url: str,
) -> TripletTrust:
    """Fetch a triplet's supporting documents and score each one for trust."""
    itype = (hit.get("interaction") or {}).get("preferred_term", "")
    tt = TripletTrust(
        interaction=itype,
        triplet_key=hit.get("triplet_key", ""),
        bx_score=float(hit.get("score", 0.0)),
        docs_count=int(hit.get("docs_count", 0)),
    )

    docs = fetch_supporting_docs(tt.triplet_key, collection=collection)[:max_docs]
    if not docs:
        return tt

    items, years = [], []
    for d in docs:
        meta = d.get("document", {}) or {}
        doi = (meta.get("doi") or "").strip()
        year = meta.get("pubyear")
        year = int(year) if isinstance(year, str) and year.isdigit() else None
        if year:
            years.append(year)
        items.append({
            "text": _passage_text(d),
            "species1": species1,
            "species2": species2,
            "interaction_type": itype,
            "pub_year": year,
            "doi": doi or None,
            # MEDLINE's curated labels — a far better evidence signal than the
            # text patterns, and already present in BiotXplorer's own payload.
            "publication_types": meta.get("publication_types") or None,
        })

    r = requests.post(f"{trust_url}/score_batch", json={"items": items}, timeout=300)
    r.raise_for_status()
    results = r.json().get("results", [])

    trusts = [x["composite_trust"] for x in results]
    for x, it in zip(results, items):
        if x.get("retracted"):
            tt.n_retracted += 1
            if it["doi"]:
                tt.retracted_dois.append(it["doi"])
        et = x.get("evidence_type", "unknown")
        tt.evidence_types[et] = tt.evidence_types.get(et, 0) + 1

    tt.n_scored = sum(1 for it in items if it["doi"])
    tt.mean_trust = round(sum(trusts) / len(trusts), 3) if trusts else 0.0
    tt.min_trust = round(min(trusts), 3) if trusts else 0.0
    tt.median_year = sorted(years)[len(years) // 2] if years else None
    # Attestation strength, discounted by the credibility of what attests it.
    tt.trust_weighted_score = round(tt.bx_score * tt.mean_trust, 3)
    return tt


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(sp1: str, sp2: str, triplets: List[TripletTrust]) -> None:
    print()
    print("=" * 96)
    print(f"  {sp1}  x  {sp2}")
    print("=" * 96)

    by_bx = sorted(triplets, key=lambda t: -t.bx_score)
    by_trust = sorted(triplets, key=lambda t: -t.trust_weighted_score)
    rank_of = {t.triplet_key: i for i, t in enumerate(by_trust)}

    print(f"\n{'interaction':<26} {'BX score':>9} {'docs':>5} {'mean':>6} "
          f"{'min':>6} {'retr':>5} {'yr':>5} {'weighted':>9} {'Δrank':>6}")
    print("-" * 96)
    for i, t in enumerate(by_bx):
        delta = i - rank_of[t.triplet_key]
        arrow = f"{delta:+d}" if delta else "0"
        flag = f" {t.n_retracted}!" if t.n_retracted else f" {t.n_retracted}"
        print(f"{t.interaction[:26]:<26} {t.bx_score:>9.2f} {t.docs_count:>5} "
              f"{t.mean_trust:>6.3f} {t.min_trust:>6.3f} {flag:>5} "
              f"{str(t.median_year or '-'):>5} {t.trust_weighted_score:>9.2f} {arrow:>6}")

    retracted = [t for t in triplets if t.n_retracted]
    if retracted:
        print("\n  RETRACTED SOURCES DETECTED")
        for t in retracted:
            print(f"    '{t.interaction}' — {t.n_retracted}/{t.docs_count} supporting "
                  f"document(s) retracted:")
            for doi in t.retracted_dois:
                print(f"        {doi}")
    else:
        print("\n  No retracted sources among the supporting documents.")

    ev: Dict[str, int] = {}
    for t in triplets:
        for k, v in t.evidence_types.items():
            ev[k] = ev.get(k, 0) + v
    if ev:
        total = sum(ev.values())
        summary = ", ".join(f"{k} {v} ({100*v//total}%)"
                            for k, v in sorted(ev.items(), key=lambda x: -x[1]))
        print(f"\n  Evidence types across all supporting passages: {summary}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taxon1", required=True, help="species name or ott:ID")
    ap.add_argument("--taxon2", required=True, help="species name or ott:ID")
    ap.add_argument("--collection", default="medline")
    ap.add_argument("--top-triplets", type=int, default=8,
                    help="how many triplets to trust-annotate")
    ap.add_argument("--max-docs", type=int, default=40,
                    help="max supporting documents to score per triplet")
    ap.add_argument("--trust-url", default=TRUST_SERVICE)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    try:
        requests.get(f"{args.trust_url}/health", timeout=5).raise_for_status()
    except Exception as e:
        sys.exit(f"Trust service unavailable at {args.trust_url}: {e}\n"
                 f"Start it with: bash classifier/start_trust_service.sh")

    ott1 = normalize_taxon(args.taxon1)
    ott2 = normalize_taxon(args.taxon2)
    if not ott1 or not ott2:
        sys.exit(f"Could not resolve taxa: {args.taxon1!r} -> {ott1}, {args.taxon2!r} -> {ott2}")
    print(f"Resolved: {args.taxon1} -> {ott1} | {args.taxon2} -> {ott2}", flush=True)

    hits = search_triplets(ott1, ott2, collection=args.collection)
    if not hits:
        sys.exit("No interaction triplets found for this pair.")
    hits = sorted(hits, key=lambda h: -float(h.get("score", 0)))[:args.top_triplets]
    print(f"Trust-annotating {len(hits)} triplet(s) ...", flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        triplets = list(pool.map(
            lambda h: score_triplet(h, args.taxon1, args.taxon2,
                                    args.collection, args.max_docs, args.trust_url),
            hits,
        ))

    print_report(args.taxon1, args.taxon2, triplets)

    if args.json_out:
        args.json_out.write_text(
            json.dumps([t.__dict__ for t in triplets], indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
