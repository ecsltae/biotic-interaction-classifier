"""
Improved taxon-name -> OTT-id resolver for the ampliseq reranking join key.

Motivation
----------
`ampliseq_rerank.py` resolves each detected taxon name to an OTT id via the
online SIBiLS autocomplete API, one name at a time. On the fungal ITS2 dataset
that layer silently dropped 184/911 names (20%); every miss is a co-detected
pair the coherence reranker never sees. Two independent failures were found:

  Fix A (offline, no new dependency): some dropped names are *present in the
         local OTT dump* (ott_v3.7.2.json) but SIBiLS autocomplete's fuzzy/
         prefix matching missed them. Resolving directly against the local
         dump rescues these, with no rate limit.

  Fix B (richer backbone): other dropped names are absent from OTT entirely
         but present in Catalogue of Life / GBIF. Global Names (gnverifier)
         matches them and, queried with data source = Open Tree of Life,
         returns the OTT id -- so BiotXplorer's `source=ott` join key is kept.

`resolve()` tries Fix A first (fast, offline), then Fix B for the remainder.
"""
from __future__ import annotations
import json
import os
import pickle
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
OTT_DUMP = ROOT / "classifier" / "data" / "taxonomies" / "ott_v3.7.2.json"
INDEX_CACHE = Path(__file__).parent / "ott_local_index.pkl"

GNVERIFIER = "https://verifier.globalnames.org/api/v1/verifications"
OTT_SOURCE_ID = 179  # Open Tree of Life data-source id in Global Names


# ---------------------------------------------------------------------------
# Fix A: offline local OTT dump
# ---------------------------------------------------------------------------
def build_local_index(dump: Path = OTT_DUMP, cache: Path = INDEX_CACHE) -> Dict[str, tuple]:
    """Build {name_lower -> (ott_id, rank)} from the local OTT dump. Cached."""
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    print(f"  building local OTT index from {dump.name} (one-off) ...", flush=True)
    with open(dump) as fh:
        data = json.load(fh)
    index: Dict[str, tuple] = {}
    for c in data["concepts"]:
        cid = c.get("id")
        if not cid:
            continue
        rank = (c.get("rank") or {}).get("name", "")
        terms = []
        pt = c.get("preferred_term") or {}
        if pt.get("term"):
            terms.append(pt["term"])
        for syn in c.get("synonyms") or []:
            if syn.get("term"):
                terms.append(syn["term"])
        for t in terms:
            key = t.strip().lower()
            # prefer the first (preferred_term) binding; don't let synonyms clobber
            if key and key not in index:
                index[key] = (cid, rank)
    with open(cache, "wb") as fh:
        pickle.dump(index, fh)
    print(f"  local OTT index: {len(index):,} name keys", flush=True)
    return index


def resolve_local(names: List[str], index: Dict[str, tuple]) -> Dict[str, Optional[str]]:
    """Fix A. name -> 'ott:ID' from the local dump, else None."""
    out: Dict[str, Optional[str]] = {}
    for n in names:
        hit = index.get(n.strip().lower())
        out[n] = f"ott:{hit[0]}" if hit else None
    return out


# ---------------------------------------------------------------------------
# Fix B: gnverifier, restricted to the OTT data source
# ---------------------------------------------------------------------------
def resolve_gnverifier_ott(names: List[str]) -> Dict[str, Optional[str]]:
    """Fix B. Batch name -> 'ott:ID' via Global Names, OTT source only.

    withCapitalization=True is required: the ampliseq caches store lowercased
    names, and gnverifier returns NoMatch for un-capitalized binomials.
    """
    if not names:
        return {}
    body = json.dumps(
        {
            "nameStrings": names,
            "withCapitalization": True,
            "dataSources": [OTT_SOURCE_ID],
            "withAllMatches": False,
        }
    ).encode()
    req = urllib.request.Request(
        GNVERIFIER, data=body, headers={"Content-Type": "application/json"}
    )
    resp = json.load(urllib.request.urlopen(req, timeout=120))
    out: Dict[str, Optional[str]] = {}
    # Only accept full-name matches. Partial* means gnverifier dropped the
    # species epithet and matched the genus -> wrong rank for our purposes.
    accept = {"Exact", "Fuzzy"}
    for rec in resp.get("names", []):
        name = rec.get("name")
        best = rec.get("bestResult") or {}
        if rec.get("matchType") not in accept or not best:
            out[name] = None
            continue
        # recordId is e.g. "986732" or "sf-<uuid>|986732|sf-<uuid>";
        # the OTT taxon id is the numeric token (verified to match the dump).
        rid = None
        for tok in str(best.get("recordId", "")).split("|"):
            if tok.isdigit():
                rid = tok
                break
        out[name] = f"ott:{rid}" if rid else None
    return out


# ---------------------------------------------------------------------------
# Combined resolver
# ---------------------------------------------------------------------------
def resolve(names: List[str], index: Optional[Dict[str, tuple]] = None) -> Dict[str, Optional[str]]:
    """Fix A (local) first, Fix B (gnverifier/OTT) for the remainder."""
    if index is None:
        index = build_local_index()
    resolved = resolve_local(names, index)
    missing = [n for n, v in resolved.items() if v is None]
    if missing:
        for n, v in resolve_gnverifier_ott(missing).items():
            if v:
                resolved[n] = v
    return resolved


if __name__ == "__main__":
    import sys

    scratch = "/tmp/claude-1001/-home-egaillac-MetaP/7195b8b3-359d-4aec-9f83-195ecca6274f/scratchpad/unresolved_names.json"
    names = json.load(open(sys.argv[1] if len(sys.argv) > 1 else scratch))
    print(f"Resolving {len(names)} names SIBiLS autocomplete dropped\n")

    idx = build_local_index()

    t0 = time.time()
    a = resolve_local(names, idx)
    a_ok = [n for n, v in a.items() if v]
    print(f"Fix A (local OTT dump, offline): {len(a_ok)}/{len(names)} "
          f"resolved in {time.time()-t0:.1f}s")

    missing = [n for n, v in a.items() if not v]
    t0 = time.time()
    b = resolve_gnverifier_ott(missing)
    b_ok = [n for n, v in b.items() if v]
    print(f"Fix B (gnverifier -> OTT source), on the {len(missing)} A missed: "
          f"{len(b_ok)}/{len(missing)} resolved in {time.time()-t0:.1f}s")

    combined = {**a, **{n: v for n, v in b.items() if v}}
    c_ok = [n for n, v in combined.items() if v]
    print(f"\nCombined: {len(c_ok)}/{len(names)} "
          f"({100*len(c_ok)/len(names):.0f}%) now resolve to an OTT id")
    still = [n for n, v in combined.items() if not v]
    print(f"Still unresolved ({len(still)}): {still[:12]}")
