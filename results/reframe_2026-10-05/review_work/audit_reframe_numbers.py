"""Lookup helper for auditing numbers in paperA_reframe.tex.

Loads every non-backup JSON under results/paperA_v2/sentence_level/ (and llm/),
flattens it, and for each queried printed value lists the keys whose value rounds to it.
Usage: python audit_reframe_numbers.py 0.965 0.930 ... [--grep substr]
"""
import argparse
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SL = ROOT / "results/paperA_v2/sentence_level"


def flatten(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flatten(v, f"{p}.{k}" if p else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from flatten(v, f"{p}[{i}]")
    else:
        yield p, o


def load() -> list[tuple[str, str, float]]:
    out = []
    for f in sorted(SL.rglob("*.json")):
        if ".bak" in f.name or f.name == "sentence_tables.json":
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for k, v in flatten(d):
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                out.append((str(f.relative_to(ROOT)), k, float(v)))
    return out


def rounds_to(v: float, printed: str) -> bool:
    s = printed.lstrip("+")
    neg = s.startswith("-")
    dec = len(s.split(".")[1]) if "." in s else 0
    q = Decimal(1).scaleb(-dec)
    r = Decimal(repr(v)).quantize(q, rounding=ROUND_HALF_UP)
    return r == Decimal(s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("vals", nargs="+")
    ap.add_argument("--grep", default="")
    ap.add_argument("--pct", action="store_true", help="also match value*100")
    a = ap.parse_args()
    data = load()
    for p in a.vals:
        hits = [(f, k, v) for f, k, v in data if a.grep in f + ":" + k and
                (rounds_to(v, p) or (a.pct and rounds_to(v * 100, p)))]
        print(f"== {p}: {len(hits)} hits")
        for f, k, v in hits[:25]:
            print(f"   {f}:{k} = {v}")
