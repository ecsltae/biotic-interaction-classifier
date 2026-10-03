#!/usr/bin/env python3
"""Check every number in Paper A against the result files.

Collects every numeric value in the result JSONs (recursively; also x100 for percentages and
EP F1, and differences between same-level values of two arms for "gap" numbers), extracts every
number from paperA.tex with its line, and reports the numbers that no result value explains at
the paper's printed precision. Unexplained numbers are not necessarily wrong (dataset sizes,
hyperparameters, years, section counts); they are the list to check by hand.

Usage
  python3 scripts/audit_paper_numbers.py [--all]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCES = [*sorted((REPO / "results/paperA_v2").glob("*.json")),
           REPO / "results/shipping_2026-10-02/eval_a05.json",
           REPO / "results/shipping_2026-10-02/eval_v3.json",
           *sorted((REPO / "results/paperA_rebuild_2026-09-28").glob("*.json"))]


def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(o, (int, float)) and not isinstance(o, bool) and math.isfinite(o):
        yield path, float(o)


def collect():
    vals = []
    for f in SOURCES:
        if not f.exists() or "backup" in str(f):
            continue
        try:
            o = json.loads(f.read_text())
        except Exception:
            continue
        flat = list(walk(o, f.stem))
        vals += flat
        vals += [(p + "*100", v * 100) for p, v in flat if abs(v) <= 1.0]
        # differences between sibling metrics of two arms (e.g. pair.auprc_mean - sentence.auprc_mean)
        by_leaf = {}
        for p, v in flat:
            leaf = re.sub(r"^[^.]*\.[^.]*", "", p)       # drop file and arm
            by_leaf.setdefault(leaf, []).append((p, v))
        for leaf, items in by_leaf.items():
            if len(items) > 40:
                continue
            for (p1, v1), (p2, v2) in combinations(items, 2):
                vals.append((f"{p1} - {p2}", v1 - v2))
                vals.append((f"{p2} - {p1}", v2 - v1))
    return vals


NUM = re.compile(r"(?<![\w.\\{])([+-]?\d+(?:\{,\}\d{3})*(?:\.\d+)?)(?:\s*\\times\s*10\^\{(-?\d+)\})?")


def explained(tok, exp, vals):
    s = tok.replace("{,}", "")
    try:
        x = float(s)
    except ValueError:
        return None
    if exp is not None:                                  # a p-value like 1.3\times10^{-7}
        target = x * 10 ** int(exp)
        digits = len(s.split(".")[1]) if "." in s else 0
        return [p for p, v in vals if v > 0 and abs(float(f"{v:.{digits}e}") - target) <= target * 1e-9]
    digits = len(s.split(".")[1]) if "." in s else 0
    tol = 0.5 * 10 ** -digits + 1e-9
    return [p for p, v in vals if abs(v - x) <= tol]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="also list explained numbers with one source each")
    a = ap.parse_args()
    vals = collect()
    tex = (REPO / "paperA/paperA.tex").read_text().split("\n")
    start = next(i for i, l in enumerate(tex) if l.startswith(r"\begin{abstract}"))
    unexplained, n = [], 0
    for i, line in enumerate(tex[start:], start + 1):
        if line.lstrip().startswith("%"):
            continue
        for m in NUM.finditer(line):
            tok, exp = m.group(1), m.group(2)
            core = tok.lstrip("+-").replace("{,}", "")
            if exp is None and "." not in core and int(float(core)) < 10:
                continue                                 # small integers: section numbers, seeds...
            n += 1
            hit = explained(tok, exp, vals)
            if not hit:
                unexplained.append((i, tok + (f"e{exp}" if exp else ""), line.strip()[:110]))
            elif a.all:
                print(f"ok  {i:5d} {tok:>10}  {hit[0][:70]}")
    print(f"\n{n} numbers checked, {len(unexplained)} not explained by a result file:\n")
    for i, tok, ctx in unexplained:
        print(f"{i:5d} {tok:>12}  {ctx}")


if __name__ == "__main__":
    main()
