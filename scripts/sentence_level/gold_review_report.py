#!/usr/bin/env python3
"""The user's answers on the 22-item gold-review sheet against the current gold (proposals only).

Runs only when all 22 items have both a SENTENCE and a PAIR answer; until then it reports the
answer counts and does NOT read the sealed key's labels. When the sheet is complete it writes
results/reframe_2026-10-05/gold_review_22.md:
  * PAIR answers versus the key's `label` (the current gold): agreement (n, %), Cohen's kappa
    (YES/NO answers only; UNSURE is listed, not counted);
  * every disagreement: item, passage excerpt, taxa, the user's answer, current gold, p_triple and
    p_shipped;
  * the UNSURE items;
  * SENTENCE versus PAIR answers (how many sentence-YES and pair-NO).
Nothing is applied: gold labels change only by the user's decision.

Provenance (CONTEXT.md, 2026-10-05): the answers come from the user's adjudication of a model's
pre-filled sheet (not blind), and 11 of the 22 items were selected by model-vs-gold disagreement
(key column `_kind`), so the report breaks agreement down by `_kind`. If results/reframe_2026-10-05/
gold_review_22.md already exists (the parent session wrote one, with provenance notes, at 02:37), this
script writes gold_review_22.script.md next to it instead of replacing it; `--out` overrides.

Usage
  python3 scripts/sentence_level/gold_review_report.py
  python3 scripts/sentence_level/gold_review_report.py --sheet X.xlsx --key K.csv --out R.md   # tests
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.REPO / "results/reframe_2026-10-05/gold_review_22.md"


def sheet_full(path: Path) -> pd.DataFrame:
    """The review sheet with its text columns (item, taxa, relation, passage, answers, note)."""
    import openpyxl
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True)["review"]
    rows = list(ws.iter_rows(values_only=True))
    hdr = [str(h) for h in rows[0]]
    d = pd.DataFrame([r[: len(hdr)] for r in rows[1:]], columns=hdr)
    pick = lambda p: next(c for c in hdr if c.upper().startswith(p))  # noqa: E731
    norm = lambda x: (str(x).strip().upper() or None) if x is not None else None  # noqa: E731
    return pd.DataFrame({"item": pd.to_numeric(d["#"]).astype(int), "taxon1": d[pick("TAXON 1")],
                         "relation": d[pick("RELATION")], "taxon2": d[pick("TAXON 2")], "passage": d[pick("PASSAGE")],
                         "SENTENCE": d[pick("SENTENCE")].map(norm), "PAIR": d[pick("PAIR")].map(norm),
                         "note": d[pick("NOTE")]})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", default=str(C.GOLD_SHEET))
    ap.add_argument("--key", default=str(C.GOLD_KEY))
    ap.add_argument("--out", default=None, help="default gold_review_22.md, or gold_review_22.script.md if that exists")
    a = ap.parse_args()
    s = sheet_full(Path(a.sheet))
    n_s, n_p = int(s.SENTENCE.notna().sum()), int(s.PAIR.notna().sum())
    complete = len(s) == 22 and n_s == 22 and n_p == 22
    print(f"gold-review sheet: {len(s)} items, {n_s} SENTENCE answers, {n_p} PAIR answers")
    if not complete:
        print("incomplete: the key's labels are not read and no report is written")
        return
    out = Path(a.out) if a.out else (OUT if not OUT.exists() else OUT.with_name("gold_review_22.script.md"))
    k = pd.read_csv(a.key)                       # all columns: allowed only now
    d = s.merge(k.rename(columns={"_kind": "kind"}), on="item", how="left", validate="one_to_one")
    assert d.label.notna().all(), "key misses items"
    yn = d.PAIR.isin(["YES", "NO"])
    u = (d.PAIR[yn] == "YES").astype(int).to_numpy()
    g = d.label[yn].astype(int).to_numpy()
    agree = int((u == g).sum())
    kap = C.kappa(u, g)
    dis = d[yn & ((d.PAIR == "YES").astype(int) != d.label.astype(int))]
    uns = d[~yn]
    excerpt = lambda t: (str(t)[:240] + ("…" if len(str(t)) > 240 else "")).replace("|", "/").replace("\n", " ")  # noqa: E731
    L = ["# Gold review: the user's 22 answers against the current gold (proposals, not applied)", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/sentence_level/gold_review_report.py` from "
         f"`{Path(a.sheet).name}` and the gold-review key. Gold labels change only by the user's decision; "
         "nothing below has been applied.", "",
         "Provenance: the answers come from adjudicating a model's pre-filled sheet, so they are not blind; "
         "11 of the 22 items were selected by model-vs-gold disagreement (column `_kind`), so they are not a "
         "random sample.", "",
         "## PAIR answers versus current gold", "",
         f"- Items with a YES/NO PAIR answer: {int(yn.sum())} of 22 (UNSURE: {int((~yn).sum())}, listed below, not counted).",
         f"- Agreement: {agree}/{int(yn.sum())} = {agree / max(int(yn.sum()), 1):.1%}; Cohen's kappa "
         f"{'undefined' if kap is None else f'{kap:.3f}'}.",
         f"- The user says YES where the gold is 0: {int(((u == 1) & (g == 0)).sum())}; NO where the gold is 1: "
         f"{int(((u == 0) & (g == 1)).sum())}.", "",
         "", "Agreement by how the items were selected (`_kind`):", ""]
    for kind, gk in d[yn].groupby("kind"):
        L.append(f"- {kind}: {int(((gk.PAIR == 'YES').astype(int) == gk.label.astype(int)).sum())}/{len(gk)} agree")
    L += ["", "## Disagreements (each one is a proposal to change the gold)", ""]
    if len(dis):
        L += ["| item | taxa | relation | passage (excerpt) | user PAIR | current gold | p_triple | p_shipped | selected as | note |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in dis.itertuples():
            L.append(f"| {r.item} | {r.taxon1} / {r.taxon2} | {r.relation} | {excerpt(r.passage)} | {r.PAIR} | "
                     f"{int(r.label)} | {r.p_triple:.3f} | {r.p_shipped:.3f} | {r.kind} | {'' if r.note is None else excerpt(r.note)} |")
    else:
        L.append("None.")
    L += ["", "## UNSURE PAIR answers (dropped from every evaluation, per the user's rule)", ""]
    if len(uns):
        L += ["| item | taxa | passage (excerpt) | SENTENCE | PAIR | current gold |", "|---|---|---|---|---|---|"]
        for r in uns.itertuples():
            L.append(f"| {r.item} | {r.taxon1} / {r.taxon2} | {excerpt(r.passage)} | {r.SENTENCE} | {r.PAIR} | {int(r.label)} |")
    else:
        L.append("None.")
    both = d.SENTENCE.isin(["YES", "NO"]) & yn
    sy, py = d.SENTENCE[both] == "YES", d.PAIR[both] == "YES"
    L += ["", "## SENTENCE versus PAIR answers", "",
          f"On the {int(both.sum())} items with YES/NO for both questions: sentence-YES and pair-NO "
          f"{int((sy & ~py).sum())}; both YES {int((sy & py).sum())}; both NO {int((~sy & ~py).sum())}; "
          f"sentence-NO and pair-YES {int((~sy & py).sum())} (inconsistent by definition: a pair interaction is "
          "a biotic interaction)."]
    su = d[d.SENTENCE == "UNSURE"]
    L += ["", f"UNSURE SENTENCE answers (dropped from the sentence level): {', '.join(str(i) for i in su.item) or 'none'}."]
    if out.exists():
        out.replace(out.with_name(f"{out.name}.bak.{time.strftime('%Y%m%d-%H%M%S')}"))
    out.write_text("\n".join(L) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
