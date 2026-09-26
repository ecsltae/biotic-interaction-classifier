#!/usr/bin/env python3
"""Build the combined direction gold set from the curation workbooks.

Merges the two annotation blocks into one schema:
  - direction_curation_220.xlsx  sheet 'annotate', column DIRECTION  (F / R / N / U)
  - direction_curation_v2.xlsx   sheet 'annotate', column SUBJECT_IS (1 / 2 / ?)
  - direction_curation_SHORT.xlsx sheet 'annotate', column SUBJECT_IS (optional 2nd annotator)

Output columns:
  item_id, block, row_id, source, sentence, species1, relation, species2,
  gold          : FORWARD | REVERSE | UNDECIDABLE
  gold_raw      : the original cell value, kept verbatim
  decidable     : bool
  stored_pred   : always FORWARD (the stored-order baseline)
  syn_class     : coarse syntactic class of the relation surface form
  robi_id       : ROBI concept matched by the surface form, or empty
  robi_inverse  : ROBI concept id of the lexical inverse, or empty
  in_med25_pool : whether the sentence appears verbatim in the distillation pool
  pool_doc_id   : doc_id of that pool passage (use it to hold the document out)

Usage: python scripts/build_direction_gold.py [-o OUT.csv]
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "evaluation"
ROBI = ROOT / "data" / "taxonomies" / "robiext_v2025.json"
POOL = ROOT / "data" / "training" / "distill" / "med25_pool.parquet"

# ROBI ships no inverseOf field; this map is built by hand from the preferred terms.
ROBI_INVERSE_PAIRS = [
    ("RO_0002444", "RO_0002445"),  # parasite of        / parasitized by
    ("RO_0002453", "RO_0002454"),  # host of            / has host
    ("RO_0002439", "RO_0002458"),  # preys on           / preyed upon by
    ("RO_0002455", "RO_0002456"),  # pollinates         / pollinated by
    ("RO_0002459", "RO_0002460"),  # is vector for      / has vector
    ("RO_0002470", "RO_0002471"),  # eats               / is eaten by
    ("RO_0002618", "RO_0002619"),  # visits             / visited by
    ("RO_0002634", "RO_0002635"),  # endoparasite of    / has endoparasite
    ("RO_0002208", "RO_0002209"),  # parasitoid of      / has parasitoid
    ("RO_0002457", "RO_0002469"),  # acquires nutrients from / provides nutrients for
]


def normalise_text(value: object) -> str:
    """Collapse whitespace and lowercase, for exact-match comparison."""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def load_robi(path: Path = ROBI) -> tuple[dict[str, str], dict[str, str]]:
    """Return (surface form -> concept id, concept id -> inverse concept id)."""
    data = json.loads(path.read_text())
    surface: dict[str, str] = {}
    for concept in data["concepts"]:
        terms = [concept["preferred_term"]["term"]]
        terms += [s["term"] for s in concept.get("synonyms", [])]
        for term in terms:
            surface.setdefault(term.lower().strip(), concept["id"])
    inverse: dict[str, str] = {}
    for a, b in ROBI_INVERSE_PAIRS:
        inverse[a] = b
        inverse[b] = a
    return surface, inverse


def match_robi(relation: object, surface: dict[str, str]) -> str:
    """Map a pipe-separated relation surface form onto a ROBI concept id."""
    forms = [f.strip().lower() for f in str(relation).split("|") if f.strip()]
    for form in forms:
        if form in surface:
            return surface[form]
    for form in forms:
        for variant in (form.rstrip("s"), f"{form} of", f"{form.rstrip('s')} of", f"{form}s"):
            if variant in surface:
                return surface[variant]
    return ""


def syntactic_class(relation: object) -> str:
    """Coarse syntactic class of the relation surface form."""
    form = str(relation).split("|")[0].strip().lower()
    if re.search(r"\bby\b", form):
        return "C_passive_by"
    if re.search(r"\b(of|on|for|to|with|in)\s*$", form):
        return "B_prep_headed"
    if form.endswith("ing"):
        return "D_present_participle"
    if form.endswith("ed"):
        return "E_past_participle_bare"
    return "A_bare_noun"


_GOLD_MAP = {
    "F": "FORWARD", "FORWARD": "FORWARD", "1": "FORWARD", "1.0": "FORWARD",
    "R": "REVERSE", "REVERSE": "REVERSE", "2": "REVERSE", "2.0": "REVERSE",
    "N": "UNDECIDABLE", "NEITHER": "UNDECIDABLE",
    "U": "UNDECIDABLE", "UNCLEAR": "UNDECIDABLE", "?": "UNDECIDABLE",
}


def map_gold(raw: object) -> str:
    """Map a raw annotation cell onto FORWARD / REVERSE / UNDECIDABLE."""
    return _GOLD_MAP.get(str(raw).strip().upper(), "")


def read_block(path: Path, column: str, block: str) -> pd.DataFrame:
    """Read one curation workbook and keep only the annotated rows."""
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_excel(path, sheet_name="annotate")
    if column not in frame.columns:
        return pd.DataFrame()
    frame = frame[frame[column].notna()].copy()
    if frame.empty:
        return frame
    frame["block"] = block
    frame["gold_raw"] = frame[column].astype(str).str.strip()
    frame["gold"] = frame[column].map(map_gold)
    keep = ["block", "row_id", "source", "sentence", "species1", "relation",
            "species2", "gold", "gold_raw"]
    return frame[keep]


def build() -> pd.DataFrame:
    """Assemble the combined gold table with its provenance columns."""
    blocks = [
        read_block(EVAL / "direction_curation_220.xlsx", "DIRECTION", "gold20"),
        read_block(EVAL / "direction_curation_v2.xlsx", "SUBJECT_IS", "v2_112"),
        read_block(EVAL / "direction_curation_SHORT.xlsx", "SUBJECT_IS", "short_35"),
    ]
    blocks = [b for b in blocks if not b.empty]
    if not blocks:
        return pd.DataFrame()
    gold = pd.concat(blocks, ignore_index=True)

    # A row_id annotated in more than one block is a double annotation, not a duplicate item.
    gold["dup_row_id"] = gold.row_id.duplicated(keep=False)

    surface, inverse = load_robi()
    gold["syn_class"] = gold.relation.map(syntactic_class)
    gold["robi_id"] = [match_robi(r, surface) for r in gold.relation]
    gold["robi_inverse"] = gold.robi_id.map(lambda c: inverse.get(c, ""))
    gold["decidable"] = gold.gold.isin(["FORWARD", "REVERSE"])
    gold["stored_pred"] = "FORWARD"

    if POOL.exists():
        pool = pd.read_parquet(POOL, columns=["passage", "doc_id"])
        pool["nz"] = pool.passage.map(normalise_text)
        lookup = pool.drop_duplicates("nz").set_index("nz").doc_id
        nz = gold.sentence.map(normalise_text)
        gold["pool_doc_id"] = nz.map(lookup).fillna("")
    else:
        gold["pool_doc_id"] = ""
    gold["in_med25_pool"] = gold.pool_doc_id != ""

    gold.insert(0, "item_id", range(1, len(gold) + 1))
    return gold


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", type=Path,
                        default=EVAL / "direction_gold_combined.csv")
    args = parser.parse_args()

    gold = build()
    if gold.empty:
        print("no annotated rows found in any block")
        return
    gold.to_csv(args.out, index=False)

    n = len(gold)
    dec = int(gold.decidable.sum())
    print(f"wrote {args.out}  ({n} annotated items)")
    print(gold.groupby("block").gold.value_counts().to_string())
    print(f"\ndecidable          : {dec}/{n} ({dec / n:.1%})")
    print(f"undecidable        : {n - dec}/{n} ({(n - dec) / n:.1%})")
    if dec:
        fwd = int((gold.gold == "FORWARD").sum())
        print(f"stored-order acc   : {fwd}/{dec} = {fwd / dec:.3f}  (on decidable items)")
    print(f"double-annotated   : {int(gold.dup_row_id.sum())} rows")
    print(f"in med25_pool      : {int(gold.in_med25_pool.sum())}/{n} "
          f"-> hold out these doc_ids before any distant-supervision training")
    print(f"robi inverse known : {int((gold.robi_inverse != '').sum())}/{n}")


if __name__ == "__main__":
    main()
