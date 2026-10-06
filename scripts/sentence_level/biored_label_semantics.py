#!/usr/bin/env python3
"""What the derived BioRED sentence label means, and how far it can be from "states a relation".

The sentence benchmark (`data/benchmarks/biored_bc8_sentence`) labels a sentence positive iff it
co-mentions at least one pair of annotated concepts that BioRED relates somewhere in the abstract
(one of its candidate rows in `data/benchmarks/biored_bc8` has label 1). BioRED annotates relations
per document (pmid, concept pair, relation type, novelty); it records no evidence sentence. So a
positive sentence is one that co-mentions a related pair, which is not the same as one that states
the relation. This script measures how much room the annotations leave between the two.

Units. A *related pair* is a (pmid, unordered concept-key pair) with label 1 in the candidate file,
where the concept key is the sorted set of the entity's concept IDs (as in
`scripts/convert_biored_verify.py`). Its *k* is the number of sentences of the abstract that
co-mention it (that hold a candidate row for it). Because some mentions carry several IDs (e.g. a
gene normalised to a human and a rat ID), the same numbers are also given with the BioRED relation
itself (pmid, unordered ID pair from the relation lines) as the unit: a candidate row co-mentions
every gold relation whose two IDs it carries.

Positive sentences.
  * k=1 sentence: holds at least one related pair co-mentioned in no other sentence. If that
    relation is stated within any single sentence at all, it is stated in this one.
  * ambiguous-only sentence: every related pair it holds is co-mentioned in >= 2 sentences, so its
    positive status could rest entirely on co-mentions whose relation is stated elsewhere.

Bounds on the share of positive sentences that state none of their related pairs ("non-stating").
  * 0 if every sentence that co-mentions a related pair states it (optimistic; the annotations do
    not rule it out).
  * Assume instead that each related pair is stated in at least one of its k co-mentioning
    sentences, possibly only one. A positive sentence is non-stating iff it is not a stating
    sentence for any of its pairs. The largest number of non-stating sentences is then P minus the
    smallest set of positive sentences that together co-mention every related pair: a minimum set
    cover, solved exactly per abstract as a 0/1 integer program (scipy.optimize.milp). The simple
    bound "number of ambiguous-only sentences" is also valid (a sentence holding a k=1 pair must
    state it under this assumption) but looser.
  * Illustration, not a bound: if each related pair's stating sentence were one of its k
    co-mentioning sentences chosen uniformly and independently, a sentence is non-stating with
    probability prod_p (1 - 1/k_p); the sum over sentences is the expected count.
  * With no assumption at all (a relation may be stated only across sentences, or only inferred
    from the whole abstract), the annotations do not bound the share below 100%.
Each figure is also split by position: title sentences versus the rest. A title sentence is one
that starts inside the PubTator title span; this is sentence index 0 except where spaCy splits a
title at a colon or similar (then the title fragments are all title sentences) or runs the title
into the first abstract sentence (then that merged sentence 0 counts as a title sentence). Without
the raw files the rule falls back to sentence index 0. For each position the script gives both
the joint maximum's composition (range over all optimal covers) and the maximum for that position
taken alone (a cover that uses as few sentences of that position as possible).

Document level (needs the raw PubTator files of the BioREDirect release). Every BioRED relation of
the split's abstracts is put in one of: co-mentioned by a sentence-level candidate row (visible to
the sentence benchmark); co-mentioned in one sentence but dropped by the candidate filters (entity
types not related in train, identical surface strings); never co-mentioned in a single sentence
(cross-sentence only, invisible to sentence-scope candidates); or an ID absent from the entity
annotations. The script re-segments the abstracts exactly as the converter did (spaCy
en_core_web_sm) and checks that it reproduces the candidate CSV row for row before using the
segmentation. Without the raw files, only the CSV-based parts are computed.

Splits follow the BC8 protocol used to build `biored_bc8`: train = BioRED train+dev, dev = BioRED
test (100 abstracts), test = BioCreative VIII test (400 abstracts).

Usage
  python3 scripts/sentence_level/biored_label_semantics.py --split test dev
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from convert_biored_verify import concept_key, read_pubtator, related, relation_type_pairs  # noqa: E402

CAND = REPO / "data/benchmarks/biored_bc8"
SENT = REPO / "data/benchmarks/biored_bc8_sentence"
DOC = REPO / "data/benchmarks/biored_bc8_doc"
OUT = REPO / "results/paperA_v2/sentence_level/biored_label_semantics.json"
RAW_CANDIDATES = [REPO / "data/raw/bioredirect", REPO.parents[1] / "biored_baseline/bioredirect"]
RAW_FILES = {"train": "train_dev", "dev": "test", "test": "bc8_test"}   # BC8 protocol
SPLIT_ROLE = {"train": "BioRED train + dev (500 abstracts)", "dev": "BioRED test (100 abstracts)",
              "test": "BioCreative VIII BioRED test (400 abstracts)"}


# ------------------------------------------------------------------------------------ helpers
def frac(n: int | float, d: int | float) -> float:
    return round(float(n) / d, 4) if d else float("nan")


def nshare(n: int | float, d: int | float) -> dict:
    return {"n": int(n) if float(n).is_integer() else round(float(n), 2), "share": frac(n, d)}


def sent_index(sent_id: str) -> int:
    return int(str(sent_id).rsplit("_", 1)[1])


def k_bucket(k: int) -> str:
    return f"k={k}" if k <= 3 else "k>=4"


def min_cover(sets: dict[str, set], cost: dict[str, float] | None = None) -> set[str]:
    """Cheapest set of keys of `sets` whose union covers the union of all values (exact 0/1 ILP).

    Every key costs 1 unless `cost` says otherwise; with unit costs this is a minimum set cover.
    """
    keys = list(sets)
    elems = sorted(set().union(*sets.values()), key=str)
    if not elems:
        return set()
    eidx = {e: i for i, e in enumerate(elems)}
    A = lil_matrix((len(elems), len(keys)))
    for j, s in enumerate(keys):
        for e in sets[s]:
            A[eidx[e], j] = 1
    c = np.array([(cost or {}).get(s, 1.0) for s in keys])
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lb=1, ub=np.inf),
               integrality=np.ones(len(keys)), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"set cover ILP failed: {res.message}")
    chosen = {s for s, x in zip(keys, res.x) if x > 0.5}
    assert set().union(*(sets[s] for s in chosen)) == set(elems)
    return chosen


# ------------------------------------------------------------------ sentence-level analysis
def analyse_positive_sentences(sent_pairs: dict[str, set], k: dict, is_title: dict[str, bool]) -> dict:
    """Ambiguity and non-stating bounds for positive sentences.

    sent_pairs: positive sent_id -> set of related-pair keys (pmid-qualified) it co-mentions.
    k: related-pair key -> number of sentences co-mentioning it.
    Under "each related pair is stated in >= 1 of its co-mentioning sentences", the stating
    sentences must cover every related pair; every positive sentence outside the cover can be
    non-stating. So the maximum number of non-stating sentences in a set S = |S| minus the
    fewest sentences of S that any cover must use, found per abstract by an ILP whose costs
    make S's sentences expensive (1) and the others nearly free (1e-4).
    """
    P = len(sent_pairs)
    has_k1 = {s for s, ps in sent_pairs.items() if any(k[p] == 1 for p in ps)}
    amb = set(sent_pairs) - has_k1
    p_non = {s: float(np.prod([1 - 1 / k[p] for p in ps])) for s, ps in sent_pairs.items()}
    npairs = Counter(min(len(ps), 3) for ps in sent_pairs.values())
    titles = {s for s in sent_pairs if is_title[s]}
    body = set(sent_pairs) - titles

    by_doc = defaultdict(dict)
    for s, ps in sent_pairs.items():
        by_doc[s.rsplit("_", 1)[0]][s] = ps
    eps = 1e-4                                    # < 1 / (sentences per abstract): only breaks ties
    cover, cov_t_hi, cov_t_lo, cov_titles_only, cov_body_only = (set() for _ in range(5))
    for d in by_doc.values():
        cover |= min_cover(d)                                                    # joint maximum
        cov_t_hi |= min_cover(d, {s: 1 + eps * is_title[s] for s in d})          # ... most titles left out
        cov_t_lo |= min_cover(d, {s: 1 - eps * is_title[s] for s in d})          # ... fewest titles left out
        cov_titles_only |= min_cover(d, {s: 1.0 if is_title[s] else eps for s in d})
        cov_body_only |= min_cover(d, {s: eps if is_title[s] else 1.0 for s in d})
    assert len(cover) == len(cov_t_hi) == len(cov_t_lo)
    non = set(sent_pairs) - cover
    non_t_hi, non_t_lo = len(titles - cov_t_hi), len(titles - cov_t_lo)

    def block(members: set[str], cov_sep: set[str]) -> dict:
        n = len(members)
        return {"n_positive_sentences": n,
                "with_a_k1_pair": nshare(len(members & has_k1), n),
                "ambiguous_only": nshare(len(members & amb), n),
                "max_non_stating_this_position_alone": nshare(len(members - cov_sep), n),
                "expected_non_stating_uniform_model": nshare(sum(p_non[s] for s in members), n)}

    return {
        "n_positive_sentences": P,
        "with_a_k1_pair": nshare(len(has_k1), P),
        "ambiguous_only": nshare(len(amb), P),
        "n_related_pairs_per_positive_sentence": {"1": nshare(npairs[1], P), "2": nshare(npairs[2], P),
                                                  ">=3": nshare(npairs[3], P)},
        "non_stating_bounds": {
            "optimistic_every_co_mention_states": nshare(0, P),
            "simple_bound_ambiguous_only": nshare(len(amb), P),
            "max_if_each_pair_stated_in_at_least_one_co_mentioning_sentence": {
                **nshare(len(non), P),
                "min_stating_sentences_needed": len(cover),
                "method": "exact minimum set cover per abstract (0/1 integer program)",
                "titles_among_them_range_over_optimal_solutions": [non_t_lo, non_t_hi],
                "other_sentences_among_them_range": [len(non) - non_t_hi, len(non) - non_t_lo]},
            "expected_uniform_model_illustrative": nshare(sum(p_non.values()), P),
            "no_assumption": nshare(P, P),
        },
        "by_position": {"title_sentences": block(titles, cov_titles_only),
                        "other_sentences": block(body, cov_body_only)},
    }


def k_distribution(k: dict) -> dict:
    n = len(k)
    c = Counter(k_bucket(v) for v in k.values())
    vals = np.array(list(k.values()))
    return {"n_related_pairs_co_mentioned": n,
            **{b: nshare(c[b], n) for b in ("k=1", "k=2", "k=3", "k>=4")},
            "k_mean": round(float(vals.mean()), 3), "k_median": float(np.median(vals)), "k_max": int(vals.max())}


def csv_analysis(cand: pd.DataFrame, is_title: dict[str, bool]) -> tuple[dict, dict]:
    """k distribution and positive-sentence analysis with concept-key pairs (the CSV's own unit)."""
    pos = cand[cand.label == 1].copy()
    pos["pair"] = [f"{p}::" + "~".join(sorted((str(a), str(b)))) for p, a, b in zip(pos.pmid, pos.cid1, pos.cid2)]
    k = pos.groupby("pair").sent_id.nunique().to_dict()
    sent_pairs = pos.groupby("sent_id").pair.agg(set).to_dict()
    return k_distribution(k), analyse_positive_sentences(sent_pairs, k, is_title)


# ----------------------------------------------------------------------- raw-data analysis
def rebuild(docs: list[dict], type_pairs: set, nlp) -> tuple[pd.DataFrame, dict, dict]:
    """Re-run the converter's candidate construction, keeping concept IDs and sentence spans."""
    rows, spans, inside_by_sent = [], {}, {}
    for d in docs:
        sents = [(s.start_char, s.end_char) for s in nlp(d["text"]).sents]
        spans[d["pmid"]] = sents
        for si, (s0, s1) in enumerate(sents):
            inside = [e for e in d["ents"] if e["start"] >= s0 and e["end"] <= s1]
            inside_by_sent[(d["pmid"], si)] = inside
            first = {}
            for e in sorted(inside, key=lambda e: e["start"]):
                first.setdefault(concept_key(e), e)
            concepts = list(first.values())
            for a, b in itertools.combinations(concepts, 2):
                if frozenset((a["type"], b["type"])) not in type_pairs:
                    continue
                if a["mention"].lower() == b["mention"].lower():
                    continue
                rows.append({"pmid": d["pmid"], "sent_id": f"{d['pmid']}_{si}", "cid1": concept_key(a),
                             "cid2": concept_key(b), "label": int(related(d, a, b)),
                             "ids1": a["ids"], "ids2": b["ids"], "text": d["text"][s0:s1].strip()})
    return pd.DataFrame(rows), spans, inside_by_sent


def load_raw(split: str, src: Path, nlp, type_pairs: set) -> dict:
    """Read the split's PubTator file and re-segment it exactly as the converter did."""
    path = src / f"bioredirect_{RAW_FILES[split]}.pubtator"
    docs = read_pubtator(path)
    rb, spans, inside_by_sent = rebuild(docs, type_pairs, nlp)
    title_len = {d["pmid"]: len(d["title"]) for d in docs}
    # a sentence is a title sentence if it starts inside the title (catches split titles too)
    is_title = {f"{p}_{i}": a < title_len[p] for p, ss in spans.items() for i, (a, _) in enumerate(ss)}
    return {"path": path, "docs": docs, "rb": rb, "spans": spans, "inside_by_sent": inside_by_sent,
            "is_title": is_title}


def raw_analysis(raw: dict, cand: pd.DataFrame, is_title: dict[str, bool]) -> dict:
    path, docs, rb, spans, inside_by_sent = (raw[k] for k in ("path", "docs", "rb", "spans", "inside_by_sent"))
    lines = [l.split("\t") for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rel_lines = [f for f in lines if len(f) == 5]
    role_lines = [f for f in lines if len(f) == 4]
    out: dict = {"raw_file": os.path.relpath(path, REPO),
                 "n_documents": len(docs),
                 "relation_lines": len(rel_lines),
                 "relation_types": dict(Counter(f[1] for f in rel_lines).most_common()),
                 "novelty": dict(Counter(f[4] for f in rel_lines).most_common()),
                 "direction_lines_subject_role": len(role_lines),
                 "line_kinds_by_field_count": dict(sorted(Counter(len(f) for f in lines).items()))}

    key = ["sent_id", "cid1", "cid2", "label"]
    a = Counter(map(tuple, cand[key].astype({"cid1": str, "cid2": str}).values.tolist()))
    b = Counter(map(tuple, rb[key].astype({"cid1": str, "cid2": str}).values.tolist()))
    txt_csv = cand.drop_duplicates("sent_id").set_index("sent_id").text.sort_index()
    txt_rb = rb.drop_duplicates("sent_id").set_index("sent_id").text.reindex(txt_csv.index)
    out["reproduction_check"] = {"candidate_rows_csv": int(sum(a.values())), "candidate_rows_rebuilt": int(sum(b.values())),
                                 "rows_identical": a == b,
                                 "sentence_texts_identical": bool((txt_csv == txt_rb).all())}

    # title check: is sentence 0 exactly the title?
    eq = split_title = merged = 0
    for d in docs:
        sents, tl = spans[d["pmid"]], len(d["title"])
        eq += d["text"][sents[0][0]:sents[0][1]].strip() == d["title"].strip()
        split_title += len(sents) > 1 and sents[1][0] < tl
        merged += sents[0][1] > tl + 1
    in_csv = set(cand.sent_id)
    out["title_check"] = {"n_documents": len(docs), "sentence_0_equals_title": eq,
                          "title_split_over_several_sentences": int(split_title),
                          "sentence_0_runs_into_abstract": int(merged),
                          "benchmark_sentences_title_by_index_0": sum(sent_index(s) == 0 for s in in_csv),
                          "benchmark_sentences_title_by_span": sum(is_title[s] for s in in_csv),
                          "rule_used": "title = sentence starting inside the title span"}

    # gold-relation unit: which BioRED relations each positive candidate row co-mentions
    rels_by_doc = {d["pmid"]: d["rels"] for d in docs}
    pos = rb[rb.label == 1]
    sent_rels: dict[str, set] = defaultdict(set)
    for r in pos.itertuples():
        rels = rels_by_doc[r.pmid]
        for x in r.ids1:
            for y in r.ids2:
                fs = frozenset((x, y))
                if fs in rels:
                    sent_rels[r.sent_id].add((r.pmid, fs))
    assert set(sent_rels) == set(pos.sent_id), "a positive row co-mentions no BioRED relation"
    k_rel = Counter(rr for rs in sent_rels.values() for rr in rs)
    out["gold_relation_unit"] = {"k_distribution": {**k_distribution(dict(k_rel)),
                                                    "self_relations_among_them": sum(len(r[1]) == 1 for r in k_rel)},
                                 "positive_sentences": analyse_positive_sentences(dict(sent_rels), dict(k_rel), is_title)}

    # document level: where does each BioRED relation fall?
    cats = Counter()
    self_rel = 0
    for d in docs:
        sents = spans[d["pmid"]]
        sents_of = defaultdict(set)
        for si in range(len(sents)):
            for e in inside_by_sent[(d["pmid"], si)]:
                for i in e["ids"]:
                    sents_of[i].add(si)
        all_ids = {i for e in d["ents"] for i in e["ids"]}
        for r in d["rels"]:
            if len(r) == 1:
                self_rel += 1
                continue
            x, y = tuple(r)
            if x not in all_ids or y not in all_ids:
                cats["id_not_in_entity_annotations"] += 1
            elif (d["pmid"], r) in k_rel:
                cats["co_mentioned_by_a_sentence_candidate"] += 1
            elif sents_of[x] & sents_of[y]:
                cats["co_mentioned_but_no_candidate_row"] += 1
            else:
                cats["never_co_mentioned_in_one_sentence"] += 1
    n_rel = sum(cats.values())
    out["document_level_relations"] = {
        "n_relations_unique_unordered_pairs": n_rel, "self_relations_excluded": self_rel,
        **{c: nshare(cats[c], n_rel) for c in ("co_mentioned_by_a_sentence_candidate",
                                               "co_mentioned_but_no_candidate_row",
                                               "never_co_mentioned_in_one_sentence",
                                               "id_not_in_entity_annotations")}}
    return out


# ------------------------------------------------------------------------ annotation inventory
PUBTATOR_LINE_KINDS = {
    "1": "title line (pmid|t|text) or abstract line (pmid|a|text)",
    "4": "direction: pmid, id1, id2, Subject:<id>",
    "5": "relation: pmid, relation type, id1, id2, novelty (Novel / No)",
    "6": "entity mention: pmid, start, end, surface string, entity type, concept ID(s)",
}
PROCESSED_COLUMNS = ["pmid", "type1", "type2", "id1", "id2", "whole-abstract model input",
                     "relation type", "novelty", "subject id"]


def inventory(src: Path | None) -> dict:
    """Which BioRED annotation files exist locally, and what each kind of record carries."""
    out: dict = {"benchmark_csv_columns": {}}
    for d in sorted(REPO.glob("data/benchmarks/biored*")):
        f = d / "test.csv"
        if f.exists():
            out["benchmark_csv_columns"][str(d.relative_to(REPO))] = f.open().readline().strip().split(",")
    if src is None:
        out["raw"] = "not found"
        return out
    out["raw_pubtator_field_counts"] = {
        os.path.relpath(f, REPO): dict(sorted(Counter(len(l.split("\t")) for l in f.read_text(encoding="utf-8")
                                                       .splitlines() if l.strip()).items()))
        for f in sorted(src.glob("bioredirect_*.pubtator"))}
    out["raw_pubtator_line_kinds"] = PUBTATOR_LINE_KINDS
    proc = sorted((src / "processed").glob("*.tsv"))
    if proc:
        out["bioredirect_processed_tsv"] = {
            "files": [os.path.relpath(f, REPO) for f in proc],
            "field_counts": {f.name: dict(Counter(l.count("\t") + 1 for l in f.open(encoding="utf-8"))) for f in proc},
            "columns": PROCESSED_COLUMNS}
    out["finding"] = ("No sentence-level or evidence annotation anywhere: relations are (pmid, id1, id2, type, "
                      "novelty) plus a subject role; no record points to a sentence or text span.")
    return out


# ---------------------------------------------------------------------------------- driver
def analyse_split(split: str, src: Path | None, nlp, type_pairs: set | None) -> dict:
    cand = pd.read_csv(CAND / f"{split}.csv", dtype={"pmid": str, "cid1": str, "cid2": str})
    sent = pd.read_csv(SENT / f"{split}.csv")
    raw = load_raw(split, src, nlp, type_pairs) if src is not None else None
    if raw is not None:
        is_title = raw["is_title"]
        title_rule = "sentence starting inside the title span (raw PubTator title)"
    else:
        is_title = {s: sent_index(s) == 0 for s in sent.sent_id}
        title_rule = "sentence index 0 (raw files not found)"
    is_title = {s: is_title[s] for s in sent.sent_id}
    agg = cand.groupby("sent_id").label.max()
    k_dist, pos = csv_analysis(cand, is_title)
    res: dict = {
        "split_role": SPLIT_ROLE[split],
        "files": {"candidates": str((CAND / f"{split}.csv").relative_to(REPO)),
                  "sentences": str((SENT / f"{split}.csv").relative_to(REPO))},
        "n_documents_with_candidates": int(cand.pmid.nunique()),
        "n_sentences": len(sent), "n_positive_sentences": int(sent.label.sum()),
        "positive_rate": frac(sent.label.sum(), len(sent)),
        "title_rule": title_rule,
        "n_title_sentences": int(sum(is_title.values())),
        "positive_rate_title": frac(sent.label[sent.sent_id.map(is_title)].sum(), sum(is_title.values())),
        "positive_rate_other": frac(sent.label[~sent.sent_id.map(is_title)].sum(), len(sent) - sum(is_title.values())),
        "sentence_label_equals_max_candidate_label": bool((agg.reindex(sent.sent_id).values == sent.label.values).all()),
        "concept_key_unit": {"k_distribution": k_dist, "positive_sentences": pos},
    }
    doc_csv = DOC / f"{split}.csv"
    if doc_csv.exists():
        dd = pd.read_csv(doc_csv, usecols=["label", "co_mentioned"])
        res["crosscheck_doc_csv"] = {
            "file": str(doc_csv.relative_to(REPO)),
            "related_concept_pairs": int(dd.label.sum()),
            "related_pairs_never_sharing_a_sentence": nshare(int(((dd.label == 1) & (dd.co_mentioned == 0)).sum()),
                                                             int(dd.label.sum())),
            "note": "concept-key pairs after the candidate filters; sentence of a mention = sentence of its start"}
    if raw is not None:
        res["raw"] = raw_analysis(raw, cand, is_title)
    else:
        res["raw"] = "raw BioREDirect PubTator files not found: document-level relation lists unavailable"
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", nargs="+", choices=("train", "dev", "test"), default=["test"])
    ap.add_argument("--src", type=Path, default=None, help="directory with bioredirect_*.pubtator")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    src = a.src or next((p for p in RAW_CANDIDATES if (p / "bioredirect_bc8_test.pubtator").exists()), None)
    nlp = type_pairs = None
    if src is not None:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        type_pairs = relation_type_pairs(read_pubtator(src / f"bioredirect_{RAW_FILES['train']}.pubtator"))

    result = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "script": str(Path(__file__).resolve().relative_to(REPO)),
        "command": "python3 scripts/sentence_level/biored_label_semantics.py --split " + " ".join(a.split),
        "raw_source_found": src is not None,
        "annotation_inventory": inventory(src),
        "definitions": {
            "positive_sentence": "co-mentions >= 1 candidate concept pair that BioRED relates anywhere in the abstract",
            "related_pair_concept_key_unit": "(pmid, unordered pair of concept keys) with label 1 in the candidate CSV",
            "related_pair_gold_relation_unit": "(pmid, unordered ID pair) of a BioRED relation line, co-mentioned by a positive candidate row",
            "k": "number of sentences of the abstract holding a candidate row for the related pair",
            "with_a_k1_pair": "positive sentence holding >= 1 related pair co-mentioned in no other sentence",
            "ambiguous_only": "positive sentence all of whose related pairs are co-mentioned in >= 2 sentences",
            "non_stating": "positive sentence that states none of its related pairs",
            "max_if_each_pair_stated_in_at_least_one_co_mentioning_sentence":
                "P minus the minimum number of positive sentences that co-mention every related pair (exact set cover per abstract)",
            "expected_uniform_model_illustrative":
                "sum over positive sentences of prod_p (1 - 1/k_p): expected non-stating count if each pair's stating sentence were uniform among its k; not a bound",
            "title_sentence": "sentence that starts inside the PubTator title span; for 384/400 test abstracts this is exactly sentence index 0 (see raw.title_check)",
        },
        "splits": {s: analyse_split(s, src, nlp, type_pairs) for s in a.split},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        shutil.copy2(a.out, a.out.with_name(f"{a.out.name}.bak.{time.strftime('%H%M')}"))
    a.out.write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(f"wrote {a.out}")
    for s, r in result["splits"].items():
        u = r["concept_key_unit"]
        b = u["positive_sentences"]["non_stating_bounds"]
        print(f"{s}: {r['n_sentences']} sentences, {r['n_positive_sentences']} positive; "
              f"k=1 pairs {u['k_distribution']['k=1']['share']:.3f}; positive with a k=1 pair "
              f"{u['positive_sentences']['with_a_k1_pair']['share']:.3f}; ambiguous-only "
              f"{u['positive_sentences']['ambiguous_only']['share']:.3f}; max non-stating "
              f"{b['max_if_each_pair_stated_in_at_least_one_co_mentioning_sentence']['share']:.3f}")


if __name__ == "__main__":
    main()
