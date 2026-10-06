#!/usr/bin/env python3
"""The SENTENCE-level question on BioRED (BC8 protocol), where the sentence label derives from gold.

A test sentence (one with at least one candidate pair) is positive iff it co-mentions at least one
entity pair that BioRED relates in the document. BioRED's relations are document-level, so the label
means "co-mentions a related pair", not "states a relation" (see biored_label_semantics.py).

Arms (BiomedBERT-base, 3 seeds, the paper's recipe; ensemble = mean over seeds):
  sentlab   sentence input, trained on these sentence labels, one row per sentence
            (models/sandbox_sentence/biored_sentlab_s{k}): the fair sentence-level competitor
  sentence  the paper's sentence arm (trained on candidate rows); max over the sentence's candidates
            (its score does not depend on the candidate)
  pair-max  the paper's pair arm, order-free, max over the sentence's candidate pairs
Thresholds: the F1-maximising threshold of each arm's ensemble on the development split (BioRED's
own test split under the BC8 protocol), applied once to the BC8 test sentences.

Statistics: AUPRC per seed (mean +- sd) and of the ensemble; paired bootstrap over sentences (10,000
replicates, seed 0) for AUPRC and F1 differences, repeated with abstracts as clusters; exact McNemar
at the development thresholds; strata by concepts named in the sentence (2 vs >=3, the paper's
definition), entities in the candidate pairs (2 vs >=3) and candidate pairs (1 vs >=2); the pair
precision of a perfect sentence filter; zero-shot LLM rows on sentences whose every candidate was
scored.

Scores are computed with TF32 matmuls (torch "high"), as in scripts/paperA_tables.py; `--fp32` uses
full FP32 (separate cache), which is what the sandbox script did.

Usage
  python3 scripts/sentence_level/biored_sentence.py            # -> results/paperA_v2/sentence_level/biored_sentence.json
  python3 scripts/sentence_level/biored_sentence.py --fp32     # -> biored_sentence_fp32.json (reproduction check)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import REPO  # noqa: E402
from paperA_tables import order_free, prf  # noqa: E402

SRC = REPO / "data/benchmarks/biored_bc8"
SENT = REPO / "data/benchmarks/biored_bc8_sentence"
ARMS = {"sentlab": "models/sandbox_sentence/biored_sentlab_s{}", "sentence": "models/biored_bc8/sentence_s{}",
        "pair-max": "models/biored_bc8/pair_s{}"}
SEEDS = (1, 2, 3)
GRID = np.arange(0.01, 1.0, 0.01)
LLM_DIR = REPO / "results/paperA_v2/llm"
FILL = C.OUT / "llm"
SANDBOX_JSON = REPO / "sandbox/sentence_level/results/biored_sentence.json"
PAIRS = [("pair-max", "sentlab"), ("pair-max", "sentence"), ("sentlab", "sentence")]


# ---------------------------------------------------------------- data and scores
def candidates(split: str) -> pd.DataFrame:
    d = pd.read_csv(SRC / f"{split}.csv", dtype={"pmid": str})
    return d.rename(columns={"source_species": "species1", "target_species": "species2", "text": "sentence"}) \
        .assign(relation="")


def sentences(split: str, cand: pd.DataFrame) -> pd.DataFrame:
    """The sentence table (data/benchmarks/biored_bc8_sentence), checked against the candidates,
    with the strata columns."""
    s = pd.read_csv(SENT / f"{split}.csv")
    g = cand.groupby("sent_id")
    assert (s.sent_id.to_numpy() == np.array(sorted(g.groups))).all()
    assert (s.label.to_numpy() == g.label.max().reindex(s.sent_id).to_numpy()).all()
    ent = g.apply(lambda x: len(set(x.cid1) | set(x.cid2)), include_groups=False).reindex(s.sent_id).to_numpy()
    return s.assign(pmid=g.pmid.first().reindex(s.sent_id).to_numpy(),
                    n_concepts=g.n_concepts.first().reindex(s.sent_id).to_numpy(),
                    n_cand_entities=ent, n_pairs=g.size().reindex(s.sent_id).to_numpy(),
                    n_pos_pairs=g.label.sum().reindex(s.sent_id).to_numpy())


def best_thr(y: np.ndarray, s: np.ndarray) -> float:
    """F1-maximising threshold on the grid (first maximum, as in the sandbox)."""
    return float(max(GRID, key=lambda t: f1_score(y, (s >= t).astype(int), zero_division=0)))


def arm_scores(fp32: bool, rescore: bool) -> tuple[dict, dict, dict]:
    """Per-seed sentence scores of every arm on dev and test (cached), and the tables."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("highest" if fp32 else "high")
    pre = "fp32_" if fp32 else ""
    cand = {s: candidates(s) for s in ("dev", "test")}
    sent = {s: sentences(s, cand[s]) for s in ("dev", "test")}
    S = {}
    for arm, pat in ARMS.items():
        S[arm] = {}
        for split in ("dev", "test"):
            per = []
            for k in SEEDS:
                md = REPO / pat.format(k)
                if arm == "sentlab":
                    q = sent[split].rename(columns={"text": "sentence"}).assign(species1="", species2="", relation="")
                    per.append(C.cached(f"{pre}biored_{split}_{arm}_s{k}", lambda: order_free(md, q, dev), rescore))
                else:
                    e = C.cached(f"{pre}biored_{split}_{arm}_cand_s{k}", lambda: order_free(md, cand[split], dev), rescore)
                    per.append(cand[split].assign(s=e).groupby("sent_id").s.max()
                               .reindex(sent[split].sent_id).to_numpy())
            S[arm][split] = per
    return S, cand, sent


# ---------------------------------------------------------------- statistics
def arm_block(y: np.ndarray, per: list[np.ndarray], thr: float) -> dict:
    e = np.mean(per, axis=0)
    aps = [float(ap(y, p)) for p in per]
    return {"auprc_per_seed": aps, "auprc_mean": float(np.mean(aps)), "auprc_sd": float(np.std(aps, ddof=1)),
            "auprc_ensemble": float(ap(y, e)), "threshold_dev": thr, **prf(y, (e >= thr).astype(int)),
            "secondary": {"auroc": float(roc_auc_score(y, e)), "ap_negative_class": float(ap(1 - y, -e))}}


def stratum(y, ens, preds, mask, pmid, B) -> dict:
    yy = y[mask]
    out = {"n": int(mask.sum()), "positives": int(yy.sum()), "positive_rate": float(yy.mean()),
           "trivial_accept_all_F1": float(f1_score(yy, np.ones_like(yy))), "arms": {}}
    if len(set(yy)) < 2:
        out["note"] = "one class only: AUPRC undefined"
        return out
    for a in ens:
        out["arms"][a] = {"auprc_ensemble": float(ap(yy, ens[a][mask])), **prf(yy, preds[a][mask])}
    out["bootstrap_auprc_sentences"] = C.boot_auprc(yy, {a: e[mask] for a, e in ens.items()}, PAIRS, B=B)
    out["bootstrap_auprc_abstracts"] = C.boot_auprc(yy, {a: e[mask] for a, e in ens.items()}, PAIRS,
                                                    clusters=pmid[mask], B=B)
    out["mcnemar_exact"] = {f"{a} vs {b}": C.mcnemar_exact(preds[b][mask], preds[a][mask], yy) for a, b in PAIRS}
    return out


def interaction(y, ens, m1, m2, a, b, clusters=None, B=C.N_BOOT) -> dict:
    """Bootstrap of (AUPRC_a - AUPRC_b on m2) - (same on m1): is the advantage larger in m2?"""
    vals = []
    for W in C.boot_weights(len(y), B, C.SEED, clusters):
        d2 = C.ap_weighted(y[m2], ens[a][m2], W[:, m2]) - C.ap_weighted(y[m2], ens[b][m2], W[:, m2])
        d1 = C.ap_weighted(y[m1], ens[a][m1], W[:, m1]) - C.ap_weighted(y[m1], ens[b][m1], W[:, m1])
        vals.append(d2 - d1)
    v = np.concatenate(vals)
    obs = (ap(y[m2], ens[a][m2]) - ap(y[m2], ens[b][m2])) - (ap(y[m1], ens[a][m1]) - ap(y[m1], ens[b][m1]))
    return {"observed": float(obs), "ci95": C.ci(v), "p_le_0": float((v <= 0).mean())}


# ---------------------------------------------------------------- LLM rows
def llm_sentence_rows(test_c: pd.DataFrame, sent: pd.DataFrame, ens: dict, preds: dict, B: int) -> dict:
    """Zero-shot LLMs at the sentence level, on sentences whose every candidate has a pair score.

    sentence-question score: the P(YES) of one call per sentence (the lowest-numbered candidate row
    that carries it; the paper's run asked it once per candidate row, and repeated calls differ
    slightly because of batched inference). pair-max: the max P(YES) over the sentence's candidates;
    its greedy verdict is YES if any candidate's is.
    """
    from biored_llm_fill import sample_sentences
    out = {"coverage": {}, "subsets": {
        "paper_run_complete": "sentences whose every candidate the paper's 3,000-candidate run scored; "
                              "biased to single-pair sentences (a sentence with many candidates is rarely complete)",
        "random500": "qwen3-32b only: 500 random test sentences (seed 0), completed by biored_llm_fill.py"}}
    sample500 = set(sample_sentences(test_c))
    sid_of_row = test_c.sent_id.to_numpy()
    y_all = sent.label.to_numpy()
    pos = {s: i for i, s in enumerate(sent.sent_id)}
    runs = [(f.stem[len("biored_"):-len("_pair")], "paper_run_complete") for f in sorted(LLM_DIR.glob("biored_*_pair.csv"))]
    runs.append(("qwen3-32b", "random500"))
    for name, subset in runs:
        pr = pd.read_csv(LLM_DIR / f"biored_{name}_pair.csv")
        sf = LLM_DIR / f"biored_{name}_sentence.csv"
        sr = pd.read_csv(sf) if sf.exists() else None
        if subset == "random500":                    # the 500-sentence completion (biored_llm_fill.py)
            fp, fs = FILL / "biored_qwen3-32b_sent500_pair.csv", FILL / "biored_qwen3-32b_sent500_sentence.csv"
            if fp.exists():
                pr = pd.concat([pr, pd.read_csv(fp)[pr.columns]], ignore_index=True)
            if fs.exists() and sr is not None:
                sr = pd.concat([sr, pd.read_csv(fs)[sr.columns]], ignore_index=True)
        tc_r = test_c.iloc[pr.row.to_numpy()]
        assert (pr.label.to_numpy() == tc_r.label.to_numpy()).all() \
            and (pr.sentence.astype(str).to_numpy() == tc_r.sentence.astype(str).to_numpy()).all() \
            and (pr.species1.astype(str).to_numpy() == tc_r.species1.astype(str).to_numpy()).all(), \
            f"biored_{name}_pair.csv ({subset}): rows misaligned with the test candidates"
        pr = pr.drop_duplicates("row").assign(sent_id=sid_of_row[pr.drop_duplicates("row").row])
        n_scored = pr.groupby("sent_id").size()
        n_all = test_c.groupby("sent_id").size()
        complete = n_scored.index[n_scored.to_numpy() == n_all.reindex(n_scored.index).to_numpy()]
        if sr is not None:
            sr = sr.assign(sent_id=sid_of_row[sr.row]).sort_values("row").drop_duplicates("sent_id")
            complete = complete.intersection(sr.sent_id)
        if subset == "random500":
            complete = complete.intersection(sorted(sample500))
            if len(complete) < len(sample500):      # an incomplete fill is not a random subset: no row
                print(f"random500: only {len(complete)} of 500 sentences complete; row skipped", file=sys.stderr)
                out["coverage"][f"{name} random500"] = {"pair_rows": int(len(pr)), "complete_sentences": int(len(complete)),
                                                        "row": "skipped: incomplete"}
                continue
        key = name if subset == "paper_run_complete" else f"{name} random500"
        out["coverage"][key] = {"pair_rows": int(len(pr)), "complete_sentences": int(len(complete))}
        if len(complete) < 30:
            continue
        idx = np.array(sorted(pos[s] for s in complete))
        sids = sent.sent_id.to_numpy()[idx]
        y = y_all[idx]
        g = pr.groupby("sent_id")
        pmax, vmax = g.p_yes.max().reindex(sids).to_numpy(), g.verdict.max().reindex(sids).to_numpy()
        r = {"paper_comparison_model": C.paper_comparison_llm(name),
             "n_sentences": int(len(idx)), "positive_rate": float(y.mean()),
             "n_pairs_distribution": {k: int(v) for k, v in
                                      pd.Series(sent.n_pairs.to_numpy()[idx]).clip(upper=4).value_counts().sort_index().items()},
             "trivial_accept_all_F1": float(f1_score(y, np.ones_like(y)))}
        sc = {"llm pair-max": pmax}
        r["llm pair-max"] = {"auprc": float(ap(y, pmax)), "greedy": prf(y, vmax.astype(int))}
        if sr is not None:
            s_ = sr.set_index("sent_id")
            ps, vs = s_.p_yes.reindex(sids).to_numpy(), s_.verdict.reindex(sids).to_numpy().astype(int)
            sc["llm sentence"] = ps
            r["llm sentence"] = {"auprc": float(ap(y, ps)), "greedy": prf(y, vs)}
        r["trained_arms_same_sentences"] = {a: {"auprc_ensemble": float(ap(y, e[idx])), **prf(y, preds[a][idx])}
                                            for a, e in ens.items()}
        if "llm sentence" in sc and len(set(y)) == 2:
            r["bootstrap_auprc"] = C.boot_auprc(y, sc, [("llm pair-max", "llm sentence")], B=B)
            multi = sent.n_pairs.to_numpy()[idx] >= 2
            r["by_n_pairs"] = {}
            for lab, m in (("1_pair", ~multi), ("ge2_pairs", multi)):
                if m.sum() >= 20 and len(set(y[m])) == 2:
                    r["by_n_pairs"][lab] = {"n": int(m.sum()), "positive_rate": float(y[m].mean()),
                                            "llm pair-max": float(ap(y[m], pmax[m])),
                                            "llm sentence": float(ap(y[m], sc["llm sentence"][m])),
                                            **{a: float(ap(y[m], e[idx][m])) for a, e in ens.items()}}
        out[key] = r
    return out


# ---------------------------------------------------------------- main
def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--fp32", action="store_true", help="full-FP32 scoring (the sandbox's); separate cache and JSON")
    ap_.add_argument("--rescore", action="store_true")
    ap_.add_argument("--boot", type=int, default=C.N_BOOT)
    a = ap_.parse_args()
    S, cand, sent = arm_scores(a.fp32, a.rescore)
    st, sd = sent["test"], sent["dev"]
    y, ydev = st.label.to_numpy(), sd.label.to_numpy()
    res = {"benchmark": "BioRED BC8 test sentences with >= 1 candidate pair; label = co-mentions a related pair",
           "precision": "fp32" if a.fp32 else "tf32 (paper convention)",
           "n_sentences": int(len(y)), "positive_rate": float(y.mean()),
           "trivial_accept_all": prf(y, np.ones_like(y)), "trivial_ap_negative_class": float(1 - y.mean()), "arms": {}}
    ens, preds = {}, {}
    for arm in ARMS:
        thr = best_thr(ydev, np.mean(S[arm]["dev"], axis=0))
        res["arms"][arm] = arm_block(y, S[arm]["test"], thr)
        ens[arm] = np.mean(S[arm]["test"], axis=0)
        preds[arm] = (ens[arm] >= thr).astype(int)
    if a.fp32:
        sb = json.loads(SANDBOX_JSON.read_text())
        res["reproduction_of_sandbox"] = {arm: {k: res["arms"][arm][k] == sb[arm][k] for k in
                                                ("auprc_per_seed", "auprc_ensemble", "threshold_dev", "P", "R", "F1")}
                                          for arm in ARMS}
        C.dump(res, "biored_sentence_fp32.json")
        print(json.dumps(res["reproduction_of_sandbox"]))
        return
    pmid = st.pmid.to_numpy()
    res["bootstrap_auprc_sentences"] = C.boot_auprc(y, ens, PAIRS, B=a.boot)
    res["bootstrap_auprc_abstracts"] = C.boot_auprc(y, ens, PAIRS, clusters=pmid, B=a.boot)
    res["bootstrap_f1_sentences"] = C.boot_f1(y, preds, PAIRS, B=a.boot)
    res["bootstrap_f1_abstracts"] = C.boot_f1(y, preds, PAIRS, clusters=pmid, B=a.boot)
    res["mcnemar_exact"] = {f"{x} vs {z}": C.mcnemar_exact(preds[z], preds[x], y) for x, z in PAIRS}
    res["per_seed_auprc_differences"] = {f"{x} - {z}": [p - q for p, q in zip(res["arms"][x]["auprc_per_seed"],
                                                                            res["arms"][z]["auprc_per_seed"])]
                                         for x, z in PAIRS}
    strata = {"concepts_in_sentence": st.n_concepts.to_numpy() >= 3,
              "entities_in_candidate_pairs": st.n_cand_entities.to_numpy() >= 3,
              "candidate_pairs": st.n_pairs.to_numpy() >= 2}
    res["strata"] = {"note": "entities_in_candidate_pairs and candidate_pairs are reported separately but may be the "
                             "same partition; same_partition says whether they are on this split",
                     "same_partition": bool((strata["entities_in_candidate_pairs"] == strata["candidate_pairs"]).all())}
    for name, multi in strata.items():
        lo, hi = ("2", ">=3") if name != "candidate_pairs" else ("1", ">=2")
        res["strata"][name] = {lo: stratum(y, ens, preds, ~multi, pmid, a.boot),
                               hi: stratum(y, ens, preds, multi, pmid, a.boot),
                               "difference_in_differences": {
                                   f"{x} - {z}": interaction(y, ens, ~multi, multi, x, z, B=a.boot)
                                   for x, z in PAIRS[:2]}}
    # pair precision of a perfect sentence filter: positive candidates / all candidates of positive sentences
    tc = cand["test"]
    pos_sent = set(st.sent_id[st.label == 1])
    inpos = tc.sent_id.isin(pos_sent).to_numpy()
    k, n = int(tc.label.to_numpy()[inpos].sum()), int(inpos.sum())
    res["perfect_filter_pair_precision"] = {"positive_candidates": k, "candidates_in_positive_sentences": n,
                                            "precision": k / n, "wilson95": C.wilson(k, n),
                                            "by_candidate_pairs": {}}
    for lab, m in (("1", st.n_pairs.to_numpy() == 1), (">=2", st.n_pairs.to_numpy() >= 2)):
        ss = set(st.sent_id[(st.label.to_numpy() == 1) & m])
        mm = tc.sent_id.isin(ss).to_numpy()
        res["perfect_filter_pair_precision"]["by_candidate_pairs"][lab] = {
            "precision": float(tc.label.to_numpy()[mm].mean()), "candidates": int(mm.sum())}
    paper = json.loads((REPO / "results/paperA_v2/tables_biored.json").read_text())
    res["paper_pair_level_reference"] = {
        "source": "results/paperA_v2/tables_biored.json",
        **{arm: {k: paper[arm][k] for k in ("P", "R", "F1", "auprc_ensemble", "threshold_prespecified")}
           for arm in ("sentence", "pair")}}
    tr = pd.read_csv(SENT / "train.csv")            # what sentlab learned from (BioRED train+dev, BC8 protocol)
    res["sentlab_training_sentences"] = {
        "n": int(len(tr)), "positive_rate": float(tr.label.mean()),
        "share_with_ge2_candidate_pairs": float((tr.n_candidates >= 2).mean()),
        "positive_rate_ge2_pairs": float(tr.label[tr.n_candidates >= 2].mean()),
        "positive_rate_1_pair": float(tr.label[tr.n_candidates == 1].mean())}
    sem = C.OUT / "biored_label_semantics.json"
    res["label_semantics"] = {"file": str(sem.relative_to(REPO)) if sem.exists() else None}
    res["llm"] = llm_sentence_rows(tc, st, ens, preds, a.boot)
    if SANDBOX_JSON.exists():
        sb = json.loads(SANDBOX_JSON.read_text())
        res["difference_from_sandbox_fp32"] = {
            arm: {"auprc_mean": res["arms"][arm]["auprc_mean"] - sb[arm]["auprc_mean"],
                  "auprc_ensemble": res["arms"][arm]["auprc_ensemble"] - sb[arm]["auprc_ensemble"],
                  "threshold_dev_equal": res["arms"][arm]["threshold_dev"] == sb[arm]["threshold_dev"],
                  "F1": res["arms"][arm]["F1"] - sb[arm]["F1"]} for arm in ARMS}
    f = C.dump(res, "biored_sentence.json")
    for arm, r in res["arms"].items():
        print(f"{arm:9} AUPRC {r['auprc_mean']:.3f} ± {r['auprc_sd']:.3f} (ens {r['auprc_ensemble']:.3f}) | "
              f"thr {r['threshold_dev']:.2f} P {r['P']:.3f} R {r['R']:.3f} F1 {r['F1']:.3f}")
    for k, v in res["bootstrap_auprc_sentences"]["diff"].items():
        print(f"  {k}: {v['observed']:+.4f} CI {v['ci95'][0]:+.4f} {v['ci95'][1]:+.4f} "
              f"(abstract clusters {res['bootstrap_auprc_abstracts']['diff'][k]['ci95']})")
    print(f"-> {f}")


if __name__ == "__main__":
    main()
