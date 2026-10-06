#!/usr/bin/env python3
"""The SENTENCE-level question on the 437-passage biodiversity benchmark (Paper A reframe).

Question: does the passage describe any biotic interaction? The benchmark's gold is pair-level, so
until the user's SENTENCE answers exist the sentence labels are SILVER, in three definitions
(a pair-positive candidate always makes its passage positive):
  majority3    else the majority of three local LLMs' answers to the sentence question
               (Qwen3-32B = the corpus teacher, Qwen3.5-122B, Qwen3.8-27B); the sandbox's definition
  nonteacher2  else the two non-teacher LLMs: both YES -> 1, both NO -> 0, disagreement -> dropped
  q122b        else Qwen3.5-122B alone
LLMs are never scored against these labels (circular); they are scored against human labels only
(`--human`).

Sentence scores (BiomedBERT-base, 3 seeds; ensemble = mean over seeds):
  sentence      the paper's sentence arm (passage only; trained on candidate rows)
  sentlab       passage only, trained on the teacher's answer to the sentence question, one row per
                passage (`--sentlab`; models/sentence_level/biodiv_sentlab_s{k}): the fair competitor
  pair-max      the paper's pair arm, max over every pair of TaxoNERD taxon mentions plus the
                candidate pair (per seed, then averaged), each pair order-free
  pair-own      the paper's pair arm on the candidate pair only
  pair_matched-max / -own   the matched pair arm, if a partial labelling made A1 train it
Operating points: block-held-out thresholds (fitted on two source blocks, applied to the third), as
in the paper. Statistics: paired bootstrap over passages (10,000 replicates, seed 0) for AUPRC and
F1 differences, exact McNemar on the block-held-out decisions.

Outputs (results/paperA_v2/sentence_level/): biodiv_sentence.json (default), biodiv_sentlab.json
(`--sentlab`), biodiv_human.json (`--human`). Per-seed scores are cached under scores/.

Usage
  python3 scripts/sentence_level/biodiv_sentence.py              # silver labels, A4
  python3 scripts/sentence_level/biodiv_sentence.py --sentlab    # adds the sentence-label students (A1)
  python3 scripts/sentence_level/biodiv_sentence.py --human      # the user's SENTENCE answers (A5)
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import REPO  # noqa: E402
from eval.core import clean_benchmark  # noqa: E402
from paperA_tables import block_held_out, order_free, per_block_pred, prf  # noqa: E402

LLM_DIR = REPO / "results/paperA_v2/llm"
LLM_SL_DIR = C.OUT / "llm"
SILVER_LLMS = {"teacher": "qwen3-32b", "large": "qwen3.5-122b", "newer": "qwen3.8-27b-v035"}
MENTIONS = C.OUT / "taxonerd_mentions.json"
SANDBOX_MENTIONS = REPO / "sandbox/sentence_level/results/taxonerd_mentions.json"
SANDBOX_JSON = REPO / "sandbox/sentence_level/results/biodiv_sentence_silver.json"
SEEDS = (1, 2, 3)
MODELS = {"sentence": "models/sentence_baseline/xenc_s{}", "pair": "models/pair_baseline/xenc_s{}",
          "sentlab": "models/sentence_level/biodiv_sentlab_s{}",
          "pair_matched": "models/sentence_level/biodiv_pair_matched_s{}"}


# ---------------------------------------------------------------- data
def benchmark() -> pd.DataFrame:
    d = clean_benchmark()
    assert len(d) == 437 and int(d.label.sum()) == 246, "clean benchmark changed"
    return d


def llm_file(d: pd.DataFrame, name: str, form: str, base: Path = LLM_DIR) -> pd.DataFrame | None:
    """One zero-shot LLM output aligned with the 437 rows (asserted), or None if absent."""
    f = base / f"biodiv_{name}_{form}.csv"
    if not f.exists():
        return None
    o = pd.read_csv(f)
    assert len(o) == len(d) and (o.label.to_numpy() == d.label.to_numpy()).all() \
        and (o.sentence.astype(str).to_numpy() == d.sentence.astype(str).to_numpy()).all(), f"{f.name} misaligned"
    return o


def silver(d: pd.DataFrame) -> dict[str, np.ndarray]:
    """The three silver sentence labels; -1 marks a dropped passage."""
    v = {k: llm_file(d, m, "sentence").verdict.to_numpy().astype(int) for k, m in SILVER_LLMS.items()}
    pos = d.label.to_numpy() == 1
    maj = (np.mean([v["teacher"], v["large"], v["newer"]], axis=0) >= 0.5).astype(int)
    two = np.where(v["large"] == v["newer"], v["large"], -1)
    return {"majority3": np.where(pos, 1, maj), "nonteacher2": np.where(pos, 1, two),
            "q122b": np.where(pos, 1, v["large"])}


def mentions(d: pd.DataFrame) -> list[list[str]]:
    """TaxoNERD taxon mentions per passage (unique, case-insensitive), cached.

    The cache is the sandbox's (`en_ner_eco_biobert`, CPU), copied once next to the results; it
    is recomputed only if neither copy exists.
    """
    if not MENTIONS.exists():
        C.OUT.mkdir(parents=True, exist_ok=True)
        if SANDBOX_MENTIONS.exists():
            shutil.copyfile(SANDBOX_MENTIONS, MENTIONS)
        else:
            from taxonerd import TaxoNERD
            t = TaxoNERD(prefer_gpu=False); t.load("en_ner_eco_biobert")
            out = []
            for s in d.sentence:
                df = t.find_in_text(str(s)); seen, ms = set(), []
                for m in (df["text"].tolist() if len(df) else []):
                    k = m.strip().lower()
                    if k and k not in seen:
                        seen.add(k); ms.append(m.strip())
                out.append(ms)
            MENTIONS.write_text(json.dumps(out))
    ms = json.loads(MENTIONS.read_text())
    assert len(ms) == len(d)
    return ms


def enum_pairs(d: pd.DataFrame) -> pd.DataFrame:
    """Every pair scored by pair-max: the candidate, plus each pair of TaxoNERD mentions.

    The pair set is the sandbox's ({candidate} | sorted mention pairs); rows are put in a fixed
    order (candidate first, then sorted) so that GPU batches, and hence scores, are reproducible.
    """
    rows = []
    for i, (s, a1, a2, ms) in enumerate(zip(d.sentence, d.species1, d.species2, mentions(d))):
        pairs = {(a1, a2)} | {tuple(sorted(p)) for p in itertools.combinations(ms, 2)}
        rest = sorted(p for p in pairs if p != (a1, a2))
        rows += [{"i": i, "sentence": s, "species1": p[0], "species2": p[1], "relation": "",
                  "is_candidate": int(p == (a1, a2))} for p in [(a1, a2)] + rest]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- scores
def arm_scores(d: pd.DataFrame, P: pd.DataFrame, sentlab: bool, rescore: bool = False) -> dict:
    """Per-seed and ensemble sentence scores of every arm (cached per seed)."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    md = lambda arm, k: REPO / MODELS[arm].format(k)  # noqa: E731
    exists = lambda arm: all((md(arm, k) / "student_config.json").exists() for k in SEEDS)  # noqa: E731
    S, check = {}, {}

    def per_passage_max(arm: str, k: int) -> np.ndarray:
        e = C.cached(C.model_key("biodiv_enum", md(arm, k)), lambda: order_free(md(arm, k), P, dev), rescore)
        assert len(e) == len(P)
        return P.assign(s=e).groupby("i").s.max().reindex(range(len(d))).to_numpy()

    def own(arm: str, k: int) -> np.ndarray:
        return C.cached(C.model_key("biodiv_own", md(arm, k)), lambda: order_free(md(arm, k), d, dev), rescore)

    # the paper's arms: ensembles from the paper's saved scores, per-seed scores re-computed
    for arm, name in (("sentence", "sentence"), ("pair", "pair-own")):
        per = [own(arm, k) for k in SEEDS]
        ens = np.load(REPO / f"results/paperA_v2/S_biodiv_{arm}.npy")
        check[name] = float(np.abs(np.mean(per, axis=0) - ens).max())
        S[name] = {"per_seed": per, "ens": ens}
    per = [per_passage_max("pair", k) for k in SEEDS]
    S["pair-max"] = {"per_seed": per, "ens": np.mean(per, axis=0)}
    if sentlab:
        if exists("sentlab"):
            per = [own("sentlab", k) for k in SEEDS]
            S["sentlab"] = {"per_seed": per, "ens": np.mean(per, axis=0)}
        else:
            print("sentlab models missing: arm skipped", file=sys.stderr)
        if exists("pair_matched"):
            per = [per_passage_max("pair_matched", k) for k in SEEDS]
            S["pair_matched-max"] = {"per_seed": per, "ens": np.mean(per, axis=0)}
            per = [own("pair_matched", k) for k in SEEDS]
            S["pair_matched-own"] = {"per_seed": per, "ens": np.mean(per, axis=0)}
    return S, {"max_abs_diff_mean_of_rescored_seeds_vs_paper_ensemble": check}


# ---------------------------------------------------------------- evaluation
def compare_pairs(arms: list[str]) -> list[tuple[str, str]]:
    """The differences reported, a - b, among the arms present."""
    want = [("pair-max", "sentence"), ("pair-own", "sentence"), ("pair-max", "pair-own"),
            ("pair-max", "sentlab"), ("pair-own", "sentlab"), ("sentlab", "sentence"),
            ("pair_matched-max", "sentlab"), ("pair_matched-own", "sentlab")]
    return [(a, b) for a, b in want if a in arms and b in arms]


def strata(y: np.ndarray, S: dict, n_pairs: np.ndarray, B: int = C.N_BOOT) -> dict:
    """AUPRC by the number of pairs pair-max scores (1 = the candidate only, vs >= 2): the analogue
    of BioRED's candidate-pair strata. Threshold-free; paired bootstrap within each stratum."""
    m = y >= 0
    out = {}
    for lab, mm in (("1_pair", m & (n_pairs == 1)), ("ge2_pairs", m & (n_pairs >= 2))):
        yy = y[mm]
        r = {"n": int(mm.sum()), "positive_rate": float(yy.mean())}
        if len(set(yy)) == 2:
            r["auprc"] = {a: float(ap(yy, s["ens"][mm])) for a, s in S.items()}
            r["bootstrap_auprc"] = C.boot_auprc(yy, {a: s["ens"][mm] for a, s in S.items()}, compare_pairs(list(S)), B=B)
        out[lab] = r
    return out


def decomposition(y: np.ndarray, S: dict, y_pair: np.ndarray, B: int = C.N_BOOT) -> dict:
    """Where an arm's sentence-level AUPRC comes from, split by what decides the label.

    pair_negative_only   the passages whose candidate is pair-negative (gold): their silver label is
                         an LLM's answer to the sentence question, i.e. the prompt the sentence-label
                         students (sentlab) imitate, so this part favours sentlab by construction
    gold_pos_vs_silver_neg  the gold pair-positive passages (positive by gold) against the silver
                         negatives: the positives do not depend on any LLM
    """
    m = y >= 0
    out = {}
    for lab, mm, yy in (("pair_negative_only", m & (y_pair == 0), y),
                        ("gold_pos_vs_silver_neg", m & ((y_pair == 1) | (y == 0)), y)):
        yv = yy[mm]
        r = {"n": int(mm.sum()), "positives": int(yv.sum())}
        if len(set(yv)) == 2:
            r["auprc"] = {a: float(ap(yv, s["ens"][mm])) for a, s in S.items()}
            r["bootstrap_auprc"] = C.boot_auprc(yv, {a: s["ens"][mm] for a, s in S.items()}, compare_pairs(list(S)), B=B)
        out[lab] = r
    return out


def evaluate(y: np.ndarray, S: dict, blocks: np.ndarray, y_pair: np.ndarray, B: int = C.N_BOOT) -> dict:
    """Every arm against one label vector (-1 = dropped): AUPRC, P/R/F1 at block-held-out
    thresholds, paired bootstrap CIs, exact McNemar, arm order, and the perfect-filter ceiling."""
    m = y >= 0
    yy, bb = y[m], blocks[m]
    out = {"n": int(m.sum()), "dropped": int((~m).sum()), "positives": int(yy.sum()),
           "positive_rate": float(yy.mean()), "arms": {},
           "trivial_accept_all": prf(yy, np.ones_like(yy)),
           "trivial_ap_negative_class": float(1 - yy.mean())}
    preds = {}
    for arm, s in S.items():
        e = s["ens"][m]
        pred, thr = block_held_out(yy, e, bb)
        preds[arm] = pred
        per = [float(ap(yy, p[m])) for p in s["per_seed"]]
        out["arms"][arm] = {"auprc": float(ap(yy, e)), "auprc_per_seed": per, "auprc_seed_mean": float(np.mean(per)),
                            "auprc_seed_sd": float(np.std(per, ddof=1)), **prf(yy, pred), "thresholds": thr,
                            "secondary": {"auroc": float(roc_auc_score(yy, e)),
                                          "ap_negative_class": float(ap(1 - yy, -e))}}
    pairs = compare_pairs(list(S))
    ens = {a: s["ens"][m] for a, s in S.items()}
    out["bootstrap_auprc"] = C.boot_auprc(yy, ens, pairs, B=B)
    out["bootstrap_f1"] = C.boot_f1(yy, preds, pairs, B=B)
    out["mcnemar_exact"] = {f"{a} vs {b}": C.mcnemar_exact(preds[b], preds[a], yy) for a, b in pairs}
    out["order_by_auprc"] = sorted(S, key=lambda a: -out["arms"][a]["auprc"])
    out["order_by_F1"] = sorted(S, key=lambda a: -out["arms"][a]["F1"])
    # a perfect sentence filter accepts every sentence-positive candidate: its pair-level precision
    k, n = int(y_pair[m][yy == 1].sum()), int((yy == 1).sum())
    yp, ys = y_pair[m], yy
    rat = [(W @ (yp * ys)) / np.maximum(W @ ys, 1) for W in C.boot_weights(len(ys), B, C.SEED)]
    out["perfect_filter_pair_precision"] = {"pair_positive": k, "sentence_positive": n, "precision": k / n,
                                            "wilson95": C.wilson(k, n), "bootstrap95": C.ci(np.concatenate(rat))}
    return out


def pair_gold_view(y_pair: np.ndarray, S: dict, blocks: np.ndarray) -> dict:
    """Each sentence-level score used as a PAIR filter for the benchmark's candidate (pair gold)."""
    out = {}
    for arm, s in S.items():
        pred, thr = block_held_out(y_pair, s["ens"], blocks)
        out[arm] = {"auprc": float(ap(y_pair, s["ens"])),
                    "auprc_per_seed": [float(ap(y_pair, p)) for p in s["per_seed"]], **prf(y_pair, pred),
                    "thresholds": thr}
    return out


def reproduction(res: dict) -> dict:
    """The sandbox's numbers (CONTEXT.md §3) against this run, majority3 labels."""
    if not SANDBOX_JSON.exists():
        return {"sandbox_json": None}
    sb = json.loads(SANDBOX_JSON.read_text())
    r = res["by_silver"]["majority3"]
    diff = {}
    for arm in ("sentence", "pair-max", "pair-own"):
        for key in ("auprc", "P", "R", "F1", "accepts"):
            diff[f"{arm}.{key}"] = float(r["arms"][arm][key] - sb[arm][key])
        diff[f"{arm}.thresholds_equal"] = r["arms"][arm]["thresholds"] == sb[arm]["thresholds"]
    diff["pairs_scored"] = res["enumeration"]["pairs_scored"] - sb["pairs_scored"]
    diff["pair_max_auprc_per_seed"] = [a - b for a, b in zip(r["arms"]["pair-max"]["auprc_per_seed"],
                                                              sb["pair_max_auprc_per_seed"])]
    diff["sentence_positive_rate"] = r["positive_rate"] - sb["sentence_positive_rate"]
    diff["ceiling"] = r["perfect_filter_pair_precision"]["precision"] - \
        sb["precision_ceiling_of_a_perfect_sentence_filter_on_pair_gold"]
    worst = max(abs(v) for v in pd.Series(diff).explode().tolist() if isinstance(v, (int, float)) and not isinstance(v, bool))
    return {"sandbox_json": str(SANDBOX_JSON.relative_to(REPO)), "differences": diff,
            "max_abs_difference": worst,
            "thresholds_identical": all(v for k, v in diff.items() if k.endswith("thresholds_equal"))}


# ---------------------------------------------------------------- human labels
def human(d: pd.DataFrame, S: dict, P: pd.DataFrame, sil: dict) -> dict:
    """Every arm and the LLM rows on the user's SENTENCE answers from the 437-row sheet (the only
    benchmark source), agreement with silver and gold; plus the 22-item gold review as a spot check
    of the silver labels only (`gold_review_spot_check`)."""
    ys, info = C.human_labels("SENTENCE", "second_annotation")
    yp_h, pinfo = C.human_labels("PAIR", "second_annotation")
    out = {"benchmark_source": "second_annotation_2026-10-04 sheet: the 437 rows in a block-balanced random order; "
                               "its 20 SKIP rows (gold-review items) never get answers there",
           "sentence_labels": info, "pair_labels": pinfo, "answer_counts": C.answer_counts(C.human_answers())}
    if len(ys) == 0:
        out["status"] = "no human labels yet (437-row sheet)"
    else:
        idx = ys.index.to_numpy(); y = ys.to_numpy(); blocks = d.source.to_numpy()
        two = len(set(y)) == 2
        out["status"] = f"{len(y)} passages with a human SENTENCE label (437-row sheet)"
        out["n"], out["positives"] = int(len(y)), int(y.sum())
        # thresholds: the block-held-out thresholds fitted on the majority3 silver labels over all
        # 437 passages (pre-specified, never fitted on the human labels)
        ysil = sil["majority3"]
        out["arms"] = {}
        for arm, s in S.items():
            _, thr = block_held_out(ysil, s["ens"], blocks)
            pred = per_block_pred(s["ens"], blocks, thr)[idx]
            out["arms"][arm] = {"auprc": float(ap(y, s["ens"][idx])) if two else None, **prf(y, pred),
                                "thresholds_from_silver_majority3": thr}
        if two and len(y) >= 20:
            pairs = compare_pairs(list(S))
            out["bootstrap_auprc"] = C.boot_auprc(y, {a: s["ens"][idx] for a, s in S.items()}, pairs)
        out["llm"] = llm_rows_human(d, P, idx, y)
        out["human_vs_silver"] = agreement_with_silver(sil, idx, y)
    if len(yp_h):
        idx = yp_h.index.to_numpy(); y = yp_h.to_numpy(); g = d.label.to_numpy()[idx]
        out["pair_answers_vs_benchmark_pair_gold"] = {
            "n": int(len(y)), "agreement": float((y == g).mean()), "kappa": C.kappa(y, g),
            "human_yes_gold_no": int(((y == 1) & (g == 0)).sum()), "human_no_gold_yes": int(((y == 0) & (g == 1)).sum()),
            "note": "agreement only; gold labels change only by the user's decision"}
        out["sentence_vs_pair_answers"] = sentence_vs_pair(ys, yp_h)
    out["gold_review_spot_check"] = spot_check(d, sil, ys)
    return out


def agreement_with_silver(sil: dict, idx: np.ndarray, y: np.ndarray) -> dict:
    """Human SENTENCE answers against each silver definition (dropped silver rows excluded)."""
    out = {}
    for k, v in sil.items():
        m = v[idx] >= 0
        out[k] = {"n": int(m.sum()), "agreement": float((v[idx][m] == y[m]).mean()) if m.any() else None,
                  "kappa": C.kappa(v[idx][m], y[m]),
                  "human_no_silver_yes": int(((y[m] == 0) & (v[idx][m] == 1)).sum()),
                  "human_yes_silver_no": int(((y[m] == 1) & (v[idx][m] == 0)).sum())}
    return out


def sentence_vs_pair(ys: pd.Series, yp: pd.Series) -> dict:
    both = ys.index.intersection(yp.index)
    a, b = ys[both].to_numpy(), yp[both].to_numpy()
    return {"n": int(len(both)), "sentence_yes_pair_no": int(((a == 1) & (b == 0)).sum()),
            "sentence_no_pair_yes": int(((a == 0) & (b == 1)).sum()),
            "both_yes": int(((a == 1) & (b == 1)).sum()), "both_no": int(((a == 0) & (b == 0)).sum())}


def spot_check(d: pd.DataFrame, sil: dict, ys_second: pd.Series) -> dict:
    """The 22-item gold review as a spot check of the silver sentence labels (never a benchmark)."""
    gs, ginfo = C.human_labels("SENTENCE", "gold_review")
    gp, gpinfo = C.human_labels("PAIR", "gold_review")
    out = {"use": "spot check of the silver sentence labels only, NOT a benchmark result: 22 items, 11 of them "
                  "selected by model-vs-gold disagreement; answered by adjudicating a model's pre-filled sheet "
                  "(not blind). Item-level comparison with the gold: results/reframe_2026-10-05/gold_review_22.md",
           "sheet": str(C.GOLD_SHEET.relative_to(REPO)), "sentence_labels": ginfo, "pair_labels": gpinfo}
    if len(gs) == 0:
        out["status"] = "gold-review sheet incomplete: not read"
        return out
    idx, y = gs.index.to_numpy(), gs.to_numpy()
    out["status"] = f"{len(y)} items with a SENTENCE YES/NO answer"
    out["sentence_vs_silver"] = agreement_with_silver(sil, idx, y)
    out["silver_yes_forced_by_pair_gold"] = int((d.label.to_numpy()[idx] == 1).sum())
    if len(gp):
        pi, py = gp.index.to_numpy(), gp.to_numpy(); g = d.label.to_numpy()[pi]
        out["pair_answers_vs_current_pair_gold"] = {"n": int(len(py)), "agreement": float((py == g).mean()),
                                                    "kappa": C.kappa(py, g)}
        out["sentence_vs_pair_answers"] = sentence_vs_pair(gs, gp)
    rr = gs.index.intersection(ys_second.index)
    out["rows_answered_in_both_sheets"] = {"n": int(len(rr)),
                                           "same_sentence_answer": int((gs[rr] == ys_second[rr]).sum())}
    return out


def llm_rows_human(d: pd.DataFrame, P: pd.DataFrame, idx: np.ndarray, y: np.ndarray) -> dict:
    """Zero-shot LLMs at the sentence level on the human-labelled passages: the sentence question,
    pair-own (the candidate pair) and, where the enumerated pairs were scored, pair-max."""
    out = {}
    two = len(set(y)) == 2
    for f in sorted(LLM_DIR.glob("biodiv_*_sentence.csv")):
        name = f.stem[len("biodiv_"):-len("_sentence")]
        r = {"paper_comparison_model": C.paper_comparison_llm(name)}
        for form in ("sentence", "pair"):
            o = llm_file(d, name, form)
            if o is not None:
                r["sentence" if form == "sentence" else "pair-own"] = {
                    "auprc": float(ap(y, o.p_yes.to_numpy()[idx])) if two else None,
                    "greedy": prf(y, o.verdict.to_numpy()[idx])}
        e = LLM_SL_DIR / f"biodiv_{name}_enum_pair.csv"
        if e.exists():
            o = pd.read_csv(e)
            if len(o) == len(P):
                mx = o.groupby("i").p_yes.max().reindex(range(len(d))).to_numpy()
                vx = o.groupby("i").verdict.max().reindex(range(len(d))).to_numpy()
                r["pair-max"] = {"auprc": float(ap(y, mx[idx])) if two else None, "greedy": prf(y, vx[idx])}
        out[name] = r
    return out


# ---------------------------------------------------------------- main
def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--sentlab", action="store_true", help="add the sentence-label students; write biodiv_sentlab.json")
    ap_.add_argument("--human", action="store_true", help="evaluate on the user's SENTENCE answers; write biodiv_human.json")
    ap_.add_argument("--rescore", action="store_true", help="recompute the cached per-seed scores")
    ap_.add_argument("--boot", type=int, default=C.N_BOOT)
    ap_.add_argument("--out", default=None, help="output JSON name under results/paperA_v2/sentence_level/ (default per mode)")
    ap_.add_argument("--sentlab-pattern", default=None, help=argparse.SUPPRESS)   # code-path tests only
    a = ap_.parse_args()
    if a.sentlab_pattern:
        MODELS["sentlab"] = a.sentlab_pattern
    d = benchmark()
    y_pair, blocks = d.label.to_numpy(), d.source.to_numpy()
    sil = silver(d)
    P = enum_pairs(d)
    with_sentlab = a.sentlab or a.human
    S, check = arm_scores(d, P, with_sentlab, a.rescore)
    if a.human:
        res = human(d, S, P, sil)
        f = C.dump(res, a.out or "biodiv_human.json")
        print(f"{res['status']} -> {f}")
        return
    per_passage = P.groupby("i").size()
    res = {"benchmark": "437-passage biodiversity benchmark (clean rows), sentence level",
           "labels": {"majority3": "pair-positive, else majority of Qwen3-32B, Qwen3.5-122B, Qwen3.8-27B sentence answers",
                      "nonteacher2": "pair-positive, else Qwen3.5-122B and Qwen3.8-27B agreeing; disagreements dropped",
                      "q122b": "pair-positive, else Qwen3.5-122B"},
           "silver_llm_files": {k: f"results/paperA_v2/llm/biodiv_{m}_sentence.csv" for k, m in SILVER_LLMS.items()},
           "arms": list(S), "models": {k: v.format("{1,2,3}") for k, v in MODELS.items()},
           "thresholds": "block-held-out (paperA_tables.block_held_out)",
           "pair_positive_rate": float(y_pair.mean()),
           "enumeration": {"pairs_scored": int(len(P)), "mean_pairs_per_passage": float(per_passage.mean()),
                           "median_pairs_per_passage": float(per_passage.median()),
                           "max_pairs_per_passage": int(per_passage.max()),
                           "passages_with_candidate_only": int((per_passage == 1).sum()),
                           "mentions_cache": str(MENTIONS.relative_to(REPO))},
           "score_consistency": check, "by_silver": {}}
    n_pairs = per_passage.reindex(range(len(d))).to_numpy()
    for k, y in sil.items():
        res["by_silver"][k] = evaluate(y, S, blocks, y_pair, a.boot)
        res["by_silver"][k]["strata_by_pairs_scored"] = strata(y, S, n_pairs, a.boot)
        res["by_silver"][k]["decomposition"] = decomposition(y, S, y_pair, a.boot)
    res["order_changes_across_silver"] = {
        "by_auprc": {k: v["order_by_auprc"] for k, v in res["by_silver"].items()},
        "by_F1": {k: v["order_by_F1"] for k, v in res["by_silver"].items()},
        "auprc_order_identical": len({tuple(v["order_by_auprc"]) for v in res["by_silver"].values()}) == 1,
        "F1_order_identical": len({tuple(v["order_by_F1"]) for v in res["by_silver"].values()}) == 1}
    res["silver_agreement"] = {f"{a_} vs {b_}": {"kappa_all": C.kappa(sil[a_][(sil[a_] >= 0) & (sil[b_] >= 0)],
                                                                      sil[b_][(sil[a_] >= 0) & (sil[b_] >= 0)])}
                               for a_, b_ in (("majority3", "q122b"), ("majority3", "nonteacher2"), ("nonteacher2", "q122b"))}
    res["llm_sentence_votes_on_pair_negative_passages"] = {
        k: int(llm_file(d, m, "sentence").verdict.to_numpy()[y_pair == 0].sum()) for k, m in SILVER_LLMS.items()}
    res["n_pair_negative"] = int((y_pair == 0).sum())
    res["pair_gold_view"] = pair_gold_view(y_pair, S, blocks)
    if not a.sentlab:
        res["reproduction_of_sandbox"] = reproduction(res)
    f = C.dump(res, a.out or ("biodiv_sentlab.json" if a.sentlab else "biodiv_sentence.json"))
    for k, r in res["by_silver"].items():
        print(f"[{k}] n={r['n']} pos={r['positive_rate']:.3f} " + " | ".join(
            f"{arm} AUPRC {v['auprc']:.3f} F1 {v['F1']:.3f}" for arm, v in r["arms"].items()))
    if "reproduction_of_sandbox" in res:
        print("reproduction of the sandbox: max |diff|", res["reproduction_of_sandbox"].get("max_abs_difference"),
              "thresholds identical", res["reproduction_of_sandbox"].get("thresholds_identical"))
    print(f"-> {f}")


if __name__ == "__main__":
    main()
