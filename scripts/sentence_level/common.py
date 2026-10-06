"""Shared pieces of the sentence-level analysis (Paper A reframe, 2026-10-05).

Statistics: a weighted average precision that equals sklearn's `average_precision_score` and runs
vectorised over bootstrap replicates; paired bootstrap CIs for AUPRC and F1 differences (resampling
items, or clusters of items); the exact McNemar test; Wilson intervals; Cohen's kappa. Plus the
reader of the human SENTENCE / PAIR answers in the two blind annotation sheets, and a small cache of
per-seed model scores so that every script reads the same numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from sklearn.metrics import average_precision_score, cohen_kappa_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "results/paperA_v2/sentence_level"
CACHE = OUT / "scores"
N_BOOT, SEED = 10_000, 0

GOLD_SHEET = REPO / "data/evaluation/gold_review_2026-10-02_v2_eg_curated.xlsx"   # the user's completed review (02:32); the BLIND copy stays empty
GOLD_KEY = REPO / "results/paperA_rebuild_2026-09-28/gold_review_v2_KEY_do_not_open_before_review.csv"
SECOND_SHEET = REPO / "data/evaluation/second_annotation_2026-10-04_BLIND.xlsx"
SECOND_KEY = REPO / "data/evaluation/second_annotation_2026-10-04_KEY_rows_only.csv"


# ---------------------------------------------------------------- average precision, bootstrap
def ap_weighted(y: np.ndarray, s: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Average precision of scores `s` for labels `y` under each row of item weights `W` (B x n).

    With integer weights (bootstrap counts) this is sklearn's average_precision_score on the
    resampled items: tied scores form one threshold, AP = sum_k (R_k - R_{k-1}) P_k.
    """
    y = np.asarray(y, float); s = np.asarray(s, float); W = np.atleast_2d(np.asarray(W, float))
    order = np.argsort(-s, kind="mergesort")
    last = np.r_[np.flatnonzero(np.diff(s[order]) != 0), len(s) - 1]
    Wo, yo = W[:, order], y[order]
    tp = np.cumsum(Wo * yo, axis=1)[:, last]
    fp = np.cumsum(Wo * (1 - yo), axis=1)[:, last]
    P = tp / np.maximum(tp + fp, 1e-12)
    R = tp / np.maximum(tp[:, -1:], 1e-12)
    dR = np.diff(np.c_[np.zeros(len(R)), R], axis=1)
    return (dR * P).sum(1)


def boot_weights(n: int, B: int = N_BOOT, seed: int = SEED, clusters: np.ndarray | None = None,
                 chunk: int = 1000):
    """Yield (b x n) count matrices of a nonparametric bootstrap, in chunks.

    Items are resampled with replacement; with `clusters`, whole clusters are (each item gets the
    count of its cluster), for a cluster bootstrap.
    """
    rng = np.random.default_rng(seed)
    if clusters is None:
        for i in range(0, B, chunk):
            b = min(chunk, B - i)
            yield rng.multinomial(n, np.full(n, 1 / n), size=b).astype(float)
    else:
        cid, inv = np.unique(clusters, return_inverse=True)
        k = len(cid)
        for i in range(0, B, chunk):
            b = min(chunk, B - i)
            yield rng.multinomial(k, np.full(k, 1 / k), size=b).astype(float)[:, inv]


def ci(x: np.ndarray, alpha: float = 0.05) -> list[float]:
    """Percentile interval."""
    return [float(np.quantile(x, alpha / 2)), float(np.quantile(x, 1 - alpha / 2))]


def boot_auprc(y: np.ndarray, scores: dict[str, np.ndarray], pairs: list[tuple[str, str]],
               clusters: np.ndarray | None = None, B: int = N_BOOT, seed: int = SEED) -> dict:
    """Paired bootstrap of AUPRC: a 95% CI for each arm and for each difference a - b in `pairs`.

    All arms share the same resamples, so a difference is paired. A resample with no positive or no
    negative (possible only for small sets, e.g. a few human labels) makes AP degenerate; such
    replicates are dropped and counted (`degenerate_dropped`).
    """
    y = np.asarray(y)
    vals = {a: [] for a in scores}
    keep = []
    for W in boot_weights(len(y), B, seed, clusters):
        keep.append(((W @ y) > 0) & ((W @ (1 - y)) > 0))
        for a, s in scores.items():
            vals[a].append(ap_weighted(y, s, W))
    keep = np.concatenate(keep)
    vals = {a: np.concatenate(v)[keep] for a, v in vals.items()}
    out = {"B": B, "seed": seed, "resample": "clusters" if clusters is not None else "items",
           "degenerate_dropped": int((~keep).sum()),
           "auprc_ci95": {a: ci(v) for a, v in vals.items()}, "diff": {}}
    for a, b in pairs:
        d = vals[a] - vals[b]
        obs = float(average_precision_score(y, scores[a]) - average_precision_score(y, scores[b]))
        out["diff"][f"{a} - {b}"] = {"observed": obs, "ci95": ci(d), "p_le_0": float((d <= 0).mean()),
                                     "p_ge_0": float((d >= 0).mean())}
    return out


def _f1_w(y, p, W):
    tp = W @ (p * y); fp = W @ (p * (1 - y)); fn = W @ ((1 - p) * y)
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)


def boot_f1(y: np.ndarray, preds: dict[str, np.ndarray], pairs: list[tuple[str, str]],
            clusters: np.ndarray | None = None, B: int = N_BOOT, seed: int = SEED) -> dict:
    """Paired bootstrap of F1 at fixed predictions (thresholds are not refitted per replicate)."""
    y = np.asarray(y, float)
    P = {a: np.asarray(p, float) for a, p in preds.items()}
    vals = {a: [] for a in P}
    for W in boot_weights(len(y), B, seed, clusters):
        for a, p in P.items():
            vals[a].append(_f1_w(y, p, W))
    vals = {a: np.concatenate(v) for a, v in vals.items()}
    out = {"B": B, "seed": seed, "f1_ci95": {a: ci(v) for a, v in vals.items()}, "diff": {}}
    for a, b in pairs:
        d = vals[a] - vals[b]
        out["diff"][f"{a} - {b}"] = {"observed": float(_f1_w(y, P[a], np.ones((1, len(y))))[0]
                                                       - _f1_w(y, P[b], np.ones((1, len(y))))[0]),
                                     "ci95": ci(d)}
    return out


def mcnemar_exact(a_pred: np.ndarray, b_pred: np.ndarray, y: np.ndarray) -> dict:
    """Exact (binomial) McNemar test of b against a: p, b fixes (a wrong, b right), b breaks."""
    ca, cb = np.asarray(a_pred) == y, np.asarray(b_pred) == y
    fix, brk = int((~ca & cb).sum()), int((ca & ~cb).sum())
    p = 1.0 if fix + brk == 0 else float(binomtest(min(fix, brk), fix + brk, 0.5).pvalue)
    return {"p_exact": p, "fixes": fix, "breaks": brk}


def wilson(k: int, n: int) -> list[float]:
    """Wilson 95% interval for k of n."""
    if n == 0:
        return [float("nan"), float("nan")]
    z = norm.ppf(0.975); p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def kappa(a, b) -> float | None:
    """Cohen's kappa, None when undefined (fewer than two items or a single class in both)."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(set(a) | set(b)) < 2:
        return None
    return float(cohen_kappa_score(a, b))


def paper_comparison_llm(name: str) -> bool:
    """True for the paper's zero-shot comparison models: the Qwen3 (3.0) family on the system Ollama
    and Qwen3.5-122B. Runs on the newer server (-v035), Qwen3.8 and MedGemma are extra, not comparisons."""
    return name == "qwen3.5-122b" or (name.startswith("qwen3-") and not name.endswith("-v035"))


# ---------------------------------------------------------------- per-seed score cache
def cached(name: str, fn, rescore: bool = False) -> np.ndarray:
    """Scores saved under results/paperA_v2/sentence_level/scores/<name>.npy, computed once.

    Model scoring on the GPU depends slightly (~1e-6) on batch composition; reading cached scores
    keeps every script and every re-run on identical numbers. `rescore` recomputes (and keeps the
    previous file as <name>.npy.bak).
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}.npy"
    if f.exists() and not rescore:
        return np.load(f)
    if f.exists():
        f.replace(f.with_suffix(".npy.bak"))
    s = np.asarray(fn(), float)
    np.save(f, s)
    return s


def model_key(prefix: str, md: Path) -> str:
    """Cache name for one checkpoint's scores: the model path plus a hash of its student_config.json,
    so a retrained or different checkpoint never reads another one's scores."""
    import hashlib
    cfg = Path(md) / "student_config.json"
    h = hashlib.sha1(cfg.read_bytes()).hexdigest()[:8] if cfg.exists() else "nocfg"
    rel = str(Path(md).resolve().relative_to(REPO)).replace("/", "__")
    return f"{prefix}__{rel}__{h}"


# ---------------------------------------------------------------- human annotation sheets
def _sheet(path: Path, sheet: str) -> pd.DataFrame:
    import openpyxl
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True)[sheet]
    rows = list(ws.iter_rows(values_only=True))
    hdr = [str(h) for h in rows[0]]
    d = pd.DataFrame([r[: len(hdr)] for r in rows[1:]], columns=hdr)
    scol = next(c for c in hdr if c.upper().startswith("SENTENCE"))
    pcol = next(c for c in hdr if c.upper().startswith("PAIR"))

    def norm_(x):
        x = "" if x is None else str(x).strip().upper()
        return x if x else None
    return pd.DataFrame({"item": pd.to_numeric(d["#"], errors="coerce").astype("Int64"),
                         "SENTENCE": d[scol].map(norm_), "PAIR": d[pcol].map(norm_)})


def human_answers() -> pd.DataFrame:
    """Every filled human answer, mapped to the 437-row clean benchmark (column `bench_row`).

    The 437-row sheet maps through its rows-only key (item -> bench_row, an index into the 437
    rows); its SKIP rows (the gold-review items) are dropped. The 22-item sheet maps through the
    sealed key's `item` and `bench_row` columns only (an index into the 449-row
    unified_test_set.csv), converted to the 437 rows; its items outside the 437 are dropped. Labels
    in the sealed key are never read here. Answers are YES / NO / UNSURE / None (empty).
    """
    from eval.core import clean_mask
    orig = np.flatnonzero(clean_mask())
    to437 = {int(o): i for i, o in enumerate(orig)}
    a = _sheet(SECOND_SHEET, "annotate").merge(pd.read_csv(SECOND_KEY)[["item", "bench_row"]], on="item")
    skip = a.SENTENCE.fillna("").str.startswith("SKIP") | a.PAIR.fillna("").str.startswith("SKIP")
    a = a[~skip].assign(sheet="second_annotation")
    g = _sheet(GOLD_SHEET, "review").merge(pd.read_csv(GOLD_KEY, usecols=["item", "bench_row"]), on="item")
    g = g.assign(bench_row=g.bench_row.map(lambda b: to437.get(int(b))), sheet="gold_review")
    g = g[g.bench_row.notna()]
    out = pd.concat([a, g], ignore_index=True)
    out["bench_row"] = out.bench_row.astype(int)
    return out[["sheet", "item", "bench_row", "SENTENCE", "PAIR"]]


def gold_review_rows() -> set[int]:
    """The 437-row indices of the 22-item gold-review sheet's items (from the key's `bench_row` only)."""
    from eval.core import clean_mask
    to437 = {int(o): i for i, o in enumerate(np.flatnonzero(clean_mask()))}
    k = pd.read_csv(GOLD_KEY, usecols=["item", "bench_row"])
    return {to437[int(b)] for b in k.bench_row if int(b) in to437}


def gold_review_complete(h: pd.DataFrame | None = None) -> bool:
    """True iff all 22 gold-review items have both a SENTENCE and a PAIR answer."""
    g = _sheet(GOLD_SHEET, "review") if h is None else h[h.sheet == "gold_review"]
    return len(g) == 22 and bool(g.SENTENCE.notna().all() and g.PAIR.notna().all())


def human_labels(col: str, sheet: str = "second_annotation") -> tuple[pd.Series, dict]:
    """bench_row -> 1 (YES) / 0 (NO) for one question (`SENTENCE` or `PAIR`) from ONE sheet.

    `second_annotation` (the 437-row sheet, block-balanced random order) is the only source of
    benchmark results. `gold_review` (22 items, 11 of them selected by model-vs-gold disagreement,
    answered by adjudicating a model's pre-filled sheet: not blind, not random) is a spot check of
    the silver labels only. UNSURE answers are dropped (the user's decision). Until the gold-review
    sheet is complete, nothing is read from it, and every row belonging to a gold-review item is also
    left out of the second sheet (the sealed review stays blind: no aggregate may expose the current
    gold of those items).
    """
    h = human_answers()
    complete = gold_review_complete(h)
    h = h[h.sheet == sheet]
    if not complete:
        if sheet == "gold_review":
            h = h.iloc[:0]
        else:
            h = h[~h.bench_row.isin(gold_review_rows())]
    h = h[h[col].notna()]
    info = {"question": col, "sheet": sheet, "gold_review_complete": complete, "answers": int(len(h)),
            "unsure_dropped": int((h[col] == "UNSURE").sum()),
            "other_dropped": int((~h[col].isin(["YES", "NO", "UNSURE"])).sum())}
    h = h[h[col].isin(["YES", "NO"])]
    assert h.bench_row.is_unique, f"{sheet}: a benchmark row answered twice"
    lab = pd.Series((h[col] == "YES").astype(int).to_numpy(), index=h.bench_row.to_numpy(), dtype=int).sort_index()
    info["labelled_rows"] = int(len(lab))
    return lab, info


def answer_counts(h: pd.DataFrame) -> dict:
    """Filled SENTENCE / PAIR answers per sheet (SKIP rows already removed)."""
    out = {}
    for sh, g in h.groupby("sheet"):
        out[sh] = {"rows": int(len(g)),
                   "SENTENCE": {k: int(v) for k, v in g.SENTENCE.fillna("(empty)").value_counts().items()},
                   "PAIR": {k: int(v) for k, v in g.PAIR.fillna("(empty)").value_counts().items()}}
    return out


def dump(obj: dict, name: str) -> Path:
    """Write a result JSON under results/paperA_v2/sentence_level/, keeping a dated backup of the old one."""
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / name
    if f.exists():
        import time
        f.replace(f.with_name(f"{f.name}.bak.{time.strftime('%Y%m%d-%H%M%S')}"))
    f.write_text(json.dumps(obj, indent=2, default=float))
    return f
