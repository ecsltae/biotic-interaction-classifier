#!/usr/bin/env python3
"""What the pair enumeration of "pair-max" costs on the 437-passage biodiversity benchmark.

pair-max turns pair verification into a sentence decision: every pair of taxon mentions that
TaxoNERD finds in a passage, plus the benchmark's own candidate pair, is scored by the
pair-conditioned cross-encoder (BiomedBERT-base, 3 seeds, both argument orders, max), and the
passage score is the max over its pairs. The "sentence" arm reads the passage alone (one forward
pass per passage per seed); "pair-own" scores only the candidate pair (2 passes per seed).

Parts (run separately so that each command stays short; each run merges its keys into the JSON):
  stats     enumeration statistics from the cached TaxoNERD mentions (the cache is only read)
  taxonerd  TaxoNERD extraction time per passage on a fixed 50-passage sample (seed 0): CPU, and
            GPU if spaCy can use it; plus a check that the sample's mentions equal the cached ones
  gpu       cross-encoder scoring time on the GPU, all 437 passages, 3 seeds: pair-max vs sentence
            vs pair-own; model load timed separately, one warm-up, two timed repetitions
  cpu       the same three arms on the CPU, 50-passage sample, one seed, extrapolated per passage
  check     the timed scorer against scripts/paperA_tables.py:order_free run now (pair-own, sentence)

The scorer below re-implements scripts/eval_unified.py:score (same input builder, tokenizer call,
max_len, batch 64, softmax) so that model loading can be kept out of the timed window. The check part
compares it with paperA_tables.order_free run at the same time, and the gpu part compares pair-max with the
sandbox's per-seed AUPRCs and pair-own / sentence with the saved S_biodiv_*.npy scores.

Output: results/paperA_v2/sentence_level/biodiv_enum_cost.json (an existing file is first copied to
<name>.bak.<HHMM>).

Usage
  python3 scripts/sentence_level/taxonerd_cost.py --parts stats,taxonerd
  CUDA_VISIBLE_DEVICES=0 python3 scripts/sentence_level/taxonerd_cost.py --parts gpu
  python3 scripts/sentence_level/taxonerd_cost.py --parts cpu --threads 8
  CUDA_VISIBLE_DEVICES=0 python3 scripts/sentence_level/taxonerd_cost.py --parts check
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from eval.core import clean_benchmark  # noqa: E402

OUT = REPO / "results/paperA_v2/sentence_level/biodiv_enum_cost.json"
MENTION_CACHE = REPO / "sandbox/sentence_level/results/taxonerd_mentions.json"
SILVER_JSON = REPO / "sandbox/sentence_level/results/biodiv_sentence_silver.json"
PAIR_DIR = "models/pair_baseline/xenc_s{}"
SENT_DIR = "models/sentence_baseline/xenc_s{}"
SEEDS = (1, 2, 3)
BS = 64
SAMPLE_N, SAMPLE_SEED = 50, 0
LLMS = ("qwen3-32b", "qwen3.5-122b", "qwen3.8-27b-v035")   # silver labels, as in biodiv_sentence.py
BINS = (("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-5", 4, 5), ("6-9", 6, 9), (">=10", 10, 10**9))


# ---------------------------------------------------------------- data
def load_data() -> tuple[pd.DataFrame, list[list[str]]]:
    """The 437 clean benchmark rows and the cached TaxoNERD mentions (read only)."""
    d = clean_benchmark()
    ments = json.loads(MENTION_CACHE.read_text())
    assert len(ments) == len(d) == 437, "mention cache not aligned with the benchmark"
    return d, ments


def passage_pairs(a1: str, a2: str, ms: list[str]) -> list[tuple[str, str]]:
    """Pairs scored for one passage, exactly as biodiv_sentence.py builds them (sorted for a fixed order)."""
    return sorted({(a1, a2)} | {tuple(sorted(p)) for p in itertools.combinations(ms, 2)})


def build_pairs(d: pd.DataFrame, ments: list[list[str]]) -> pd.DataFrame:
    """One row per (passage, pair), passages contiguous, relation empty (the pair format ignores it)."""
    rows = []
    for i, (s, a1, a2, ms) in enumerate(zip(d.sentence, d.species1, d.species2, ments)):
        rows += [{"i": i, "sentence": s, "species1": p[0], "species2": p[1], "relation": ""}
                 for p in passage_pairs(a1, a2, ms)]
    return pd.DataFrame(rows)


def sample_idx(n: int) -> np.ndarray:
    """Fixed random sample of passage indices (seed 0), sorted."""
    return np.sort(np.random.default_rng(SAMPLE_SEED).choice(n, SAMPLE_N, replace=False))


def dedupe(found: list[str]) -> list[str]:
    """Unique mention strings, case-insensitive, first spelling kept (as in biodiv_sentence.py)."""
    seen, out = set(), []
    for m in found:
        k = m.strip().lower()
        if k and k not in seen:
            seen.add(k); out.append(m.strip())
    return out


def summary(x: np.ndarray | list[float]) -> dict[str, float]:
    x = np.asarray(x, float)
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "max": float(x.max()), "total": float(x.sum())}


# ---------------------------------------------------------------- environment
def nvidia_smi() -> dict[str, Any]:
    """One snapshot of the GPU (name, memory, utilisation) and of the processes using it."""
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,driver_version",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        name, used, total, util, drv = [v.strip() for v in g.strip().splitlines()[0].split(",")]
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        apps = [dict(zip(("pid", "process", "mem_mib"), [v.strip() for v in ln.split(",")]))
                for ln in a.strip().splitlines() if ln.strip()]
        for x in apps:
            x["process"] = Path(x["process"]).name
        return {"time": dt.datetime.now().isoformat(timespec="seconds"), "gpu": name, "driver": drv,
                "mem_used_mib": int(used), "mem_total_mib": int(total), "util_pct": int(util), "processes": apps}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


class SmiPoller:
    """Polls nvidia-smi once a second while a timed section runs (utilisation includes this job)."""

    def __init__(self, every: float = 1.0) -> None:
        self.every, self.samples, self._stop = every, [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                g = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                   timeout=10).stdout
                mem, util = [int(v) for v in g.strip().splitlines()[0].split(",")]
                self.samples.append((mem, util))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.every)

    def __enter__(self) -> "SmiPoller":
        self._t.start(); return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set(); self._t.join()

    def stats(self) -> dict[str, Any]:
        if not self.samples:
            return {"n_samples": 0}
        m, u = np.array(self.samples).T
        return {"n_samples": int(len(m)), "mem_used_mib_min": int(m.min()), "mem_used_mib_max": int(m.max()),
                "util_pct_mean": float(u.mean()), "util_pct_min": int(u.min()), "util_pct_max": int(u.max())}


def environment() -> dict[str, Any]:
    import torch
    import transformers
    cpu = next((ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                if ln.startswith("model name")), platform.processor())
    env = {"date_time": dt.datetime.now().isoformat(timespec="seconds"), "python": platform.python_version(),
           "torch": torch.__version__, "transformers": transformers.__version__, "cpu_model": cpu,
           "cpu_logical_cores": os.cpu_count(), "load_average_1_5_15": list(os.getloadavg()),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import spacy
        import taxonerd
        env["spacy"], env["taxonerd"] = spacy.__version__, getattr(taxonerd, "__version__", "1.5.4 (pip)")
    except Exception as e:  # noqa: BLE001
        env["taxonerd_import_error"] = repr(e)
    if torch.cuda.is_available():
        env["gpu_name_torch"] = torch.cuda.get_device_name(0)
    return env


# ---------------------------------------------------------------- part 1: enumeration statistics
def enum_stats(d: pd.DataFrame, ments: list[list[str]]) -> dict[str, Any]:
    nm = np.array([len(m) for m in ments])
    pairs = [passage_pairs(a1, a2, ms) for a1, a2, ms in zip(d.species1, d.species2, ments)]
    npairs = np.array([len(p) for p in pairs])
    n = len(d)
    dist = {lab: {"passages": int(((nm >= lo) & (nm <= hi)).sum()),
                  "share": float(((nm >= lo) & (nm <= hi)).mean())} for lab, lo, hi in BINS}
    low = [{m.lower() for m in ms} for ms in ments]

    def found(sp: str, lows: set[str], alts: bool) -> bool:
        cands = [x.strip() for x in str(sp).split("|")] if alts else [str(sp).strip()]
        return any(c.lower() in lows for c in cands if c)

    both = np.array([found(a, L, False) and found(b, L, False) for a, b, L in zip(d.species1, d.species2, low)])
    both_alt = np.array([found(a, L, True) and found(b, L, True) for a, b, L in zip(d.species1, d.species2, low)])
    any_alt = np.array([found(a, L, True) or found(b, L, True) for a, b, L in zip(d.species1, d.species2, low)])
    k = int(np.ceil(0.10 * n))
    top = np.argsort(-npairs, kind="mergesort")[:k]
    # the candidate is kept in its benchmark order, mention pairs are sorted: a candidate that TaxoNERD
    # also finds, in the other alphabetical order, is scored twice (the same pair, already order-free)
    dup = sum((a2, a1) in set(p) and a1 != a2 for a1, a2, p in zip(d.species1, d.species2, pairs))
    return {
        "source": str(MENTION_CACHE.relative_to(REPO)),
        "n_passages": n,
        "mentions_per_passage": {"mean": float(nm.mean()), "median": float(np.median(nm)), "max": int(nm.max()),
                                 "total": int(nm.sum()), "distribution": dist},
        "pairs_per_passage_incl_candidate": {**{k_: v for k_, v in summary(npairs).items() if k_ != "n"},
                                             "total": int(npairs.sum()), "p90_method": "numpy linear interpolation"},
        "share_candidate_only_pair": float((npairs == 1).mean()),
        "passages_candidate_only_pair": int((npairs == 1).sum()),
        "candidate_both_strings_in_mentions": {
            "note": ("crude check: case-insensitive exact string match of species1 and species2 against the "
                     "passage's TaxoNERD mention strings; misses spelling variants, abbreviations "
                     "('A. aegypti'), plurals and partial spans"),
            "share_exact": float(both.mean()), "passages_exact": int(both.sum()),
            "share_allowing_pipe_alternatives": float(both_alt.mean()),
            "share_at_least_one_string_allowing_pipe_alternatives": float(any_alt.mean()),
            "candidates_with_pipe_alternatives": int(sum(("|" in str(a)) or ("|" in str(b))
                                                         for a, b in zip(d.species1, d.species2)))},
        "top10pct_passages_by_pairs": {"n_passages": k, "pairs": int(npairs[top].sum()),
                                       "share_of_all_scored_pairs": float(npairs[top].sum() / npairs.sum()),
                                       "min_pairs_in_top": int(npairs[top].min())},
        "candidate_scored_twice_as_reversed_mention_pair": int(dup),
        "forward_passes_per_passage": {
            "pair_max": {"formula": "2 orders x pairs x 3 seeds", "mean": float(2 * npairs.mean() * len(SEEDS)),
                         "total": int(2 * npairs.sum() * len(SEEDS))},
            "sentence": {"formula": "1 x 3 seeds", "mean": float(len(SEEDS)), "total": int(n * len(SEEDS))},
            "pair_own": {"formula": "2 orders x 1 pair x 3 seeds", "mean": float(2 * len(SEEDS)),
                         "total": int(2 * n * len(SEEDS))}},
    }


def seq_lengths(md: Path, df: pd.DataFrame) -> tuple[np.ndarray, int, str, int]:
    """Token length of each sequence one pass (one order) encodes, and the padded tokens in batches of BS."""
    import xenc_format
    from eval_unified import model_format, model_max_len
    from transformers import AutoTokenizer
    fmt, ml = model_format(md), model_max_len(md)
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True)
    q, p = xenc_format.build_many(fmt, df.species1, df.relation, df.species2, df.sentence.astype(str))
    enc = (tok(q, truncation=True, max_length=ml) if p is None
           else tok(q, p, truncation="only_second", max_length=ml))
    L = np.array([len(x) for x in enc["input_ids"]])
    padded = int(sum(len(L[i:i + BS]) * L[i:i + BS].max() for i in range(0, len(L), BS)))
    return L, padded, fmt, ml


def token_stats(d: pd.DataFrame, P: pd.DataFrame) -> dict[str, Any]:
    """Tokens each arm encodes (a hardware-independent cost proxy), with and without batch padding."""
    S = len(SEEDS)
    out: dict[str, Any] = {"note": ("tokens per encoded sequence after truncation at the checkpoint's max_len; "
                                    "padded = sum over batches of 64 (in scoring order) of batch size x longest "
                                    "sequence in the batch; totals multiply by argument orders and seeds")}
    for arm, pat, df, mult in (("pair_max", PAIR_DIR, P, 2 * S), ("pair_own", PAIR_DIR, d, 2 * S),
                               ("sentence", SENT_DIR, d, S)):
        md = REPO / pat.format(1)
        L, padded, fmt, ml = seq_lengths(md, df)
        out[arm] = {"input_format": fmt, "max_len": ml, "sequences_one_pass": int(len(L)),
                    "tokens_per_sequence_mean": float(L.mean()), "tokens_per_sequence_max": int(L.max()),
                    "sequences_truncated_at_max_len": int((L >= ml).sum()),
                    "tokens_total_all_seeds_orders": int(L.sum() * mult),
                    "padded_tokens_total_all_seeds_orders": int(padded * mult)}
    for k in ("tokens_total_all_seeds_orders", "padded_tokens_total_all_seeds_orders"):
        out[f"ratio_pair_max_to_sentence_{k}"] = out["pair_max"][k] / out["sentence"][k]
        out[f"ratio_pair_max_to_pair_own_{k}"] = out["pair_max"][k] / out["pair_own"][k]
    return out


def scorer_check(d: pd.DataFrame) -> dict[str, Any]:
    """The timed scorer against scripts/paperA_tables.py:order_free, both run now on the GPU."""
    import torch
    from paperA_tables import order_free
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res: dict[str, Any] = {"device": str(dev), "nvidia_smi_before": nvidia_smi()}
    for arm, pat, npy in (("pair_own", PAIR_DIR, "S_biodiv_pair.npy"), ("sentence", SENT_DIR, "S_biodiv_sentence.npy")):
        ref, mine = [], []
        for k in SEEDS:
            ref.append(order_free(REPO / pat.format(k), d, dev))
            sc = Scorer(REPO / pat.format(k), dev)
            mine.append(sc.order_free(d)[0]); del sc
        saved = np.load(REPO / "results/paperA_v2" / npy)
        res[arm] = {"timed_scorer_vs_order_free_now_max_abs_diff_per_seed":
                    [float(np.abs(a - b).max()) for a, b in zip(mine, ref)],
                    "order_free_now_vs_saved_npy_max_abs_diff": float(np.abs(np.mean(ref, 0) - saved).max()),
                    "order_free_now_vs_saved_npy_mean_abs_diff": float(np.abs(np.mean(ref, 0) - saved).mean()),
                    "saved_npy": f"results/paperA_v2/{npy}"}
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------- part 2: TaxoNERD timing
def taxonerd_run(texts: list[str], prefer_gpu: bool) -> dict[str, Any]:
    """Construct + load TaxoNERD (timed), one untimed warm-up call, then time each passage."""
    from taxonerd import TaxoNERD
    t0 = time.perf_counter()
    t = TaxoNERD(prefer_gpu=prefer_gpu)
    t.load("en_ner_eco_biobert")
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter(); t.find_in_text(str(texts[0])); warm_s = time.perf_counter() - t0
    per, found = [], []
    for s in texts:
        t0 = time.perf_counter()
        df = t.find_in_text(str(s))
        per.append(time.perf_counter() - t0)
        found.append(dedupe(df["text"].tolist() if len(df) else []))
    return {"load_seconds": load_s, "warmup_call_seconds": warm_s,
            "seconds_per_passage": summary(per), "found": found}


def taxonerd_timing(d: pd.DataFrame, ments: list[list[str]], threads: int) -> dict[str, Any]:
    import torch
    torch.set_num_threads(threads)
    idx = sample_idx(len(d))
    texts = d.sentence.iloc[idx].astype(str).tolist()
    res: dict[str, Any] = {"sample": {"n": SAMPLE_N, "seed": SAMPLE_SEED, "indices": idx.tolist(),
                                      "mean_chars": float(np.mean([len(s) for s in texts])),
                                      "mean_chars_all_437": float(d.sentence.astype(str).str.len().mean())},
                           "torch_num_threads": threads,
                           "timing_excludes": "construction + model load (reported as load_seconds) and one warm-up call"}

    def compare(found: list[list[str]]) -> dict[str, Any]:
        cached = [ments[i] for i in idx]
        exact = [f == c for f, c in zip(found, cached)]
        as_set = [{x.lower() for x in f} == {x.lower() for x in c} for f, c in zip(found, cached)]
        mism = [{"index": int(i), "cached": c, "found": f}
                for i, f, c, e in zip(idx, found, cached, as_set) if not e]
        return {"passages_identical_list": int(sum(exact)), "passages_same_set_case_insensitive": int(sum(as_set)),
                "passages_mismatch": int(len(exact) - sum(as_set)), "mismatches": mism}

    t0 = time.time()
    cpu = taxonerd_run(texts, prefer_gpu=False)
    res["cpu"] = {k: v for k, v in cpu.items() if k != "found"}
    res["cpu"]["mentions_vs_cache"] = compare(cpu["found"])
    res["cpu"]["wall_seconds_incl_load"] = time.time() - t0
    print(f"TaxoNERD CPU: load {cpu['load_seconds']:.1f}s, {cpu['seconds_per_passage']['mean']*1000:.0f} ms/passage",
          flush=True)

    gpu: dict[str, Any] = {"attempted": True}
    try:
        import importlib.util
        gpu["cupy_installed"] = importlib.util.find_spec("cupy") is not None
        g = taxonerd_run(texts, prefer_gpu=True)
        gpu.update({k: v for k, v in g.items() if k != "found"})
        gpu["mentions_vs_cache"] = compare(g["found"])
        gpu["status"] = "ok"
        gpu["nvidia_smi_after"] = nvidia_smi()
    except Exception as e:  # noqa: BLE001
        gpu["status"] = "skipped"
        gpu["error"] = f"{type(e).__name__}: {e}"
        gpu["why"] = ("TaxoNERD(prefer_gpu=True) calls spacy.require_gpu(), which needs cupy; cupy is not "
                      "installed in the environment and installing packages was not allowed, so GPU "
                      "extraction was not timed")
    res["gpu"] = gpu
    print(f"TaxoNERD GPU: {gpu.get('status')} {gpu.get('error', '')}", flush=True)
    return res


# ---------------------------------------------------------------- parts 3: cross-encoder timing
class Scorer:
    """A loaded checkpoint; `scores` mirrors scripts/eval_unified.py:score, `order_free` paperA_tables.order_free."""

    def __init__(self, md: Path, dev: Any) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from eval_unified import model_format, model_max_len
        self.md, self.dev = md, dev
        self.fmt, self.ml = model_format(md), model_max_len(md)
        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(md, local_files_only=True)
        self.m = AutoModelForSequenceClassification.from_pretrained(md, local_files_only=True).to(dev).eval()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - t0

    def scores(self, d: pd.DataFrame, bs: int = BS) -> np.ndarray:
        import torch
        import xenc_format
        q, p = xenc_format.build_many(self.fmt, d.species1, d.relation, d.species2, d.sentence.astype(str))
        P = []
        with torch.no_grad():
            for i in range(0, len(q), bs):
                if p is None:
                    e = self.tok(q[i:i + bs], truncation=True, max_length=self.ml, padding=True, return_tensors="pt")
                else:
                    e = self.tok(q[i:i + bs], p[i:i + bs], truncation="only_second", max_length=self.ml,
                                 padding=True, return_tensors="pt")
                e = e.to(self.dev)
                P.extend(torch.softmax(self.m(**e).logits.float(), -1)[:, 1].cpu().numpy())
        return np.array(P)

    def order_free(self, d: pd.DataFrame) -> tuple[np.ndarray, int]:
        """Scores and the number of sequences encoded (2 per row for order-sensitive formats)."""
        s = self.scores(d)
        if self.fmt == "sentence":
            return s, len(d)
        sw = d.copy()
        sw["species1"], sw["species2"] = d.species2.values, d.species1.values
        return np.maximum(s, self.scores(sw)), 2 * len(d)


def timed(fn: Any, dev: Any) -> tuple[Any, float]:
    import torch
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def time_arm(models: list[Scorer], df: pd.DataFrame, dev: Any) -> dict[str, Any]:
    """Score `df` with every seed's model; per-seed and total seconds, sequences, per-seed scores."""
    secs, seqs, scores = [], 0, []
    for sc in models:
        (s, k), t = timed(lambda sc=sc: sc.order_free(df), dev)
        secs.append(t); seqs += k; scores.append(s)
    total = float(sum(secs))
    return {"seconds_total": total, "seconds_per_seed": secs, "sequences_encoded": seqs,
            "sequences_per_second": seqs / total, "_scores": scores}


def strip(r: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def silver_labels(d: pd.DataFrame) -> np.ndarray:
    votes = np.mean([pd.read_csv(REPO / f"results/paperA_v2/llm/biodiv_{m}_sentence.csv").verdict.to_numpy()
                     for m in LLMS], axis=0)
    return np.where(d.label.to_numpy() == 1, 1, (votes >= 0.5).astype(int))


def gpu_timing(d: pd.DataFrame, P: pd.DataFrame, reps: int) -> dict[str, Any]:
    import torch
    from sklearn.metrics import average_precision_score as ap
    if not torch.cuda.is_available():
        return {"status": "skipped", "why": "torch.cuda.is_available() is False"}
    dev = torch.device("cuda")
    n = len(d)
    res: dict[str, Any] = {"device": torch.cuda.get_device_name(0), "batch_size": BS, "precision": "fp32",
                           "seeds": list(SEEDS), "n_passages": n, "n_pairs": int(len(P)),
                           "nvidia_smi_before": nvidia_smi(),
                           "timing_includes": "input building, tokenisation (CPU), forward pass, softmax, copy to host",
                           "timing_excludes": "model loading (load_seconds) and one untimed warm-up pass",
                           "note": ("the GPU was shared with another job during the measurement (see nvidia_smi_*), "
                                    "so these timings are upper bounds for a dedicated GPU")}
    torch.cuda.reset_peak_memory_stats()
    # pair checkpoints serve pair-max and pair-own
    pair = [Scorer(REPO / PAIR_DIR.format(k), dev) for k in SEEDS]
    res["load_seconds_pair_models"] = [s.load_seconds for s in pair]
    _, res["warmup_seconds_pair_own"] = timed(lambda: [s.order_free(d) for s in pair], dev)
    reps_out: list[dict[str, Any]] = []
    for r in range(reps):
        with SmiPoller() as poll:
            pm = time_arm(pair, P, dev)
            po = time_arm(pair, d, dev)
        reps_out.append({"pair_max": pm, "pair_own": po, "nvidia_smi_during": poll.stats()})
        print(f"GPU rep {r + 1}: pair-max {pm['seconds_total']:.1f}s, pair-own {po['seconds_total']:.1f}s", flush=True)
    del pair; torch.cuda.empty_cache()
    sent = [Scorer(REPO / SENT_DIR.format(k), dev) for k in SEEDS]
    res["load_seconds_sentence_models"] = [s.load_seconds for s in sent]
    _, res["warmup_seconds_sentence"] = timed(lambda: [s.order_free(d) for s in sent], dev)
    for r in range(reps):
        with SmiPoller() as poll:
            se = time_arm(sent, d, dev)
        reps_out[r]["sentence"] = se
        reps_out[r]["nvidia_smi_during_sentence"] = poll.stats()
        print(f"GPU rep {r + 1}: sentence {se['seconds_total']:.1f}s", flush=True)
    del sent; torch.cuda.empty_cache()
    res["peak_memory_allocated_by_this_job_mib"] = float(torch.cuda.max_memory_allocated() / 2**20)
    res["nvidia_smi_after"] = nvidia_smi()

    # sanity: the timed scorer reproduces the paper's scores
    last = reps_out[-1]
    own = np.mean(last["pair_own"]["_scores"], axis=0)
    sen = np.mean(last["sentence"]["_scores"], axis=0)
    per_seed_max = [P.assign(s=s).groupby("i").s.max().reindex(range(n)).to_numpy()
                    for s in last["pair_max"]["_scores"]]
    check: dict[str, Any] = {
        "pair_own_vs_S_biodiv_pair_npy_max_abs_diff":
            float(np.abs(own - np.load(REPO / "results/paperA_v2/S_biodiv_pair.npy")).max()),
        "sentence_vs_S_biodiv_sentence_npy_max_abs_diff":
            float(np.abs(sen - np.load(REPO / "results/paperA_v2/S_biodiv_sentence.npy")).max())}
    try:
        y = silver_labels(d)
        mine = [float(ap(y, s)) for s in per_seed_max]
        ref = json.loads(SILVER_JSON.read_text())["pair_max_auprc_per_seed"]
        check["pair_max_silver_auprc_per_seed"] = mine
        check["pair_max_silver_auprc_per_seed_sandbox"] = ref
        check["pair_max_silver_auprc_max_abs_diff"] = float(np.max(np.abs(np.array(mine) - np.array(ref))))
    except Exception as e:  # noqa: BLE001
        check["pair_max_silver_auprc_error"] = repr(e)
    res["reproduces_paper_scores"] = check

    res["repetitions"] = [{k: (strip(v) if isinstance(v, dict) and "_scores" in v else v) for k, v in rr.items()}
                          for rr in reps_out]
    arms = {}
    for arm, fpp in (("pair_max", 2 * len(P) * len(SEEDS) / n), ("sentence", len(SEEDS)), ("pair_own", 2 * len(SEEDS))):
        secs = [rr[arm]["seconds_total"] for rr in reps_out]
        arms[arm] = {"seconds_total_per_rep": secs, "seconds_total_min": float(min(secs)),
                     "ms_per_passage_per_rep": [1000 * s / n for s in secs],
                     "ms_per_passage_min": float(1000 * min(secs) / n),
                     "forward_passes_per_passage": float(fpp),
                     "forward_passes_total": int(reps_out[0][arm]["sequences_encoded"])}
    res["arms"] = arms
    res["ratio_pair_max_to_sentence_per_rep"] = [a / b for a, b in zip(arms["pair_max"]["seconds_total_per_rep"],
                                                                       arms["sentence"]["seconds_total_per_rep"])]
    res["ratio_pair_max_to_pair_own_per_rep"] = [a / b for a, b in zip(arms["pair_max"]["seconds_total_per_rep"],
                                                                       arms["pair_own"]["seconds_total_per_rep"])]
    return res


def cpu_timing(d: pd.DataFrame, P: pd.DataFrame, threads: int, reps: int) -> dict[str, Any]:
    import torch
    torch.set_num_threads(threads)
    dev = torch.device("cpu")
    idx = sample_idx(len(d))
    ds = d.iloc[idx].reset_index(drop=True)
    Ps = P[P.i.isin(set(idx.tolist()))].reset_index(drop=True)
    n, N = len(idx), len(d)
    res: dict[str, Any] = {"device": "cpu", "torch_num_threads": threads,
                           "rayon_num_threads_tokenizer": os.environ.get("RAYON_NUM_THREADS"),
                           "seed_used": 1, "batch_size": BS, "precision": "fp32",
                           "sample": {"n": n, "seed": SAMPLE_SEED, "pairs": int(len(Ps)),
                                      "pairs_per_passage": float(len(Ps) / n),
                                      "pairs_per_passage_all_437": float(len(P) / N)},
                           "load_average_before": list(os.getloadavg()),
                           "timing_includes": "input building, tokenisation, forward pass, softmax",
                           "timing_excludes": "model loading (load_seconds) and a one-batch warm-up"}
    pair = Scorer(REPO / PAIR_DIR.format(1), dev)
    sent = Scorer(REPO / SENT_DIR.format(1), dev)
    res["load_seconds"] = {"pair": pair.load_seconds, "sentence": sent.load_seconds}
    pair.scores(Ps.head(BS)); sent.scores(ds.head(BS))           # warm-up
    jobs = {"pair_max": (pair, Ps), "pair_own": (pair, ds), "sentence": (sent, ds)}
    reps_out = []
    for r in range(reps):
        rr = {}
        for arm, (sc, df) in jobs.items():
            rr[arm] = strip(time_arm([sc], df, dev))
        reps_out.append(rr)
        print(f"CPU rep {r + 1}: " + ", ".join(f"{a} {v['seconds_total']:.1f}s" for a, v in rr.items()), flush=True)
    res["repetitions"] = reps_out
    S = len(SEEDS)
    full_seq = {"pair_max": 2 * len(P) * S, "pair_own": 2 * N * S, "sentence": N * S}
    full_df = {"pair_max": (pair.md, P), "pair_own": (pair.md, d), "sentence": (sent.md, d)}
    arms = {}
    for arm, (sc, df) in jobs.items():
        secs = [rr[arm]["seconds_total"] for rr in reps_out]
        t, seq = float(min(secs)), reps_out[0][arm]["sequences_encoded"]
        orders = seq // len(df)
        pad_s = seq_lengths(sc.md, df)[1] * orders                  # one seed, sample, as scored
        pad_f = seq_lengths(*full_df[arm])[1] * orders * S           # three seeds, all 437 passages
        est_tok = t / pad_s * pad_f
        arms[arm] = {"seconds_one_seed_sample_per_rep": secs, "seconds_one_seed_sample_min": t,
                     "sequences_one_seed_sample": int(seq),
                     "ms_per_passage_one_seed": 1000 * t / n,
                     "ms_per_passage_three_seeds_extrapolated": 1000 * S * t / n,
                     "ms_per_sequence": 1000 * t / seq,
                     "padded_tokens_one_seed_sample": int(pad_s),
                     "padded_tokens_three_seeds_all_437": int(pad_f),
                     "est_seconds_all_437_three_seeds_by_passage": S * t / n * N,
                     "est_seconds_all_437_three_seeds_by_sequence": t / seq * full_seq[arm],
                     "est_seconds_all_437_three_seeds_by_padded_tokens": est_tok,
                     "est_ms_per_passage_three_seeds_by_padded_tokens": 1000 * est_tok / N}
    res["arms"] = arms
    res["ratio_pair_max_to_sentence"] = arms["pair_max"]["seconds_one_seed_sample_min"] / \
        arms["sentence"]["seconds_one_seed_sample_min"]
    res["ratio_pair_max_to_pair_own"] = arms["pair_max"]["seconds_one_seed_sample_min"] / \
        arms["pair_own"]["seconds_one_seed_sample_min"]
    res["ratio_pair_max_to_sentence_by_padded_tokens_estimate"] = \
        arms["pair_max"]["est_seconds_all_437_three_seeds_by_padded_tokens"] / \
        arms["sentence"]["est_seconds_all_437_three_seeds_by_padded_tokens"]
    res["note"] = ("one seed timed on the 50-passage sample; three-seed and full-benchmark figures are linear "
                   "extrapolations by passages, by encoded sequences, and by padded tokens (batches of 64 in "
                   "scoring order). The sample is lighter than the benchmark (pairs per passage "
                   f"{len(Ps) / n:.2f} vs {len(P) / N:.2f}), so the per-passage extrapolation understates pair-max; "
                   "the padded-token estimate also corrects for sequence length and batch padding")
    return res


# ---------------------------------------------------------------- output
def write(update: dict[str, Any]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cur: dict[str, Any] = {}
    if OUT.exists():
        cur = json.loads(OUT.read_text())
        bak = OUT.with_name(f"{OUT.name}.bak.{dt.datetime.now():%H%M}")
        if bak.exists():
            bak = OUT.with_name(f"{OUT.name}.bak.{dt.datetime.now():%H%M%S}")
        shutil.copy2(OUT, bak)
    cur.update(update)
    OUT.write_text(json.dumps(cur, indent=2, default=float))
    print(f"wrote {OUT.relative_to(REPO)} (keys: {', '.join(update)})", flush=True)


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--parts", default="stats", help="comma list of stats,taxonerd,gpu,cpu,check")
    a.add_argument("--threads", type=int, default=8, help="CPU threads for torch (TaxoNERD and cpu part)")
    a.add_argument("--reps", type=int, default=2, help="timed repetitions for the gpu and cpu parts")
    args = a.parse_args()
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    bad = set(parts) - {"stats", "taxonerd", "gpu", "cpu", "check"}
    if bad:
        a.error(f"unknown parts {bad}")
    if "cpu" in parts or "taxonerd" in parts:
        os.environ.setdefault("RAYON_NUM_THREADS", str(args.threads))   # HF fast-tokenizer pool
    d, ments = load_data()
    P = build_pairs(d, ments)
    assert len(P) == 6253, f"expected 6,253 pairs, built {len(P)}"
    upd: dict[str, Any] = {"description": ("cost of the pair enumeration behind pair-max on the 437-passage "
                                           "biodiversity benchmark; written by scripts/sentence_level/taxonerd_cost.py")}
    env = environment()
    prev =json.loads(OUT.read_text()).get("environment_per_part", {}) if OUT.exists() else {}
    envs = dict(prev)
    if "stats" in parts:
        upd["enumeration"] = enum_stats(d, ments)
        upd["enumeration"]["tokens"] = token_stats(d, P)
        envs["stats"] = env
    if "check" in parts:
        upd["scorer_check"] = scorer_check(d)
        envs["check"] = env
    if "taxonerd" in parts:
        upd["taxonerd_extraction_time"] = taxonerd_timing(d, ments, args.threads)
        envs["taxonerd"] = env
    if "gpu" in parts:
        upd["xenc_scoring_time_gpu"] = gpu_timing(d, P, args.reps)
        envs["gpu"] = env
    if "cpu" in parts:
        upd["xenc_scoring_time_cpu"] = cpu_timing(d, P, args.threads, args.reps)
        envs["cpu"] = env
    upd["environment_per_part"] = envs
    write(upd)


if __name__ == "__main__":
    main()
