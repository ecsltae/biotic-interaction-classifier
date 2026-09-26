#!/usr/bin/env python3
"""End-to-end CPU throughput: marking + tokenisation + forward + span pooling, 8 threads.

Everything a deployment actually pays for is inside the timed region, and every arm is run
interleaved round-robin so a change in machine load hits all arms equally.
"""
import sys, time, json, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "scripts"))
import fmt, data as D, xenc_format
from model import DirModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def rows(n=256):
    d = pd.read_csv(REPO / "data/evaluation/unified_test_set.csv")
    d = d[~d.in_train].reset_index(drop=True)
    i = np.arange(n) % len(d)
    return d.iloc[i].reset_index(drop=True)


def run_baseline(md, d, bs, fmt_name):
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True)
    m = AutoModelForSequenceClassification.from_pretrained(md, local_files_only=True).eval()
    def f():
        with torch.no_grad():
            for i in range(0, len(d), bs):
                c = d.iloc[i:i + bs]
                q, p = xenc_format.build_many(fmt_name, c.species1, c.relation, c.species2,
                                              c.sentence.astype(str))
                e = tok(q, p, truncation="only_second", max_length=256, padding=True,
                        return_tensors="pt")
                torch.softmax(m(**e).logits.float(), -1)[:, 1].numpy()
    return f


def run_joint(md, d, bs):
    from eval_dirhead import load
    m, tok, _ = load(md, device="cpu")
    dd = d.rename(columns={"sentence": "text", "species1": "s1", "species2": "s2",
                           "relation": "rel"}).copy()
    dd["y_bin"], dd["y_dir"] = -100, -100
    cf = D.collate(tok)
    def f():
        ds = D.Joint(dd, tok, 256, augment=False)
        with torch.no_grad():
            for i in range(0, len(ds), bs):
                b = cf([ds[j] for j in range(i, min(i + bs, len(ds)))])
                b.pop("ok"); b.pop("y_bin"); b.pop("y_dir")
                bl, dl = m(**b)
                torch.softmax(bl.float(), -1)[:, 1].numpy(); torch.softmax(dl.float(), -1).numpy()
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", default="models/dirhead/joint_rule_s1")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    d = rows(a.n)
    arms = {
        "A shipped binary, triple (today)": run_baseline(REPO / "models/student_v3/xenc_s1", d, a.bs, "triple"),
        "B binary, mark_canon": run_baseline(REPO / "models/lever_arch/fmt_mark_canon_s1", d, a.bs, "mark_canon"),
        "C joint binary+direction, mark_canon": run_joint(REPO / a.joint, d, a.bs),
    }
    for f in arms.values():
        f()                                          # warm-up, untimed
    res = {k: [] for k in arms}
    for _ in range(a.reps):
        for k, f in arms.items():
            t0 = time.perf_counter(); f(); res[k].append(a.n / (time.perf_counter() - t0))
    out = {k: dict(pairs_per_s_median=float(np.median(v)), runs=[round(x, 2) for x in v])
           for k, v in res.items()}
    base = out["A shipped binary, triple (today)"]["pairs_per_s_median"]
    for k in out:
        out[k]["vs_today"] = round(out[k]["pairs_per_s_median"] / base, 4)
    out["_meta"] = dict(threads=a.threads, n=a.n, batch=a.bs, reps=a.reps,
                        loadavg=list(__import__("os").getloadavg()))
    print(json.dumps(out, indent=2))
    (REPO / "results/dirhead").mkdir(parents=True, exist_ok=True)
    (REPO / "results/dirhead/cpu_bench.json").write_text(json.dumps(out, indent=2))
