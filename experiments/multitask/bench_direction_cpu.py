#!/usr/bin/env python3
"""CPU throughput: binary-only vs binary+direction, end to end, on a quiet box.

Everything the deployment does is inside the timed region: marking, tokenisation,
the forward pass, span pooling and the head. Arms are interleaved round-robin so
any residual load hits them equally.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
REPO = Path("/home/egaillac/MetaP/classifier")
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(Path(__file__).parent))
import xenc_format as X
from train_direction import DirectionHead, span_pool, AT_ID, HASH_ID

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n", type=int, default=256); ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    d = pd.read_csv(REPO / "data/evaluation/unified_test_set.csv").head(a.n)
    s1, s2 = d.species1.astype(str).tolist(), d.species2.astype(str).tolist()
    rel, txt = d.relation.astype(str).tolist(), d.sentence.astype(str).tolist()
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    bert = AutoModelForSequenceClassification.from_pretrained(a.model, local_files_only=True).eval()
    head = DirectionHead(bert.config.hidden_size).eval()
    head.load_state_dict(torch.load(Path(a.model) / "direction_head.pt", map_location="cpu"))
    pol = np.ones(len(d), dtype="int64")

    def arm(fmt, with_dir):
        with torch.no_grad():
            A, B = X.build_many(fmt, s1, rel, s2, txt)          # marking is inside the timer
            for i in range(0, len(A), a.bs):
                e = tok(A[i:i+a.bs], B[i:i+a.bs], truncation="only_second", max_length=256,
                        padding=True, return_tensors="pt")
                if with_dir:
                    o = bert(**e, output_hidden_states=True)
                    h = o.hidden_states[-1]
                    hA, okA = span_pool(h, e["input_ids"], AT_ID)
                    hB, okB = span_pool(h, e["input_ids"], HASH_ID)
                    head(hA, hB, h[:, 0], torch.tensor(pol[i:i+a.bs]))
                    torch.softmax(o.logits, -1)
                else:
                    torch.softmax(bert(**e).logits, -1)

    arms = {"A_binary_only_pair_canon": ("pair_canon", False),
            "B_binary_only_mark_canon": ("mark_canon", False),
            "C_binary+direction_mark_canon": ("mark_canon", True)}
    arm("mark_canon", True)                                      # warm-up
    times = {k: [] for k in arms}
    for r in range(a.reps):
        for k, (fmt, wd) in arms.items():
            t0 = time.perf_counter(); arm(fmt, wd); times[k].append(time.perf_counter() - t0)
    out = {"threads": a.threads, "n": len(d), "bs": a.bs, "reps": a.reps, "arms": {}}
    base = None
    for k, ts in times.items():
        med = float(np.median(ts)); pps = len(d) / med
        if base is None: base = pps
        out["arms"][k] = {"median_s": med, "pairs_per_s": pps, "ratio_vs_A": pps / base}
        print(f"{k:34s} {pps:7.2f} pairs/s   ratio {pps/base:.3f}   median {med:.2f}s")
    if a.out: json.dump(out, open(a.out, "w"), indent=2); print("wrote", a.out)

if __name__ == "__main__":
    main()
