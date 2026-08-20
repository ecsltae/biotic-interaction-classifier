#!/usr/bin/env python3
"""Score a multitask checkpoint on the benchmarks and emit an honest report.

The threshold is never chosen here. It comes from the checkpoint's recorded
dev/prior value, or is passed explicitly. Probabilities are optionally divided
by the recorded training temperature (B1).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np, pandas as pd, torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "multitask"))
from eval.core import load_benchmark, report, format_report, threshold_from_prior  # noqa
from model import MultiTaskBiomedBERT  # noqa
from transformers import AutoTokenizer  # noqa


@torch.no_grad()
def predict(model, tok, texts, device, *, temperature=1.0, batch=128,
            pairs=None, max_length=256):
    model.eval()
    out_p = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        if pairs is not None:
            q = pairs[i:i + batch]
            enc = tok(q, chunk, truncation="only_second", max_length=max_length,
                      padding=True, return_tensors="pt").to(device)
        else:
            enc = tok(chunk, truncation=True, max_length=max_length,
                      padding=True, return_tensors="pt").to(device)
        o = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                  token_type_ids=enc.get("token_type_ids"))
        logits = o["cls_logits"].float() / temperature      # B1
        out_p.extend(torch.softmax(logits, -1)[:, 1].cpu().numpy().tolist())
    return np.array(out_p)


class _PlainWrap(torch.nn.Module):
    """Adapt a plain AutoModelForSequenceClassification to the multitask
    forward signature so both families score through one code path."""

    def __init__(self, inner, lowercase=False):
        super().__init__()
        self.inner = inner
        self.lowercase = lowercase

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **_):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        return {"cls_logits": self.inner(**kw).logits}


def load(model_dir, device):
    md = Path(model_dir)
    if (md / "multitask_config.json").exists():
        cfg = json.loads((md / "multitask_config.json").read_text())
        m = MultiTaskBiomedBERT.load(str(md), device=str(device))
        try:
            tok = AutoTokenizer.from_pretrained(md, local_files_only=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(cfg["encoder_name"])
        return m, tok, cfg
    # plain HF sequence classifier (template baseline, distilled_v2, ...)
    from transformers import AutoModelForSequenceClassification
    inner = AutoModelForSequenceClassification.from_pretrained(str(md), local_files_only=True)
    tok = AutoTokenizer.from_pretrained(str(md), local_files_only=True)
    m = _PlainWrap(inner).to(device).eval()
    cfg = {"encoder_name": str(md), "temperature": 1.0, "model_family": "plain_hf"}
    return m, tok, cfg


BENCHES = {
    "test500": dict(path=REPO / "data/evaluation/biotic_interaction_test_set.csv",
                    text_col="sentence", label_col="label", source_col="source"),
    "ep_relax": dict(path=REPO / "globi-relax_passages-triplets_2024-02-28_curation_EP.tsv",
                     text_col="sentence", label_col="evaluation_pair_interacting",
                     source_col=None, sep="\t", encoding="latin-1"),
    # Human-curated subset of test500 with recoverable taxon pairs (299 rows).
    # Excludes the 197 LLM-generated rows the audit flagged (A2) and is the only
    # benchmark on which a pair-conditioned model can be scored without a
    # train/inference mismatch.
    "test299": dict(path=REPO / "data/evaluation/test500_paired.csv",
                    text_col="sentence", label_col="label", source_col="source"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--benchmarks", nargs="+", default=["test500", "ep_relax"])
    ap.add_argument("--threshold", type=float, default=None,
                    help="Explicit threshold. Default: checkpoint's recorded dev threshold.")
    ap.add_argument("--threshold-source", default="dev",
                    choices=["dev", "prior", "explicit", "half"])
    ap.add_argument("--apply-temperature", action="store_true",
                    help="Divide logits by the recorded training temperature (B1)")
    ap.add_argument("--pair-conditioning", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    model, tok, cfg = load(a.model, device)
    name = a.name or Path(a.model).name
    temp = float(cfg.get("temperature", 2.0)) if a.apply_temperature else 1.0

    results = {}
    for bname in a.benchmarks:
        spec = dict(BENCHES[bname])
        path = spec.pop("path")
        # Enforce the pinned hash. Without this, expect_sha was dead code and
        # the "benchmark cannot silently change" guarantee was not in force.
        import json as _json
        _reg = REPO / "data/evaluation/BENCHMARKS.json"
        _sha = _json.loads(_reg.read_text()).get(bname, {}).get("sha256") if _reg.exists() else None
        bench = load_benchmark(bname, path, expect_sha=_sha, **spec)

        pairs = None
        if a.pair_conditioning:
            df = pd.read_csv(path, **({"sep": spec.get("sep", ",")} if "sep" in spec else {}),
                             encoding=spec.get("encoding", "utf-8"))
            c1 = next((c for c in ("species1", "species1_term", "source_species") if c in df.columns), None)
            c2 = next((c for c in ("species2", "species2_term", "target_species") if c in df.columns), None)
            if c1 and c2:
                pairs = [f"{str(x).strip()} [SEP] {str(y).strip()}"
                         for x, y in zip(df[c1], df[c2])]
            else:
                print(f"  [{bname}] no pair columns; scoring without pair conditioning")

        probs = predict(model, tok, bench.texts, device, temperature=temp, pairs=pairs)

        if a.threshold_source == "explicit":
            t = a.threshold
        elif a.threshold_source == "prior":
            t = threshold_from_prior(float(cfg.get("train_pos_rate", 0.0848)), bench.prevalence)
        elif a.threshold_source == "half":
            t = 0.5
        else:
            t = float(cfg.get("threshold_dev", cfg.get("best_threshold", 0.5)))

        rep = report(bench, probs, t, model_name=name)
        rep["threshold_source"] = a.threshold_source
        rep["temperature_applied"] = temp
        rep["model_dir"] = str(a.model)
        results[bname] = rep
        results[bname]["_probs"] = probs.tolist()
        if not a.quiet:
            print(format_report(rep)); print()

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=2))
        print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
