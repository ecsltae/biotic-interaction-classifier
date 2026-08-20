#!/usr/bin/env python3
"""V2 sweep driver.

Runs configs in-process so the 4.2M-key gazetteer automaton is built once per
dataset rather than once per run, and frees the model between runs.

Shard with --shard i --n-shards k to spread across GPUs.
"""
from __future__ import annotations
import argparse, gc, itertools, json, os, sys, time, traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO / "src"))

SOFT = "data/training/distillation_soft_labels.csv"
V14 = "data/training/training_data_v14.csv"


def build_grid():
    cfgs = []
    # Regime S — the champion's regime. Soft-label distillation, 50,041 rows,
    # 8.48% positive, 0% pair-markable so pair conditioning is impossible here.
    for kd, alpha, scheme, pre in itertools.product(
            ["fixed", "v1_bug"], [0.3, 0.5, 0.7], ["full", "full_typed"], [0, 2]):
        cfgs.append(dict(regime="S", data=SOFT, soft_labels=SOFT, kd_mode=kd,
                         alpha=alpha, ner_scheme=scheme, pretrain_ner_epochs=pre,
                         pair_conditioning=False))
    # Regime P — v14, 34,880 rows, 32.7% positive, 69.1% pair-markable.
    # Hard CE (no soft labels exist for v14). The pair=False rows are the
    # no-marker control B4 requires; comparing pair=True against the 0.868
    # soft-label number would confound the marker with the dataset.
    for pair, alpha, scheme, pre in itertools.product(
            [False, True], [0.3, 0.5, 0.7], ["full", "full_typed"], [0, 2]):
        cfgs.append(dict(regime="P", data=V14, soft_labels="none", kd_mode="fixed",
                         alpha=alpha, ner_scheme=scheme, pretrain_ner_epochs=pre,
                         pair_conditioning=pair))
    return cfgs


def tag(c, seed):
    if c["regime"] == "S":
        return f"S_kd-{c['kd_mode']}_a{c['alpha']}_{c['ner_scheme']}_pre{c['pretrain_ner_epochs']}_s{seed}"
    return f"P_pair-{int(c['pair_conditioning'])}_a{c['alpha']}_{c['ner_scheme']}_pre{c['pretrain_ner_epochs']}_s{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-root", default="models/v2_sweep")
    ap.add_argument("--res-root", default="results/v2_sweep")
    ap.add_argument("--only-regime", default=None, choices=["S", "P"])
    ap.add_argument("--configs-json", default=None,
                    help="Explicit list of config dicts (for the multi-seed stage)")
    a = ap.parse_args()

    import torch
    from train_v2 import train

    if a.configs_json:
        grid = json.loads(Path(a.configs_json).read_text())
    else:
        grid = build_grid()
        if a.only_regime:
            grid = [c for c in grid if c["regime"] == a.only_regime]

    jobs = [(c, s) for c in grid for s in a.seeds]
    jobs = [j for i, j in enumerate(jobs) if i % a.n_shards == a.shard]
    # group by dataset so the gazetteer is rebuilt as rarely as possible
    jobs.sort(key=lambda j: (j[0]["data"], j[0]["ner_scheme"]))
    print(f"shard {a.shard}/{a.n_shards}: {len(jobs)} runs", flush=True)

    done = failed = 0
    for i, (c, seed) in enumerate(jobs, 1):
        name = tag(c, seed)
        out = REPO / a.out_root / name
        res = REPO / a.res_root / name
        if (res / "train_summary.json").exists():
            print(f"[{i}/{len(jobs)}] {name}  SKIP (done)", flush=True); done += 1; continue

        args = argparse.Namespace(
            data=c["data"], encoder="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
            ner_scheme=c["ner_scheme"], alpha=c["alpha"], epochs=a.epochs,
            pretrain_ner_epochs=c["pretrain_ner_epochs"], batch_size=a.batch_size,
            lr=2e-5, max_length=256, soft_labels=c["soft_labels"], temperature=2.0,
            kd_mode=c["kd_mode"], select_on="cls_auprc", seed=seed, split_seed=42,
            val_frac=0.1, pair_conditioning=c["pair_conditioning"],
            output_dir=str(out), results_dir=str(res))
        t0 = time.time()
        print(f"[{i}/{len(jobs)}] {name}", flush=True)
        try:
            s = train(args)
            (res / "config.json").write_text(json.dumps({**c, "seed": seed, "name": name}, indent=2))
            print(f"    dev AUPRC {s['best_dev_cls_auprc']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            done += 1
        except Exception:
            traceback.print_exc()
            failed += 1
        finally:
            gc.collect(); torch.cuda.empty_cache()
    print(f"shard {a.shard} finished: {done} done, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
