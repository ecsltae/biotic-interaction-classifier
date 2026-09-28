#!/usr/bin/env python3
"""Figure 4: precision, recall and F1 against the decision threshold.

Built from the rebuilt score vector (scripts/rebuild_paperA_numbers.py) for the model the paper
actually reports, on the 437 clean rows with the benchmark's species-level labels. An earlier
version of this script scored `dirhead/joint_a05_s1`, a different checkpoint, which is why its
output never matched the paper's numbers.

The point of the figure is that the verifier is a curve where the deployed filter is a single
point, so a curator can choose the operating point their workload calls for.

Okabe-Ito palette (CVD-safe), serif, single column.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCORES = REPO / "results/paperA_rebuild_2026-09-28/S_Verifier-PC_ensemble.npy"

BLUE, VERM, GREEN, GREY, INK = "#0072B2", "#D55E00", "#009E73", "#9a9a95", "#1a1a19"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 6.9, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.edgecolor": "#55554f", "figure.dpi": 200,
})


def load():
    d = pd.read_csv(REPO / "data/evaluation/unified_test_set.csv")
    scan = pd.read_csv(REPO / "results/test_contamination_scan.csv")
    keep = ~(d.in_train.to_numpy() | (scan.maxj.to_numpy() > 0.5))
    d = d[keep].reset_index(drop=True)
    v1 = np.load(REPO / "results/v1_decisions_449.npy")[keep].astype(int)
    return d.label.to_numpy(), v1, np.load(SCORES)


def main(out_stem="fig4_threshold", wide=False):
    y, V, S = load()
    grid = np.arange(0.01, 1.00, 0.01)
    P = np.array([precision_score(y, (S >= t).astype(int), zero_division=0) for t in grid])
    R = np.array([recall_score(y, (S >= t).astype(int), zero_division=0) for t in grid])
    F = np.array([f1_score(y, (S >= t).astype(int), zero_division=0) for t in grid])
    vP, vR, vF = precision_score(y, V), recall_score(y, V), f1_score(y, V)

    fig, ax = plt.subplots(figsize=(6.6, 2.5) if wide else (3.35, 2.6))
    ax.plot(grid, P, color=BLUE, lw=1.4, label="precision")
    ax.plot(grid, R, color=VERM, lw=1.4, label="recall")
    ax.plot(grid, F, color=GREEN, lw=1.6, label="F1")

    # the deployed filter is one point, not a curve
    for val, col, lab in ((vP, BLUE, "P"), (vR, VERM, "R"), (vF, GREEN, "F1")):
        ax.axhline(val, color=col, lw=0.7, ls=(0, (1, 2.5)), alpha=0.85, zorder=1)
    ax.text(0.985, vF + 0.012, "deployed filter (dotted)", color=INK, fontsize=6.4,
            ha="right", va="bottom")

    # the reported operating point, fitted on the other two blocks
    tau = 0.04
    i = int(np.argmin(np.abs(grid - tau)))
    ax.axvline(tau, color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.plot([tau], [F[i]], marker="o", ms=4.2, color=GREEN, mec="white", mew=0.9, zorder=6)
    ax.annotate(f"held-out $\\tau={tau:.2f}$, F1 {F[i]:.3f}", (tau, F[i]),
                xytext=(0.135, 0.975), textcoords="data", fontsize=6.6, color=INK,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7,
                                shrinkA=1.5, shrinkB=3.0))

    ax.set_xlabel("decision threshold $\\tau$")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, 1.02)
    ax.set_yticks(np.arange(0.3, 1.01, 0.1))
    ax.grid(axis="y", color="#e8e8e4", lw=0.55, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower left", frameon=False, handlelength=1.6, labelspacing=0.3,
              bbox_to_anchor=(0.02, 0.02))
    fig.tight_layout(pad=0.35)
    for ext in ("pdf", "png"):
        fig.savefig(Path(__file__).parent / f"{out_stem}.{ext}", bbox_inches="tight")
    print(f"wrote {out_stem}.{{pdf,png}}   deployed filter P={vP:.3f} R={vR:.3f} F1={vF:.3f}")


if __name__ == "__main__":
    main()
    main("fig4_threshold_wide", wide=True)
