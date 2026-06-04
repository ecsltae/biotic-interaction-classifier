#!/usr/bin/env python3
"""
Export slide figures as PNGs — styled to match the Madrid Beamer presentation.
Colors, row shading, and table structure mirror the LaTeX source exactly.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

OUT = Path("/home/egaillac/MetaP/slides_export")
OUT.mkdir(exist_ok=True)

# ── Exact colors from classifier_slides.tex ───────────────────────────────────
GOOD      = "#228B22"   # \definecolor{good}
BAD       = "#B22222"   # \definecolor{bad}
WARN      = "#D28C00"   # \definecolor{warn}
NEUTRAL   = "#3C5AA0"   # \definecolor{neutral}  (Madrid header blue)
HIGHLIGHT = "#0064B4"   # \definecolor{highlight}

# Row tint variants (15–20% opacity equivalent)
GOOD15    = "#D6EDD6"
BAD15     = "#F5CECE"
WARN15    = "#FAF0CE"
GOOD20    = "#C8E6C8"
LGREY     = "#F2F2F2"   # alternating row
WHITE     = "#FFFFFF"

DPI = 180

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          12,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.titlesize":     14,
    "axes.titleweight":   "bold",
    "figure.dpi":         DPI,
})


# ── Helper: render a table the way Beamer does ────────────────────────────────
def render_table(ax, rows, cols, row_colors,
                 col_widths=None, font_size=11, row_height=0.65):
    """
    row_colors: list of hex colors, one per data row (not header).
    col_widths: relative widths; None = equal.
    """
    ax.axis("off")
    n_cols = len(cols)
    n_rows = len(rows)
    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols

    # normalise to sum=1
    total = sum(col_widths)
    col_widths = [w / total for w in col_widths]

    total_h = (n_rows + 1) * row_height   # +1 for header
    total_w = 1.0

    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)

    def draw_row(row_idx, values, bg_color, text_color="black",
                 bold=False, fontsize=None):
        if fontsize is None:
            fontsize = font_size
        y_bot = total_h - (row_idx + 1) * row_height
        x = 0.0
        for c, (val, cw) in enumerate(zip(values, col_widths)):
            ax.add_patch(plt.Rectangle((x, y_bot), cw, row_height,
                                       facecolor=bg_color, edgecolor=WHITE, lw=1.5))
            ax.text(x + cw / 2, y_bot + row_height / 2, str(val),
                    ha="center", va="center", fontsize=fontsize,
                    color=text_color,
                    fontweight="bold" if bold else "normal",
                    wrap=True)
            x += cw

    # Header row
    draw_row(0, cols, NEUTRAL, text_color=WHITE, bold=True)
    # Data rows
    for i, (row, bg) in enumerate(zip(rows, row_colors)):
        draw_row(i + 1, row, bg)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 01 — Dataset version F1 bar chart
# ══════════════════════════════════════════════════════════════════════════════
def fig_version_f1():
    versions = ["v7", "v8", "v9", "v10", "v12", "v14", "v15", "Qwen+\nsoft labels"]
    f1s      = [0.788, 0.695, 0.644, 0.722, 0.729, 0.706, 0.828, 0.868]
    colors   = [GOOD, BAD, BAD, BAD, WARN, BAD, WARN, GOOD]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(versions, f1s, color=colors, edgecolor=WHITE,
                  linewidth=1.4, width=0.62, zorder=3)

    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#333333")

    ax.set_ylim(0.58, 0.91)
    ax.set_ylabel("EP-relax F1", fontsize=12, color="#444444")
    ax.set_title("EP-relax F1 by Training Dataset Version", pad=14)
    ax.axhline(0.788, color=NEUTRAL, linestyle="--", linewidth=1.2,
               alpha=0.6, zorder=2)
    ax.text(7.4, 0.793, "v7 baseline", fontsize=9, color=NEUTRAL)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    patches = [
        mpatches.Patch(color=GOOD, label="Improvement over v7"),
        mpatches.Patch(color=WARN, label="Partial improvement"),
        mpatches.Patch(color=BAD,  label="Regression"),
    ]
    ax.legend(handles=patches, fontsize=10, framealpha=0.9,
              loc="lower left", bbox_to_anchor=(0.01, 0.02))

    fig.tight_layout()
    fig.savefig(OUT / "fig_01_version_f1.png", bbox_inches="tight")
    plt.close()
    print("fig_01_version_f1.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 02 — Model evolution F1 (matches slide 8)
# ══════════════════════════════════════════════════════════════════════════════
def fig_model_evolution():
    models = ["SVM / RF\n(TF-IDF)", "BiomedBERT\n(v7 data)",
              "Ensemble\nBERT × T5", "Distilled v2\n(student)",
              "Multi-task\n(best)"]
    f1s    = [0.620, 0.788, 0.865, 0.808, 0.868]
    colors = ["#AAAAAA", NEUTRAL, NEUTRAL, NEUTRAL, GOOD]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(models, f1s, color=colors, edgecolor=WHITE,
                  linewidth=1.4, width=0.58, zorder=3)

    for bar, val, col in zip(bars, f1s, colors):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=11.5, fontweight="bold",
                color=GOOD if col == GOOD else "#333333")

    ax.set_ylim(0.50, 0.92)
    ax.set_ylabel("EP-relax F1", fontsize=12, color="#444444")
    ax.set_title("Model Architecture Evolution — EP-relax F1", pad=14)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Step annotations
    improvements = [
        (0, 1, "+0.168\ncontextual repr."),
        (1, 2, "+0.077\ncomplementarity"),
        (3, 4, "+0.060\nNER aux. task"),
    ]
    for i, j, label in improvements:
        xm = (i + j) / 2
        y  = max(f1s[i], f1s[j]) + 0.035
        ax.annotate("", xy=(j - 0.25, y), xytext=(i + 0.25, y),
                    arrowprops=dict(arrowstyle="-|>", color="#666666", lw=1.3))
        ax.text(xm, y + 0.008, label, ha="center", fontsize=8.5,
                color="#555555")

    fig.tight_layout()
    fig.savefig(OUT / "fig_02_model_evolution.png", bbox_inches="tight")
    plt.close()
    print("fig_02_model_evolution.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 03 — Distillation table (matches slide 10)
# ══════════════════════════════════════════════════════════════════════════════
def fig_distillation_table():
    cols = ["Student model", "T", "α", "EP-relax F1"]
    rows = [
        ["v1  BiomedBERT",   "4",   "0.7", "0.785"],
        ["v2  BiomedBERT ★", "2",   "0.5", "0.808"],
        ["v3  BiomedBERT",   "4",   "0.9", "0.808"],
        ["v4  DistilBERT",   "2",   "0.5", "0.792"],
        ["v5  SciBERT",      "2",   "0.5", "0.793"],
        ["v6  BiomedBERT",   "1.5", "0.5", "0.779"],
    ]
    row_colors = [LGREY, GOOD20, WHITE, LGREY, WHITE, LGREY]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Knowledge Distillation — Hyperparameter Search", pad=14)
    render_table(ax, rows, cols, row_colors,
                 col_widths=[2.8, 0.6, 0.6, 1.0], font_size=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_03_distillation_table.png", bbox_inches="tight")
    plt.close()
    print("fig_03_distillation_table.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 04 — Full ablation table (matches slide 13)
# ══════════════════════════════════════════════════════════════════════════════
def fig_ablation_table():
    cols = ["Config", "NER scheme", "α", "NER\npre-train", "EP F1", "P", "R"]
    rows = [
        ["basic_a05",              "basic",      "0.5", "0 ep", "0.841", "0.763", "0.938"],
        ["typed_a05",              "typed",      "0.5", "0 ep", "0.852", "0.767", "0.958"],
        ["full_a03",               "full",       "0.3", "0 ep", "0.854", "0.854", "0.854"],
        ["full_typed_a05",         "full_typed", "0.5", "0 ep", "0.847", "0.746", "0.979"],
        ["full_a05_ner2",          "full",       "0.5", "2 ep", "0.824", "0.778", "0.875"],
        ["full_typed_a03_ner2",    "full_typed", "0.3", "2 ep", "0.830", "0.759", "0.917"],
        ["full_typed_a05_ner2 ★",  "full_typed", "0.5", "2 ep", "0.868", "0.793", "0.958"],
        ["full_typed_a05_ner2 5ep","full_typed", "0.5", "5 ep", "0.816", "0.764", "0.875"],
        ["── v7 hard CE",          "full_typed", "0.5", "2 ep", "0.756", "0.810", "0.708"],
        ["── v12 hard CE",         "full_typed", "0.5", "2 ep", "0.816", "0.764", "0.875"],
        ["── distilled_v2",        "—",          "—",   "—",    "0.808", "0.750", "0.875"],
        ["── Ensemble BERT×T5",    "—",          "—",   "—",    "0.865", "—",     "—"    ],
    ]
    row_colors = [
        LGREY, WHITE, LGREY, WHITE,
        LGREY, WHITE,
        GOOD20,           # best row
        LGREY,
        WARN15, WARN15,   # hard CE baselines
        WARN15, WARN15,
    ]

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_title("Multi-task Ablation — Full Results on EP-relax\n"
                 "★ = best model    ── rows = hard-CE / ensemble baselines",
                 pad=14)
    render_table(ax, rows, cols, row_colors,
                 col_widths=[3.2, 1.8, 0.6, 1.0, 0.9, 0.8, 0.8],
                 font_size=10.5, row_height=0.62)
    fig.tight_layout()
    fig.savefig(OUT / "fig_04_ablation_table.png", bbox_inches="tight")
    plt.close()
    print("fig_04_ablation_table.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 05 — Bootstrap CI (matches slide 14)
# ══════════════════════════════════════════════════════════════════════════════
def fig_bootstrap_ci():
    models    = ["BiomedBERT v7", "Distilled v2", "Multi-task (best)"]
    ep_f1     = [0.785, 0.808, 0.868]
    ep_lo     = [0.693, 0.717, 0.791]
    ep_hi     = [0.864, 0.884, 0.930]
    ev_f1     = [0.831, 0.768, 0.776]
    ev_lo     = [0.797, 0.726, 0.735]
    ev_hi     = [0.863, 0.805, 0.814]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Model Comparison — 95% Bootstrap Confidence Intervals (n = 10,000)",
                 fontsize=14, fontweight="bold", y=1.01)

    datasets = [
        (axes[0], ep_f1, ep_lo, ep_hi, "EP-relax  (n = 99, 48 pos)",  2),
        (axes[1], ev_f1, ev_lo, ev_hi, "eval_sets  (n = 599, 7 files)", 0),
    ]

    for ax, f1s, los, his, title, winner in datasets:
        x = np.arange(len(models))
        bar_colors = [GOOD if i == winner else NEUTRAL for i in range(len(models))]
        yerr_lo = [f - l for f, l in zip(f1s, los)]
        yerr_hi = [h - f for f, h in zip(f1s, his)]

        bars = ax.bar(x, f1s, width=0.55, color=bar_colors,
                      edgecolor=WHITE, linewidth=1.2, zorder=3)
        ax.errorbar(x, f1s, yerr=[yerr_lo, yerr_hi],
                    fmt="none", ecolor="#333333", capsize=7,
                    capthick=1.5, linewidth=1.5, zorder=4)

        for i, (bar, val) in enumerate(zip(bars, f1s)):
            ypos = val + yerr_hi[i] + 0.008
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=11,
                    fontweight="bold" if i == winner else "normal",
                    color=GOOD if i == winner else "#333333")

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10.5)
        ax.set_ylim(0.62, 0.98)
        ax.set_ylabel("F1", fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

    # McNemar annotations below bars
    axes[0].text(0.5, -0.16,
                 "MT vs BiomedBERT v7: p = 0.016 ✓  |  MT vs distilled: p = 0.149 (n.s.)",
                 transform=axes[0].transAxes, ha="center", fontsize=9.5,
                 color="#555555", style="italic")
    axes[1].text(0.5, -0.16,
                 "v7 vs MT: p = 0.009 ✓  |  v7 vs distilled: p = 0.003 ✓",
                 transform=axes[1].transAxes, ha="center", fontsize=9.5,
                 color="#555555", style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "fig_05_bootstrap_ci.png", bbox_inches="tight")
    plt.close()
    print("fig_05_bootstrap_ci.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 06 — Performance ceiling bar chart (matches slide 16)
# ══════════════════════════════════════════════════════════════════════════════
def fig_ceiling():
    types  = ["endoparasiteOf\n32%", "hasHost\n14%", "preysOn\n12%",
              "pathogenOf\n22%", "other\n20%"]
    shares = [32, 14, 12, 22, 20]
    colors = [BAD, BAD, BAD, GOOD, NEUTRAL]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(types, shares, color=colors, edgecolor=WHITE,
                  linewidth=1.4, width=0.62, zorder=3)

    for bar, val, col in zip(bars, shares, colors):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                f"{val}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold",
                color=BAD if col == BAD else "#333333")

    ax.set_ylabel("Share of EP-relax positives", fontsize=12)
    ax.set_ylim(0, 42)
    ax.set_title("EP-relax Positive Distribution\n"
                 "58% of positives are ecological types absent from PubMed",
                 pad=12)
    ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    patches = [
        mpatches.Patch(color=BAD,     label="Absent from PubMed training corpus"),
        mpatches.Patch(color=GOOD,    label="Present in training corpus"),
        mpatches.Patch(color=NEUTRAL, label="Partially represented"),
    ]
    ax.legend(handles=patches, fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_06_ceiling.png", bbox_inches="tight")
    plt.close()
    print("fig_06_ceiling.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 07 — NER scheme table (matches slide 13)
# ══════════════════════════════════════════════════════════════════════════════
def fig_ner_schemes():
    cols = ["Scheme", "Labels included", "What is tagged", "# labels", "Best EP F1"]
    rows = [
        ["basic",       "O, B-SP, I-SP",
         "Any species mention\n(no role distinction)", "3", "0.841"],
        ["typed",       "O, B/I-HOST, B/I-PATHOGEN, B/I-SPECIES",
         "Role-distinguished species\n(host vs pathogen vs ambiguous)", "7", "0.852"],
        ["full",        "O, B-SP, I-SP, B-INT, I-INT",
         "Any species\n+ interaction verbs", "5", "0.854"],
        ["full_typed ★","O, B/I-HOST, B/I-PATHOGEN, B/I-SPECIES, B/I-INT",
         "Role-distinguished species\n+ interaction verbs", "9", "0.868"],
    ]
    row_colors = [LGREY, WHITE, LGREY, GOOD20]

    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.set_title("NER Label Schemes  —  B = Beginning of span, I = Inside span\n"
                 "★ = winning configuration", pad=12)
    render_table(ax, rows, cols, row_colors,
                 col_widths=[1.5, 3.8, 2.8, 0.9, 1.0],
                 font_size=10.5, row_height=0.72)
    fig.tight_layout()
    fig.savefig(OUT / "fig_07_ner_schemes.png", bbox_inches="tight")
    plt.close()
    print("fig_07_ner_schemes.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 08 — Dataset version history table (matches slide 6)
# ══════════════════════════════════════════════════════════════════════════════
def fig_version_table():
    cols = ["Version", "Samples", "Positives", "EP F1", "Key change"]
    rows = [
        ["v1",                "40K",   "17.8K",  "—",     "Initial GloBI templates"],
        ["v2",                "45K",   "13.4K",  "—",     "Hard negatives added"],
        ["v3–v6",             "~32K",  "~6.5K",  "—",     "Template refinement & diversity"],
        ["v7 ★",              "25K",   "7.3K",   "0.788", "Claude API validation on every positive"],
        ["v8",                "26K",   "—",      "0.695", "SIBiLS harvest, 92% pathogen bias"],
        ["v9",                "26K",   "—",      "0.644", "Regex labels — noise"],
        ["v10",               "27K",   "8.1K",   "0.722", "Real PMC sentences, pathogen-biased"],
        ["v11–v12",           "28K",   "8.1K",   "0.729", "Diversity push + score>0 filter"],
        ["v14",               "35K",   "11.4K",  "0.706", "Filter repeated — regression"],
        ["v15–v19 (R.A.)",    "26–35K","8–11K",  "0.828↓","Research agent experiments, all regressed"],
        ["Qwen+soft labels ★","44K",   "4.1K",   "0.868", "Qwen YES/NO + ensemble soft labels — BEST"],
    ]
    row_colors = [
        LGREY, WHITE, LGREY,
        GOOD15,             # v7
        BAD15, BAD15, BAD15,
        WARN15,             # v12
        BAD15,              # v14
        WARN15,             # v15–v19
        GOOD20,             # Qwen best
    ]

    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.set_title("Training Dataset Version History\n"
                 "★ = key milestones    R.A. = Research Agent experiments",
                 pad=12)
    render_table(ax, rows, cols, row_colors,
                 col_widths=[2.2, 1.2, 1.2, 0.9, 4.5],
                 font_size=10.5, row_height=0.62)
    fig.tight_layout()
    fig.savefig(OUT / "fig_08_version_table.png", bbox_inches="tight")
    plt.close()
    print("fig_08_version_table.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 09 — Multi-task architecture diagram (matches slide 11)
# ══════════════════════════════════════════════════════════════════════════════
def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)

    def box(x, y, w, h, line1, line2="", fc="#F2F2F2", ec="#555555", lw=1.5,
            bold=False, fontsize=12):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                    facecolor=fc, edgecolor=ec, linewidth=lw))
        y_mid = y + h / 2
        if line2:
            ax.text(x + w/2, y_mid + 0.17, line1, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold" if bold else "normal")
            ax.text(x + w/2, y_mid - 0.22, line2, ha="center", va="center",
                    fontsize=fontsize - 1.5, color="#555555")
        else:
            ax.text(x + w/2, y_mid, line1, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold" if bold else "normal")

    def arr(x1, y1, x2, y2, label="", label_side="right"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.6))
        if label:
            xm = (x1+x2)/2 + (0.18 if label_side == "right" else -0.18)
            ym = (y1+y2)/2
            ax.text(xm, ym, label, ha="left" if label_side == "right" else "right",
                    va="center", fontsize=9.5, color="#555555")

    # Input box
    box(3.0, 7.8, 4.0, 0.85, "Input sentence  (tokenised)",
        fc="#E8F4FD", ec=HIGHLIGHT, lw=1.8)

    # Encoder box
    box(2.2, 5.6, 5.6, 1.8,
        "BiomedBERT Encoder",
        "12 transformer layers  —  one 768-dim vector per token",
        fc="#D6E8F8", ec=NEUTRAL, lw=2.0, bold=True)

    # Shared weights label
    ax.text(5.0, 5.45, "← shared weights for both tasks →",
            ha="center", fontsize=9.5, color=NEUTRAL, style="italic")

    # CLS head
    box(0.4, 3.0, 3.8, 1.6,
        "[CLS] classification head",
        "Linear(768 → 2)  →  P(interaction)",
        fc="#EAF5EA", ec=GOOD, lw=1.8)

    # NER head
    box(5.8, 3.0, 3.8, 1.6,
        "NER head  (per token)",
        "Linear(768 → 9)  →  BIO label",
        fc="#FFF3E0", ec=WARN, lw=1.8)

    # Output boxes
    box(0.7, 1.0, 3.2, 0.9, "YES / NO  + confidence score",
        fc="#EAF5EA", ec=GOOD)
    box(6.1, 1.0, 3.2, 0.9, "HOST / PATHOGEN / INT / O …",
        fc="#FFF3E0", ec=WARN)

    # Arrows
    arr(5.0, 7.8, 5.0, 7.4)
    arr(4.6, 5.6, 2.6, 4.6, "[CLS] token", "right")
    arr(5.4, 5.6, 7.4, 4.6, "All tokens", "right")
    arr(2.3, 3.0, 2.3, 1.9)
    arr(7.7, 3.0, 7.7, 1.9)

    ax.set_title("Multi-task BiomedBERT — Architecture",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / "fig_09_architecture.png", bbox_inches="tight")
    plt.close()
    print("fig_09_architecture.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 10 — Two-phase training diagram (matches slide 12)
# ══════════════════════════════════════════════════════════════════════════════
def fig_training_phases():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Two-Phase Training Strategy", fontsize=14, fontweight="bold", y=1.01)

    for ax in axes:
        ax.set_xlim(0, 5); ax.set_ylim(0, 5.5)
        ax.axis("off")

    def pbox(ax, x, y, w, h, line1, line2="", fc=LGREY, ec="#555555",
             frozen=False, lw=1.5):
        ls = "--" if frozen else "-"
        ec2 = BAD if frozen else ec
        lw2 = 2.2 if frozen else lw
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                    facecolor=fc, edgecolor=ec2, lw=lw2,
                                    linestyle=ls))
        ym = y + h / 2
        if line2:
            ax.text(x+w/2, ym+0.15, line1, ha="center", va="center", fontsize=10.5)
            ax.text(x+w/2, ym-0.18, line2, ha="center", va="center",
                    fontsize=9, color="#555555")
        else:
            ax.text(x+w/2, ym, line1, ha="center", va="center", fontsize=10.5)

    def parr(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.4))

    # ── Phase 1 ──────────────────────────────────────────────────
    ax = axes[0]
    ax.set_title("Phase 1 — NER pre-train (2 epochs)",
                 fontsize=12, fontweight="bold", color=NEUTRAL, pad=8)
    pbox(ax, 1.1, 4.2, 2.8, 0.9, "BiomedBERT encoder", fc="#D6E8F8", ec=NEUTRAL, lw=1.8)
    pbox(ax, 0.2, 2.5, 1.8, 1.1,
         "[CLS] head", "FROZEN", fc="#EEEEEE", frozen=True)
    pbox(ax, 3.0, 2.5, 1.8, 1.1,
         "NER head", "updated", fc="#FFF3E0", ec=WARN, lw=1.8)
    pbox(ax, 3.0, 0.8, 1.8, 0.9,
         "NER cross-entropy loss", fc="#FFE0B2", ec=WARN)
    parr(ax, 2.5, 4.2, 1.3, 3.6)
    parr(ax, 2.5, 4.2, 3.7, 3.6)
    parr(ax, 3.9, 2.5, 3.9, 1.7)
    ax.text(1.1, 3.0, "✗  no gradient", ha="center", fontsize=9,
            color=BAD, fontweight="bold")

    # ── Phase 2 ──────────────────────────────────────────────────
    ax = axes[1]
    ax.set_title("Phase 2 — Joint training (3 epochs)",
                 fontsize=12, fontweight="bold", color=GOOD, pad=8)
    pbox(ax, 1.1, 4.2, 2.8, 0.9, "BiomedBERT encoder", fc="#D6E8F8", ec=NEUTRAL, lw=1.8)
    pbox(ax, 0.2, 2.5, 1.8, 1.1,
         "[CLS] head", "updated", fc="#EAF5EA", ec=GOOD, lw=1.8)
    pbox(ax, 3.0, 2.5, 1.8, 1.1,
         "NER head", "updated", fc="#FFF3E0", ec=WARN, lw=1.8)
    pbox(ax, 0.2, 0.8, 1.8, 0.9,
         "KL soft loss  (α = 0.5)", fc="#C8E6C9", ec=GOOD)
    pbox(ax, 3.0, 0.8, 1.8, 0.9,
         "NER CE loss  (1−α = 0.5)", fc="#FFE0B2", ec=WARN)
    parr(ax, 2.5, 4.2, 1.3, 3.6)
    parr(ax, 2.5, 4.2, 3.7, 3.6)
    parr(ax, 1.1, 2.5, 1.1, 1.7)
    parr(ax, 3.9, 2.5, 3.9, 1.7)
    ax.text(2.5, 0.25, "ℒ = α · KL + (1−α) · CE_NER",
            ha="center", fontsize=10, color="#333333")

    fig.tight_layout()
    fig.savefig(OUT / "fig_10_training_phases.png", bbox_inches="tight")
    plt.close()
    print("fig_10_training_phases.png")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures...")
    fig_version_f1()
    fig_model_evolution()
    fig_distillation_table()
    fig_ablation_table()
    fig_bootstrap_ci()
    fig_ceiling()
    fig_ner_schemes()
    fig_version_table()
    fig_architecture()
    fig_training_phases()

    print(f"\nAll saved to {OUT}/")
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name}")
