"""Figures for the reproduction/saturation paper (LNCS)."""
import sys
sys.path.insert(0, "/home/ahmad/.claude/skills/create-drawio")
from drawio import *
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

FIG = Path("/mnt/c/recearch/frederated_learning/paper/gapfix/figures")
FIG.mkdir(parents=True, exist_ok=True)


def fig_reproduction():
    fig, ax = plt.subplots(figsize=(5.4, 2.9), dpi=200)
    labels = ["RF\n(features)", "XGBoost\n(features)", "SVM\n(features)",
              "BERT-base", "SciBERT", "BioBERT"]
    his = [0.88, 0.87, 0.67, 0.42, 0.51, 0.55]
    ours = [0.889, 0.875, 0.774, 0.44, 0.47, 0.47]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, his, 0.38, label="Reported (Usman & Balke)", color=MUTED_G)
    ax.bar(x + 0.2, ours, 0.38, label="Our reproduction", color=MUTED_B)
    ax.axhline(0.5, ls=":", color="#999999", lw=1)
    ax.text(3.6, 0.515, "chance", fontsize=8, color="#777777")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x, labels, fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    save_graph(fig, str(FIG / "reproduction.png"))


def fig_ladder():
    fig, ax = plt.subplots(figsize=(5.2, 2.9), dpi=200)
    labels = ["Baseline\n(orig. features)", "+ certainty\nprofiles",
              "+ paper-mill\nfeatures"]
    hp = [0.889, 0.917, 0.889]
    cv = [0.925, 0.919, 0.921]
    cv_std = [0.008, 0.008, 0.015]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, hp, 0.38, label="Original protocol (single split)", color=MUTED_B)
    ax.bar(x + 0.2, cv, 0.38, yerr=cv_std, capsize=3,
           label="5-fold CV $\\times$ 3 seeds", color=MUTED_O)
    ax.axhspan(0.925 - 0.016, 0.925 + 0.016, color="#cccccc", alpha=0.35, lw=0)
    ax.text(1.62, 0.947, "baseline CV noise band (2$\\sigma$)", fontsize=7.5, color="#777777")
    ax.set_ylim(0.84, 0.97)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x, labels, fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axes(ax)
    save_graph(fig, str(FIG / "ladder.png"))


def fig_deep_configs():
    fig, ax = plt.subplots(figsize=(5.4, 2.9), dpi=200)
    labels = ["CLS\nfrozen", "mean-pool\nfrozen", "section+attn\nfrozen",
              "fine-tune\n128 tok", "fine-tune\n512 tok", "unified\n(+features)"]
    acc = [0.55, 0.53, 0.55, 0.45, 0.47, 0.55]
    x = np.arange(len(labels))
    ax.bar(x, acc, 0.55, color=MUTED_R)
    ax.axhline(0.5, ls=":", color="#999999", lw=1)
    ax.axhline(0.925, ls="--", color=MUTED_G, lw=1.4)
    ax.text(0.0, 0.937, "engineered-feature baseline (0.925)", fontsize=8, color=MUTED_G)
    ax.text(4.7, 0.513, "chance", fontsize=8, color="#777777")
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("Accuracy (CV)")
    ax.set_xticks(x, labels, fontsize=7.5)
    style_axes(ax)
    save_graph(fig, str(FIG / "deep_configs.png"))


def fig_saturation():
    fig, ax = plt.subplots(figsize=(4.6, 2.9), dpi=200)
    n = np.logspace(np.log10(100), np.log10(100000), 100)
    p = 0.92
    half_ci = 1.96 * np.sqrt(p * (1 - p) / (0.2 * n))  # test set = 20% of corpus
    ax.plot(n, half_ci * 100, color=MUTED_B, lw=1.6)
    for N, lab, dy in [(464, "this benchmark", 0.4), (5000, "5k corpus", 0.4),
                       (50000, "50k corpus", 0.4)]:
        w = 1.96 * np.sqrt(p * (1 - p) / (0.2 * N)) * 100
        ax.scatter([N], [w], color=MUTED_R, zorder=5, s=18)
        ax.annotate(f"{lab}\n±{w:.1f} pts", (N, w), textcoords="offset points",
                    xytext=(8, dy * 10), fontsize=7.5, color="#555555")
    ax.set_xscale("log")
    ax.set_xlabel("Corpus size (log scale)")
    ax.set_ylabel("95% CI half-width (accuracy pts)")
    style_axes(ax)
    save_graph(fig, str(FIG / "saturation.png"))


def fig_profile_pipeline():
    W, H = 200, 52
    fig, ax = fig_ax(W, H)
    title(ax, 4, H - 5, "Certainty-profile extraction")
    flow_row(ax, [("Section text", "gray"), ("PySBD\nsegmentation", "blue"),
                  ("Sentence-level\ncertainty model", "purple"),
                  ("Profile: min, var,\nq10, frac-low", "orange"),
                  ("Cross-section\ntrajectory", "green")], y=16, x0=4, bw=36, bh=16, gap=4)
    save(fig, str(FIG / "profile_pipeline.png"))


if __name__ == "__main__":
    fig_reproduction()
    fig_ladder()
    fig_deep_configs()
    fig_saturation()
    fig_profile_pipeline()
    print("figures ->", FIG)
