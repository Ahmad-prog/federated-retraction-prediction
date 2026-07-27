"""Generate all paper figures from results/*.csv into paper/figures/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 200,
})

MODEL_LABELS = {
    "central_logreg": "Centralized LogReg", "central_rf": "Centralized RF (Usman-style)",
    "central_xgb": "Centralized XGBoost", "central_mlp": "Centralized MLP",
    "local_only_avg": "Local-only (avg)",
    "fl_fedavg": "FedAvg", "fl_fedprox": "FedProx", "fl_fedbal": "FedBal (ours)",
}


def fig_main_bars(stem: str) -> None:
    f = RES / f"{stem}.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    g = df.groupby("model")["auprc"].agg(["mean", "std"]).reindex(MODEL_LABELS.keys()).dropna()
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    colors = ["#888"] * 4 + ["#c44"] + ["#2a6"] * 3
    ax.bar([MODEL_LABELS[m] for m in g.index], g["mean"], yerr=g["std"],
           color=colors[: len(g)], capsize=3)
    ax.set_ylabel("AUPRC")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG / f"{stem}_auprc.png")
    plt.close(fig)


def fig_convergence(stem: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    plotted = False
    for strat, label in [("fedavg", "FedAvg"), ("fedprox", "FedProx"), ("fedbal", "FedBal (ours)")]:
        hists = sorted(RES.glob(f"{stem}_{strat}_seed*_history.json"))
        if not hists:
            continue
        dfs = [pd.DataFrame(json.loads(h.read_text())) for h in hists]
        cat = pd.concat(dfs).groupby("round")["auprc"].agg(["mean", "std"])
        ax.plot(cat.index, cat["mean"], label=label)
        ax.fill_between(cat.index, cat["mean"] - cat["std"], cat["mean"] + cat["std"], alpha=0.2)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Communication round")
    ax.set_ylabel("AUPRC")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / f"{stem}_convergence.png")
    plt.close(fig)


def fig_privacy(stem: str = "dp_publisher") -> None:
    f = RES / f"{stem}.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    g = df.groupby("sigma").agg(auprc=("auprc", "mean"), auprc_std=("auprc", "std"),
                                epsilon=("epsilon", "first")).reset_index()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    finite = g[g["epsilon"] < 1e9]
    ax.errorbar(finite["epsilon"], finite["auprc"], yerr=finite["auprc_std"],
                marker="o", capsize=3)
    nodp = g[g["epsilon"] >= 1e9]
    if len(nodp):
        ax.axhline(nodp["auprc"].iloc[0], ls="--", color="#888", label="No DP")
        ax.legend(frameon=False)
    ax.set_xscale("log")
    ax.set_xlabel(r"Privacy budget $\varepsilon$ (log scale)")
    ax.set_ylabel("AUPRC")
    fig.tight_layout()
    fig.savefig(FIG / "privacy_utility.png")
    plt.close(fig)


if __name__ == "__main__":
    for stem in ["main_publisher", "main_field"]:
        fig_main_bars(stem)
        fig_convergence(stem)
    fig_privacy()
    print("Figures ->", FIG)
