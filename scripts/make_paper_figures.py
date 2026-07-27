"""All figures for the DASFAA paper: draw.io-style architecture + academic graphs."""
import sys
sys.path.insert(0, "/home/ahmad/.claude/skills/create-drawio")
sys.path.insert(0, ".")
from drawio import *
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/mnt/c/recearch/frederated_learning")
RES, FIG = ROOT / "results", ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


# ---------------- Fig 1: system architecture (draw.io style) ----------------
def fig_architecture():
    W, H = 212, 120
    fig, ax = fig_ax(W, H)
    title(ax, 4, H - 5, "Federated retraction prediction across publisher silos")

    # three publisher silos as containers
    silos = [("Publisher A (e.g., Elsevier)", 4), ("Publisher B (e.g., Springer)", 72),
             ("Publisher C (e.g., Hindawi)", 140)]
    for name, x in silos:
        container(ax, x, 64, 60, 40, name, "blue",
                  ["Private articles + labels", "Local feature extraction", "Local SGD, 2 epochs"])

    # server container
    container(ax, 58, 8, 92, 30, "Aggregation server", "green",
              ["Weighted FedAvg of client updates", "(optional) clip + Gaussian noise (DP)"])

    # updates flow down into the server
    for _, x in silos:
        arrow(ax, (x + 30, 64), (x + 30, 40))
    ax.text(4, 52, "model updates only\n(no articles leave the silo)", fontsize=9, color="#555555")

    # global model returns along the right edge
    ax.plot([150, 204], [24, 24], color="#444444", lw=1.2)
    ax.plot([204, 204], [24, 84], color="#444444", lw=1.2)
    arrow(ax, (204, 84), (200, 84))
    ax.text(184, 46, "global\nmodel", fontsize=9, color="#555555")

    box(ax, 4, 10, 40, 10, "Round r = 1..50", "gray", fs=10)
    save(fig, str(FIG / "architecture.png"))


# ---------------- Fig: scale curve ----------------
def fig_scale():
    d = pd.read_csv(RES / "abl_scale_curve.csv").groupby("n_train")["auprc"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(4.2, 2.9), dpi=200)
    ax.errorbar(d.index, d["mean"], yerr=d["std"], marker="o", ms=4,
                color=MUTED_B, capsize=3, lw=1.4)
    ax.set_xscale("log")
    ax.set_xlabel("Training-set size (log scale)")
    ax.set_ylabel("AUPRC")
    ax.axhline(0.27, ls=":", color="#999999", lw=1)
    ax.text(600, 0.278, "random baseline", fontsize=8, color="#777777")
    style_axes(ax)
    save_graph(fig, str(FIG / "scale_curve.png"))


# ---------------- Fig: per-silo local vs federated ----------------
def fig_per_silo():
    d = pd.read_csv(RES / "abl_per_silo.csv").sort_values("n_test", ascending=False)
    d = d[d["silo"] != "Other"]
    short = {"Hindawi Publishing Corporation": "Hindawi",
             "Springer Science+Business Media": "Springer",
             "Public Library of Science": "PLOS",
             "Spandidos Publishing": "Spandidos", "SAGE Publishing": "SAGE",
             "Taylor & Francis": "T&F", "Elsevier BV": "Elsevier",
             "IOS Press": "IOS Press", "Wiley": "Wiley"}
    names = [short.get(s, s) for s in d["silo"]]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(5.6, 2.9), dpi=200)
    ax.bar(x - 0.2, d["auprc_local"], 0.38, label="Local-only", color=MUTED_O)
    ax.bar(x + 0.2, d["auprc_fed"], 0.38, label="Federated", color=MUTED_B)
    ax.set_xticks(x, names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("AUPRC (own test slice)")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    save_graph(fig, str(FIG / "per_silo.png"))


# ---------------- Fig: feature ablation ----------------
def fig_ablation():
    d = pd.read_csv(RES / "abl_features.csv")
    g = d.groupby(["features", "model"])["auprc"].mean().unstack()
    order = ["all", "metadata_only", "text_only"]
    labels = ["All features", "Metadata only", "Text only"]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(4.2, 2.9), dpi=200)
    ax.bar(x - 0.2, g.loc[order, "central_xgb"], 0.38, label="Centralized XGB", color=MUTED_G)
    ax.bar(x + 0.2, g.loc[order, "fl_fedavg"], 0.38, label="Federated MLP", color=MUTED_B)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("AUPRC")
    ax.axhline(0.27, ls=":", color="#999999", lw=1)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    save_graph(fig, str(FIG / "feature_ablation.png"))


# ---------------- Fig: main comparison (repaint in consistent style) ------
def fig_main():
    d = pd.read_csv(RES / "main_publisher.csv")
    g = d.groupby("model")["auprc"].agg(["mean", "std"])
    order = ["local_only_avg", "fl_fedavg", "fl_fedprox", "central_mlp", "central_rf", "central_xgb"]
    labels = ["Local-only", "FedAvg", "FedProx", "Central MLP", "Central RF", "Central XGB"]
    colors = [MUTED_O, MUTED_B, MUTED_B, MUTED_G, MUTED_G, MUTED_G]
    g = g.loc[order]
    fig, ax = plt.subplots(figsize=(5.0, 2.9), dpi=200)
    ax.bar(labels, g["mean"], yerr=g["std"], color=colors, capsize=3)
    ax.axhline(0.27, ls=":", color="#999999", lw=1)
    ax.text(-0.4, 0.28, "random", fontsize=8, color="#777777")
    ax.set_ylabel("AUPRC")
    ax.tick_params(axis="x", labelsize=8, rotation=20)
    style_axes(ax)
    save_graph(fig, str(FIG / "main_results.png"))


# ---------------- Fig: convergence + DP (consistent restyle) --------------
def fig_convergence():
    fig, ax = plt.subplots(figsize=(4.2, 2.9), dpi=200)
    for strat, label, c in [("fedavg", "FedAvg", MUTED_B), ("fedprox", "FedProx", MUTED_G)]:
        hs = sorted(RES.glob(f"main_publisher_{strat}_seed*_history.json"))
        cat = pd.concat([pd.DataFrame(json.loads(h.read_text())) for h in hs])
        m = cat.groupby("round")["auprc"].agg(["mean", "std"])
        ax.plot(m.index, m["mean"], label=label, color=c, lw=1.4)
        ax.fill_between(m.index, m["mean"] - m["std"], m["mean"] + m["std"], alpha=0.15, color=c)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("AUPRC")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    save_graph(fig, str(FIG / "convergence.png"))


def fig_dp():
    d = pd.read_csv(RES / "dp_publisher.csv")
    g = d.groupby("sigma").agg(auprc=("auprc", "mean"), std=("auprc", "std"),
                               eps=("epsilon", "first")).reset_index()
    fig, ax = plt.subplots(figsize=(4.2, 2.9), dpi=200)
    fin = g[np.isfinite(g["eps"])]
    ax.errorbar(fin["eps"], fin["auprc"], yerr=fin["std"], marker="o", ms=4,
                color=MUTED_B, capsize=3, lw=1.4)
    for _, r in fin.iterrows():
        ax.annotate(f"$\\sigma$={r['sigma']}", (r["eps"], r["auprc"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color="#555555")
    nodp = g[~np.isfinite(g["eps"])]
    if len(nodp):
        ax.axhline(nodp["auprc"].iloc[0], ls="--", color="#999999", lw=1)
        ax.text(20, nodp["auprc"].iloc[0] + 0.006, "no DP", fontsize=8, color="#777777")
    ax.set_xscale("log")
    ax.set_xlabel(r"Privacy budget $\varepsilon$ (log scale, lower = more private)")
    ax.set_ylabel("AUPRC")
    style_axes(ax)
    save_graph(fig, str(FIG / "privacy_utility.png"))


if __name__ == "__main__":
    fig_architecture()
    fig_scale()
    fig_per_silo()
    fig_ablation()
    fig_main()
    fig_convergence()
    fig_dp()
    print("figures ->", FIG)
