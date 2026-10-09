"""P2: figures of the paper (and poster) from results_v2/ and paper/final/numbers.json.

  fig1_regimes.pdf/png  pooled AUPRC by training regime for the three backbones + baselines
  fig2_privacy.pdf/png  AUPRC vs epsilon: record-level DP-SGD vs client-level DP-FedAvg
Colour-blind-safe palette (Okabe-Ito); every series is also marked by shape or hatch.

Usage: python scripts/v2/p2_paper_figures.py [--out paper/final] [--poster]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results_v2"
OK = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "red": "#D55E00", "purple": "#CC79A7", "grey": "#8C8C8C"}


def fig_regimes(num: dict, out: Path, scale: float):
    fams = [("c3_lora_abstracts", "ModernBERT-base"), ("c3_lora_abstracts_modernbert-large", "ModernBERT-large"),
            ("c3_lora_abstracts_qwen3-1.7b", "Qwen3-1.7B")]
    regs = [("local_lora|pooled", "Single publisher\n(on all publishers)", OK["grey"], "//"),
            ("fl_fedavg_lora|pooled", "FedAvg", OK["blue"], ""),
            ("central_lora|pooled", "Central\n(data pooled)", OK["orange"], ".."),
            ("fl_fedavg_lora_ft|pooled_personal", "FedAvg + local\nfine-tuning", OK["green"], "xx")]
    fig, ax = plt.subplots(figsize=(7.0 * scale, 2.8 * scale))
    w = 0.2
    x = np.arange(len(fams))
    for i, (key, lab, col, hat) in enumerate(regs):
        vals = [num["lora"][f].get(key, {}).get("auprc", np.nan) for f, _ in fams]
        sds = [num["lora"][f].get(key, {}).get("auprc_sd", 0) or 0 for f, _ in fams]
        b = ax.bar(x + (i - 1.5) * w, vals, w, yerr=sds, color=col, hatch=hat, edgecolor="black", lw=0.4,
                   label=lab.replace("\n", " "), capsize=2, error_kw={"lw": 0.6})
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6.5 * scale)
    tab = num["b2"]["pub"]["publisher|central_xgb"]
    topic = float(num["topic"]["Topic only, XGBoost"]["auprc"])
    for y, lab, ls in [(tab, f"Tabular XGBoost ({tab:.2f})", "--"), (topic, f"Topic only ({topic:.2f})", "-."),
                       (0.290, "Chance (0.29)", ":")]:
        ax.axhline(y, color="black", ls=ls, lw=0.8, label=lab)
    ax.set_xticks(x, [l for _, l in fams], fontsize=8 * scale)
    ax.set_ylabel("Pooled AUPRC", fontsize=8 * scale)
    ax.set_ylim(0.25, 0.86)
    ax.tick_params(labelsize=7 * scale)
    ax.legend(ncol=4, fontsize=6.6 * scale, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.30))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig1_regimes.{ext}", dpi=300)
    plt.close(fig)


def fig_privacy(num: dict, out: Path, scale: float):
    rec = pd.read_csv(RES / "b11_dp_heads/record.csv")
    jou = pd.read_csv(RES / "b11_dp_heads/journal.csv")
    b4 = pd.read_csv(RES / "b4_scale_dp/results.csv")
    fig, ax = plt.subplots(figsize=(7.0 * scale, 2.5 * scale))
    eps_inf = 2000  # where "no privacy" is drawn

    def series(df, xcol, mdl=None, lab="", col="", mk="o", ls="-"):
        d = df if mdl is None else df[df.model == mdl]
        d = d[d[xcol] != -1].copy()
        d["x"] = d[xcol].replace(np.inf, eps_inf)
        g = d.groupby("x").auprc.agg(["mean", "std"]).reset_index().sort_values("x")
        ax.errorbar(g.x, g["mean"], yerr=g["std"], color=col, marker=mk, ls=ls, lw=1.1, ms=4 * scale,
                    capsize=2, label=lab)

    series(rec, "epsilon", "central_dpsgd_linear", "Record DP-SGD, central head", OK["orange"], "o")
    series(rec, "epsilon", "fl_dpsgd_linear", "Record DP-SGD, federated head", OK["blue"], "s")
    series(jou, "epsilon", None, "Client DP, 230 journal clients", OK["red"], "v", "--")
    pub = b4[b4.exp == "dp_publisher"]
    series(pub.rename(columns={}), "epsilon", None, "Client DP, 10 publishers (tabular)", OK["purple"], "^", ":")
    lora = [r for r in num["dp_lora"] if r["fam"] == "ModernBERT-base" and r["model"].startswith("DP-SGD central")]
    if lora:
        ax.scatter([float(r["eps"]) for r in lora], [float(r["auprc"].split()[0]) for r in lora], marker="*",
                   s=70 * scale, color=OK["green"], edgecolor="black", lw=0.4, zorder=5, label="Record DP-SGD, LoRA (central)")
        base = num["lora"]["c3_lora_abstracts"]["central_lora|pooled"]["auprc"]
        ax.scatter([eps_inf], [base], marker="*", s=70 * scale, color=OK["green"], edgecolor="black", lw=0.4, zorder=5)
    ax.axhline(0.29, color="black", ls=":", lw=0.8)
    ax.text(0.45, 0.295, "chance", fontsize=6.5 * scale)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8, 16, 50, 150, 400, eps_inf], ["0.5", "1", "2", "4", "8", "16", "50", "150", "400", "no DP"])
    ax.minorticks_off()
    ax.set_xlabel("privacy budget $\\varepsilon$", fontsize=8 * scale)
    ax.set_ylabel("Pooled AUPRC", fontsize=8 * scale)
    ax.tick_params(labelsize=7 * scale)
    ax.legend(fontsize=6.6 * scale, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig2_privacy.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper/final"))
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    num = json.loads((Path(ROOT / "paper/final") / "numbers.json").read_text())
    fig_regimes(num, out / "figures", a.scale)
    fig_privacy(num, out / "figures", a.scale)
    print("figures written to", out / "figures")
