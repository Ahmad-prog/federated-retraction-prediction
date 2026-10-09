"""P2: figures of the paper (and poster) from results_v2/ and paper/final/numbers.json.

  gr1.pdf/png  overview: how FedRetract is built and how the publishers train together (main text, Fig. 1)
  gr2.pdf/png  pooled AUPRC by training regime for the three backbones + baselines (supplement)
  gr3.pdf/png  AUPRC vs epsilon: record-level DP-SGD vs client-level DP-FedAvg (supplement; poster)
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


FONT = 1.0     # font multiplier (poster mode)
POSTER = False


def fig_regimes(num: dict, out: Path, scale: float):
    fams = [("c3_lora_abstracts", "ModernBERT-base"), ("c3_lora_abstracts_modernbert-large", "ModernBERT-large"),
            ("c3_lora_abstracts_qwen3-1.7b", "Qwen3-1.7B")]
    regs = [("local_lora|pooled", "Single publisher\n(on all publishers)", OK["grey"], "//"),
            ("fl_fedavg_lora|pooled", "FedAvg", OK["blue"], ""),
            ("central_lora|pooled", "Central\n(data pooled)", OK["orange"], ".."),
            ("fl_fedavg_lora_ft|pooled_personal", "FedAvg + local\nfine-tuning", OK["green"], "xx")]
    fig, ax = plt.subplots(figsize=(7.0, 4.6) if POSTER else (7.0 * scale, 2.8 * scale))
    scale = FONT
    w = 0.2
    x = np.arange(len(fams))
    for i, (key, lab, col, hat) in enumerate(regs):
        vals = [num["lora"][f].get(key, {}).get("auprc", np.nan) for f, _ in fams]
        sds = [num["lora"][f].get(key, {}).get("auprc_sd", 0) or 0 for f, _ in fams]
        b = ax.bar(x + (i - 1.5) * w, vals, w, yerr=sds, color=col, hatch=hat, edgecolor="black", lw=0.4,
                   label=lab.replace("\n", " "), capsize=2, error_kw={"lw": 0.6})
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=(5.2 if POSTER else 6.5) * scale)
    tab = num["b2"]["pub"]["publisher|central_xgb"]
    topic = float(num["topic"]["Topic only, XGBoost"]["auprc"])
    for y, lab, ls in [(tab, f"Tabular XGBoost ({tab:.2f})", "--"), (topic, f"Topic only ({topic:.2f})", "-."),
                       (0.290, "Chance (0.29)", ":")]:
        ax.axhline(y, color="black", ls=ls, lw=0.8, label=lab)
    ax.set_xticks(x, [l.replace("-", "-\n", 1) if POSTER else l for _, l in fams], fontsize=8 * scale)
    ax.set_ylabel("Pooled AUPRC", fontsize=8 * scale)
    ax.set_ylim(0.25, 0.86)
    ax.tick_params(labelsize=7 * scale)
    if POSTER:
        ax.set_ylim(0.25, 0.88)
        ax.legend(ncol=2, fontsize=6.0 * scale, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    else:
        ax.legend(ncol=4, fontsize=6.6 * scale, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.30))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"gr2.{ext}", dpi=300, bbox_inches="tight" if POSTER else None)
    plt.close(fig)


def fig_architecture(num: dict, out: Path):
    """Schematic for the main text, drawn at print size (6.5 x 1.70 in, 1 pt = 1 pt on the page)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    c = num["corpus"]
    W, H = 6.5, 1.70
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next((f for f in ["Arial", "Liberation Sans", "Nimbus Sans"] if f in have), "DejaVu Sans")
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ink, mute = "#1b2631", "#4d5966"
    col = {"amber": ("#fcebd2", "#d39a45"), "lilac": ("#e8e3f5", "#9486c4"), "blue": ("#dceaf7", "#5f93c4"),
           "green": ("#ddf0e2", "#5fa877"), "grey": ("#f1f4f7", "#9aa7b3"), "white": ("white", "#8c99a5")}

    def box(x, y, w, h, kind, title=None, sub=None, ts=7.0, ss=6.2, lw=0.8):
        fc, ec = col[kind]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=fc, ec=ec, lw=lw))
        if title and sub:
            ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=ts, weight="bold", color=ink)
            ax.text(x + w / 2, y + h * 0.29, sub, ha="center", va="center", fontsize=ss, color=mute, linespacing=1.1)
        elif title:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=ss, color=ink, linespacing=1.1)

    def arr(p0, p1, rad=0.0, color="#41505e"):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=color,
                                     shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))

    # ---------------- (a) FedRetract construction (left, x 0.02-3.0)
    ax.text(0.02, 1.69, "(a) Building FedRetract", fontsize=7.5, weight="bold", color=ink, va="top")
    bw, bh, gx = 0.90, 0.46, 0.13
    xs = [0.02, 0.02 + bw + gx, 0.02 + 2 * (bw + gx)]
    y1, y2 = 1.04, 0.45
    top = [("Retraction Watch", f"{c['n_retracted']:,} retracted\nresearch articles", "amber"),
           ("OpenAlex", "metadata, abstract,\nfield, countries", "blue"),
           ("Matched controls", f"same publisher, year\n{c['n_control']:,} articles", "amber")]
    bot = [("Publisher silos", "K = 10 publishers\nsplit 70/10/20", "green"),
           ("Leakage audit", "notice-word classifier\nat chance level", "lilac"),
           ("Cleaning", "notice text, title\nresidues, non-research", "lilac")]
    for x, (t, sb, k) in zip(xs, top):
        box(x, y1, bw, bh, k, t, sb)
    for x, (t, sb, k) in zip(xs, bot):
        box(x, y2, bw, bh, k, t, sb)
    for a, b in zip(xs[:-1], xs[1:]):
        arr((a + bw + 0.01, y1 + bh / 2), (b - 0.01, y1 + bh / 2))
        arr((b - 0.01, y2 + bh / 2), (a + bw + 0.01, y2 + bh / 2))
    arr((xs[2] + bw / 2, y1 - 0.01), (xs[2] + bw / 2, y2 + bh + 0.01))
    box(0.02, 0.04, 3.16, 0.33, "grey")
    ax.text(1.60, 0.255, f"FedRetract: {c['n']:,} research articles, {c['n_retracted'] / c['n']:.0%} retracted",
            ha="center", va="center", fontsize=6.8, weight="bold", color=ink)
    ax.text(1.60, 0.115, "+ prospective test: 213 later retractions, none of them in the corpus",
            ha="center", va="center", fontsize=6.0, color=mute)
    arr((xs[0] + bw / 2, y2 - 0.01), (xs[0] + bw / 2, 0.37 + 0.005))

    # ---------------- (b) federated training (right, x 3.35-6.48)
    x0 = 3.36
    ax.plot([x0 - 0.10, x0 - 0.10], [0.05, 1.67], color="#c5ced6", lw=0.6)
    ax.text(x0, 1.69, "(b) Training together without sharing articles", fontsize=7.5, weight="bold", color=ink, va="top")
    pw, ph, py = 0.92, 0.58, 0.96
    pxs = [x0, x0 + 1.07, x0 + 2.14]
    for x, nm in zip(pxs, ["Publisher 1", "Publisher 2", "Publisher 10"]):
        box(x, py, pw, ph, "blue")
        ax.text(x + pw / 2, py + ph - 0.10, nm, ha="center", va="center", fontsize=7.0, weight="bold", color=ink)
        box(x + 0.06, py + 0.27, pw - 0.12, 0.16, "white", "private articles", ss=6.0, lw=0.6)
        box(x + 0.06, py + 0.06, pw - 0.12, 0.16, "white", "LoRA, 1 epoch", ss=6.0, lw=0.6)
    ax.text(pxs[1] + pw + 0.06, py + ph / 2, "...", ha="center", va="center", fontsize=8, color=mute)
    sy, sh = 0.42, 0.36
    box(x0, sy, 3.06, sh, "green")
    ax.text(x0 + 1.53, sy + sh - 0.11, "Server: size-weighted FedAvg of LoRA + head, 8 rounds", ha="center",
            va="center", fontsize=6.6, weight="bold", color="#1e5c34")
    ax.text(x0 + 1.53, sy + 0.10, "4-17 M parameters per round; articles never leave a publisher",
            ha="center", va="center", fontsize=6.0, color=ink)
    for x in pxs:
        arr((x + pw * 0.35, py - 0.01), (x + pw * 0.35, sy + sh + 0.01))
        arr((x + pw * 0.65, sy + sh + 0.01), (x + pw * 0.65, py - 0.01), color="#2f7d4a")
    ax.text(pxs[0] + pw * 0.35 - 0.04, (py + sy + sh) / 2, "update", ha="right", va="center", fontsize=5.8, color=mute)
    ax.text(pxs[2] + pw * 0.65 + 0.04, (py + sy + sh) / 2, "global\nmodel", ha="left", va="center",
            fontsize=5.8, color="#2f7d4a", linespacing=1.0)
    box(x0, 0.04, 3.06, 0.31, "grey")
    ax.text(x0 + 1.53, 0.255, "Then each publisher fine-tunes its copy (FedAvg + local FT)", ha="center",
            va="center", fontsize=6.0, color=ink)
    ax.text(x0 + 1.53, 0.115, "Privacy options: DP-SGD per article, or DP-FedAvg per client", ha="center",
            va="center", fontsize=6.0, color=mute)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"gr1.{ext}", dpi=300)
    plt.close(fig)


def fig_privacy(num: dict, out: Path, scale: float):
    """AUPRC vs epsilon. The reference point of each DP series is drawn at "clip only" (clipped updates, no noise),
    so the vertical distance to it is the cost of the noise; hollow markers at "no DP" are training without clipping."""
    rec = pd.read_csv(RES / "b11_dp_heads/record.csv")
    jou = pd.read_csv(RES / "b11_dp_heads/journal.csv")
    fig, ax = plt.subplots(figsize=(7.0, 5.2) if POSTER else (6.2, 2.4))
    scale = FONT if POSTER else 1.25
    x_clip, x_none = 130, 400  # positions of "clip only" and "no DP" on the log axis

    def series(df, mdl=None, lab="", col="", mk="o", ls="-"):
        d = df if mdl is None else df[df.model == mdl]
        if "sigma" in d and mdl is None:  # journal clients: sigma = 0 is clipping without noise
            d = d.assign(x=d.epsilon.replace(np.inf, x_clip))
        else:
            d = d[np.isfinite(d.epsilon)].assign(x=lambda t: t.epsilon.replace(-1.0, x_clip))
        g = d.groupby("x").auprc.agg(["mean", "std"]).reset_index().sort_values("x")
        ax.errorbar(g.x, g["mean"], yerr=g["std"], color=col, marker=mk, ls=ls, lw=1.1, ms=4 * scale,
                    capsize=2, label=lab)
        if mdl is not None:  # no clipping, no noise
            u = df[(df.model == mdl) & np.isinf(df.epsilon)].auprc
            dx = {"central_dpsgd_linear": 0.88, "fl_dpsgd_linear": 1.12}.get(mdl, 1.0)  # keep near-equal points apart
            ax.errorbar([x_none * dx], [u.mean()], yerr=[u.std()], color=col, marker=mk, mfc="white", ls="none",
                        ms=4 * scale, capsize=2)

    series(rec, "central_dpsgd_linear", "DP-SGD (record), head, central", OK["orange"], "o")
    series(rec, "fl_dpsgd_linear", "DP-SGD (record), head, federated", OK["blue"], "s")
    series(jou, None, "DP-FedAvg (client), head, ~230 journals", OK["red"], "v", "--")
    lora = [r for r in num["dp_lora"] if r["fam"] == "ModernBERT-base" and r["model"].startswith("DP-SGD central")]
    if lora:
        ax.scatter([float(r["eps"]) for r in lora], [float(r["auprc"].split()[0]) for r in lora], marker="*",
                   s=70 * scale, color=OK["green"], edgecolor="black", lw=0.4, zorder=5, label="DP-SGD (record), LoRA, central")
        base = num["lora"]["c3_lora_abstracts"]["central_lora|pooled"]["auprc"]
        ax.scatter([x_none], [base], marker="*", s=70 * scale, color="white", edgecolor=OK["green"], lw=1.0, zorder=5)
    ax.axhline(0.29, color="black", ls=":", lw=0.8)
    ax.text(0.45, 0.30, "chance", fontsize=6.5 * scale, va="bottom")
    ax.set_xscale("log")
    tk = [0.5, 2, 8, 50, x_clip, x_none] if POSTER else [0.5, 1, 2, 4, 8, 16, 50, x_clip, x_none]
    ax.set_xticks(tk, ["clip\nonly" if t == x_clip else ("no\nDP" if t == x_none else f"{t:g}") for t in tk])
    ax.axvline(np.sqrt(50 * x_clip), color="grey", lw=0.5, ls="-")
    ax.minorticks_off()
    ax.set_xlabel("privacy budget $\\varepsilon$", fontsize=8 * scale)
    ax.set_ylabel("Pooled AUPRC", fontsize=8 * scale)
    ax.tick_params(labelsize=7 * scale)
    if POSTER:
        ax.legend(fontsize=6.0 * scale, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2)
    else:
        ax.legend(fontsize=6.4 * scale, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"gr3.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper/final"))
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--poster", action="store_true", help="large fonts for the A0 poster (figure ~12 in wide)")
    a = ap.parse_args()
    if a.poster:
        POSTER, FONT = True, 2.1
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    if a.poster:
        out = out.parent / (out.name + "_poster")
        (out / "figures").mkdir(parents=True, exist_ok=True)
    num = json.loads((Path(ROOT / "paper/final") / "numbers.json").read_text())
    if not a.poster:
        fig_architecture(num, out / "figures")
    fig_regimes(num, out / "figures", a.scale)
    fig_privacy(num, out / "figures", a.scale)
    print("figures written to", out / "figures")
