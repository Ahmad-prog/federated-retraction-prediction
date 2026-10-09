"""P1: regenerate every table and headline number of the paper from results_v2/.

Reads only the result files written by the b*/c* scripts (no model is re-run) and writes
  paper/final/tables/*.tex   LaTeX tables (main text + appendix)
  paper/final/numbers.json   every number quoted in the running text

Rules: smoke runs (`*_smoke`) and results_v2/_stale are excluded; values are mean +- std
over seeds; n = number of seeds.

Usage: python scripts/v2/p1_paper_tables.py [--out paper/final]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results_v2"
NUM: dict = {}

SILO_SHORT = {
    "Elsevier BV": "Elsevier", "Hindawi Publishing Corporation": "Hindawi", "IOS Press": "IOS Press",
    "Other": "Other", "Public Library of Science": "PLOS", "SAGE Publishing": "SAGE",
    "Spandidos Publishing": "Spandidos", "Springer Science+Business Media": "Springer",
    "Taylor & Francis": "Taylor \\& Francis", "Wiley": "Wiley", "BioMed Central": "BioMed Central",
    "Dove Medical Press": "Dove", "Frontiers Media": "Frontiers",
}
MODEL = {
    "central_logreg": "Central LogReg", "central_rf": "Central RF", "central_xgb": "Central XGBoost",
    "central_mlp": "Central MLP", "local_mlp": "Local MLP", "local_xgb": "Local XGBoost",
    "fl_fedavg": "FedAvg", "fl_fedprox": "FedProx", "fl_scaffold": "SCAFFOLD", "fl_fedbal": "FedBal",
    "fl_xgb_cyclic": "Cyclic fed.\\ XGBoost", "fl_fedavg_ft": "FedAvg + local FT",
    "fl_fedavg_ft10": "FedAvg + local FT",
    "central_lora": "Central", "fl_fedavg_lora": "FedAvg", "fl_fedavg_lora_ft": "FedAvg + local FT",
    "local_lora": "Local only", "fl_ditto_lora": "Ditto", "fl_fedper_lora": "FedPer",
    "fl_dp_lora": "Client-DP FedAvg", "dpsgd_central_lora": "DP-SGD central",
    "dpsgd_fedavg_lora": "DP-SGD federated",
    "central_dpsgd_linear": "Central DP-SGD", "fl_dpsgd_linear": "Federated DP-SGD",
    "dp_fedavg": "DP-FedAvg", "dp_fedavg_emb": "DP-FedAvg",
}
FAM = {
    "c3_lora_abstracts": "ModernBERT-base", "c3_lora_abstracts_modernbert-large": "ModernBERT-large",
    "c3_lora_abstracts_qwen3-1.7b": "Qwen3-1.7B", "c3_lora_ftabs": "ModernBERT-base (OA subset, abstract)",
    "c3_lora": "ModernBERT-base (OA subset, full text 2,048)", "c3_lora_8192": "ModernBERT-base (OA subset, full text 8,192)",
}


def esc(s) -> str:
    s = str(s)
    return SILO_SHORT.get(s, s.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%"))


def ms(x: pd.Series, d: int = 3) -> str:
    x = x.dropna()
    if len(x) == 0:
        return "--"
    if len(x) == 1:
        return f"{x.iloc[0]:.{d}f}"
    return f"{x.mean():.{d}f} $\\pm$ {x.std(ddof=1):.{d}f}"


def m(x: pd.Series, d: int = 3) -> str:
    x = x.dropna()
    return "--" if len(x) == 0 else f"{x.mean():.{d}f}"


def table(name: str, caption: str, header: list[str], rows: list[list[str]], label: str,
          align: str | None = None, size: str = "\\footnotesize", group_breaks: list[int] | None = None,
          spanner: str | None = None, section_rows: dict | None = None, place: str = "!htbp"):
    """spanner: a LaTeX header row placed above `header`; section_rows: {row index: label} inserts a full-width
    italic label row (preceded by a rule) before that row."""
    align = align or ("l" * 1 + "r" * (len(header) - 1))
    if name.startswith("a_") and place == "!htbp":
        place = "H"
    out = [f"\\begin{{table}}[{place}]", f"\\caption{{{caption}}}\\label{{{label}}}", size,
           f"\\begin{{tabular*}}{{\\hsize}}{{@{{\\extracolsep{{\\fill}}}}{align}@{{}}}}", "\\toprule"]
    if spanner:
        out.append(spanner)
    out += [" & ".join(header) + "\\\\", "\\colrule"]
    for i, r in enumerate(rows):
        if section_rows and i in section_rows:
            if i > 0:
                out.append("\\colrule")
            out.append(f"\\multicolumn{{{len(header)}}}{{@{{}}l}}{{\\textit{{{section_rows[i]}}}}}\\\\")
        elif group_breaks and i in group_breaks and i > 0:
            out.append("\\colrule")
        out.append(" & ".join(str(c) for c in r) + "\\\\")
    out += ["\\botrule", "\\end{tabular*}", "\\end{table}", ""]
    (TAB / f"{name}.tex").write_text("\n".join(out), encoding="utf-8")


def pooled_rows(d: pd.DataFrame) -> pd.DataFrame:
    """Pooled-view rows; local models report one pooled row per silo -> average per seed."""
    p = d[d.view == "pooled"]
    allr = p[p.silo == "ALL"]
    per = p[p.silo != "ALL"]
    if len(per):
        keys = [c for c in ["setting", "partition", "fam", "run", "emb", "with_tab", "model", "seed"] if c in per]
        per = per.groupby(keys, as_index=False)[["auprc", "roc_auc", "recall_at_5fpr"]].mean()
        per["view"], per["silo"] = "pooled", "ALL"
    return pd.concat([allr, per], ignore_index=True)


def client_dp_eps(sigma: float, rounds: int = 8, delta: float = 1e-5) -> float:
    """RDP epsilon of full-participation DP-FedAvg (q = 1) as used by c3_lora_fl.py --regime fedavg_dp."""
    from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
    orders = [1 + x / 10 for x in range(1, 100)] + list(range(12, 256))
    return float(get_privacy_spent(orders=orders, rdp=compute_rdp(q=1.0, noise_multiplier=sigma, steps=rounds,
                                                                  orders=orders), delta=delta)[0])


def no_smoke(files):
    return [f for f in files if "smoke" not in f and "_stale" not in f]


# --------------------------------------------------------------------------- loaders
def load_c3() -> pd.DataFrame:
    rows = []
    for f in no_smoke(glob.glob(str(RES / "c3_lora*" / "results_*.csv"))):
        d = pd.read_csv(f)
        d["fam"] = Path(f).parent.name
        d["run"] = Path(f).stem.replace("results_", "")
        rows.append(d)
    c3 = pd.concat(rows, ignore_index=True)
    # Ditto / FedPer runs also report their FedAvg backbone: keep the dedicated FedAvg runs only
    dup = c3.run.str.startswith(("ditto", "fedper")) & (c3.model == "fl_fedavg_lora")
    c3 = c3[~dup]
    return pd.concat([c3[c3.view != "pooled"], pooled_rows(c3)], ignore_index=True)


# --------------------------------------------------------------------------- tables
def t_silos():
    f = RES / "corpus_stats.json"  # copy of release/FedRetract/stats.json (written by r1_build_release.py)
    if not f.exists():
        f = ROOT / "release/FedRetract/stats.json"
    st = json.loads(f.read_text())["benchmark"]
    NUM["corpus"] = {k: st[k] for k in ("n", "n_retracted", "n_control", "n_publishers", "n_venues",
                                       "n_fields", "n_subfields", "year_min", "year_max")}
    if "silo_publisher_retracted" not in st and (ROOT / "data_v2/tabular_v2.parquet").exists():
        sys.path.insert(0, str(ROOT))
        from scripts.v2.b2_main_benchmark import partition
        d = pd.read_parquet(ROOT / "data_v2/tabular_v2.parquet", columns=["publisher", "retracted"])
        st["silo_publisher_retracted"] = d.groupby(partition(d, "publisher")).retracted.sum().astype(int).to_dict()
        full = json.loads(f.read_text())
        full["benchmark"]["silo_publisher_retracted"] = st["silo_publisher_retracted"]
        f.write_text(json.dumps(full, indent=1))
    rows = []
    for s, n in sorted(st["silo_publisher_sizes"].items(), key=lambda kv: -kv[1]):
        k = st["silo_publisher_retracted"][s]
        rows.append([esc(s), f"{n:,}", f"{k:,}", f"{k / n:.3f}"])
    rows.append(["\\textbf{Total}", f"{st['n']:,}", f"{st['n_retracted']:,}", f"{st['n_retracted'] / st['n']:.3f}"])
    table("a_silos", "FedRetract benchmark: the ten publisher silos ($K{=}10$; the 84 smaller publishers form "
          "\\emph{Other}). Controls are matched on publisher and publication year.",
          ["Silo", "Articles", "Retracted", "Retraction share"], rows, "tab:silos", group_breaks=[len(rows) - 1])


def t_leak():
    rows = []
    for src, f in [("Abstract corpus", "d1_leak_audit_abstracts.txt"), ("Full-text corpus", "d1_leak_audit_fulltext.txt")]:
        txt = (RES / f).read_text(encoding="utf-8", errors="replace")
        base = re.search(r"positive rate=([\d.]+)", txt).group(1)
        for field in re.findall(r"^\[([^\]]+)\] keyword-only", txt, re.M):
            kw = re.search(rf"^\[{re.escape(field)}\] keyword-only classifier: \{{'auprc': ([\d.]+), 'roc_auc': ([\d.]+)", txt, re.M)
            tf = re.search(rf"^\[{re.escape(field)}\] TF-IDF logistic regression \(5-fold\): \{{'auprc': ([\d.]+), 'roc_auc': ([\d.]+)", txt, re.M)
            rows.append([src, field, base, f"{float(kw.group(1)):.3f}", f"{float(kw.group(2)):.3f}",
                         f"{float(tf.group(1)):.3f}" if tf else "--", f"{float(tf.group(2)):.3f}" if tf else "--"])
            NUM.setdefault("leak", {})[f"{src}|{field}"] = {"kw_auprc": float(kw.group(1)), "kw_auc": float(kw.group(2)),
                                                           "tfidf_auprc": float(tf.group(1)) if tf else None}
    table("a_leak", "Leakage audit after cleaning. A classifier that may only use notice vocabulary "
          "(retract*, withdraw*, erratum, corrigendum, expression of concern; \\emph{Kw}) is at the base rate; "
          "a TF-IDF bag of words still separates the classes (AUPRC 0.66 on title + abstract), largely through topic "
          "vocabulary (the topic-only baseline is in Section~\\ref{sec:topic}). Kw = keyword-only; Base = base rate.",
          ["Corpus", "Field", "Base", "Kw AUPRC", "Kw ROC", "TF-IDF AUPRC", "TF-IDF ROC"], rows, "tab:leak",
          align="llrrrrr")


def t_b2():
    b = pd.read_csv(RES / "b2_main/results.csv")
    p = pooled_rows(b)
    order = ["central_logreg", "central_rf", "central_xgb", "central_mlp", "local_mlp", "local_xgb",
             "fl_fedavg", "fl_fedprox", "fl_scaffold", "fl_fedbal", "fl_xgb_cyclic"]
    for setting, cap in [("pub", "screening at publication (no citation features; main setting)"),
                         ("pub2", "screening two years after publication (+ citations in years $y..y{+}2$; subset)"),
                         ("v1leak", "the earlier feature set with 2026 citation counts (temporal leakage, for reference only)")]:
        rows = []
        for mdl in order:
            r = [MODEL[mdl]]
            for part in ["publisher", "field"]:
                d = p[(p.setting == setting) & (p.partition == part) & (p.model == mdl)]
                r += [ms(d.auprc), m(d.roc_auc), m(d.recall_at_5fpr)]
            rows.append(r)
        base = b[(b.setting == setting) & (b.view == "pooled") & (b.silo == "ALL")].pos_rate.iloc[0]
        table(f"a_b2_{setting}", f"Tabular benchmark, {cap}. Pooled test set, 10 seeds; base rate {base:.3f}. "
              "LogReg = logistic regression; RF = random forest; XGBoost = gradient-boosted trees; MLP = multilayer "
              "perceptron; R@5\\% = recall at 5\\% false-positive rate. Local models are scored on the pooled test set (each silo's model, averaged).",
              ["Model", "AUPRC", "ROC-AUC", "R@5\\%", "AUPRC", "ROC-AUC", "R@5\\%"], rows,
              f"tab:b2_{setting}", group_breaks=[4, 6],
              spanner=" & \\multicolumn{3}{c}{Publisher silos} & \\multicolumn{3}{c}{Research-field silos}\\\\ "
                      "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}")
        NUM.setdefault("b2", {})[setting] = {f"{part}|{mdl}": round(float(p[(p.setting == setting) & (p.partition == part)
                                                                         & (p.model == mdl)].auprc.mean()), 4)
                                            for part in ["publisher", "field"] for mdl in order}
    # deployment views
    v = b[(b.setting == "pub") & (b.partition == "publisher") & (b.view != "pooled")]
    rows = []
    for mdl in ["central_xgb", "central_mlp", "local_mlp", "local_xgb", "fl_fedavg", "fl_fedavg_ft", "fl_fedprox"]:
        own = v[(v.view == "own_silo") & (v.model == mdl)].groupby("seed").auprc.mean()
        out = v[(v.view == "out_silo") & (v.model == mdl)].groupby("seed").auprc.mean()
        rows.append([MODEL[mdl], ms(own), ms(out)])
    table("a_b2_views", "Tabular models in the deployment view (setting \\emph{pub}, publisher silos, 10 seeds): "
          "\\emph{own} = each publisher's model on its own test slice, averaged over silos; "
          "\\emph{other} = a publisher's model on the other publishers' slices.",
          ["Model", "Own-silo AUPRC", "Other-silo AUPRC"], rows, "tab:b2_views")
    # per silo
    o = v[v.view == "own_silo"]
    rows = []
    for s in sorted(o.silo.unique()):
        r = [esc(s)]
        for mdl in ["central_xgb", "local_xgb", "local_mlp", "fl_fedavg", "fl_fedavg_ft"]:
            r.append(m(o[(o.silo == s) & (o.model == mdl)].auprc))
        rows.append(r)
    table("a_b2_persilo", "Own-silo AUPRC per publisher (tabular, setting \\emph{pub}, 10 seeds).",
          ["Silo", "Central XGB", "Local XGB", "Local MLP", "FedAvg", "FedAvg+FT"], rows, "tab:b2_persilo")


def t_b3_b4_b7():
    b3 = pd.read_csv(RES / "b3_ablations/results.csv")
    b3 = b3[(b3.view == "pooled") & (b3.silo == "ALL")]
    lab = {"pub_all": "All features (main)", "pub_meta_only": "Metadata only", "pub_text_only": "Abstract-lexical only",
           "pub_no_year": "Without publication year", "pub_has_abstract": "Articles with an abstract only",
           "pub_silo_onehot": "+ publisher one-hot (central only)", "v1exact": "Earlier feature set incl. 2026 citations"}
    rows = [[lab[e]] + [ms(b3[(b3.exp == e) & (b3.model == mdl)].auprc) for mdl in ["central_xgb", "central_mlp", "fl_fedavg"]]
            for e in lab]
    table("a_b3", "Feature ablations (tabular, publisher silos, pooled AUPRC, 10 seeds).",
          ["Feature set", "Central XGB", "Central MLP", "FedAvg"], rows, "tab:b3")
    NUM["b3"] = {e: {mdl: round(float(b3[(b3.exp == e) & (b3.model == mdl)].auprc.mean()), 4)
                     for mdl in ["central_xgb", "central_mlp", "fl_fedavg"]} for e in lab}

    b4 = pd.read_csv(RES / "b4_scale_dp/results.csv")
    rows = []
    for k in sorted(b4[b4.exp == "ksweep"].k.dropna().unique()):
        d = b4[(b4.exp == "ksweep") & (b4.k == k)]
        rows.append([f"Publisher silos, $K{{=}}{int(k)}$", "--", "--", ms(d.auprc), m(d.roc_auc)])
    d = b4[b4.exp == "journal"]
    rows.append([f"Journal clients ({int(d.n_clients.min())}--{int(d.n_clients.max())}, by split)", "--", "--",
                 ms(d.auprc), m(d.roc_auc)])
    for exp, lab2 in [("dp_publisher", "Client DP, publishers ($K{=}10$)"), ("dp_journal", "Client DP, journals ($q{=}0.1$)")]:
        for sg in sorted(b4[b4.exp == exp].sigma.unique()):
            d = b4[(b4.exp == exp) & (b4.sigma == sg)]
            eps = d.epsilon.mean()
            rows.append([lab2, f"{sg:g}", "$\\infty$" if not np.isfinite(eps) else f"{eps:.1f}", ms(d.auprc), m(d.roc_auc)])
    table("a_b4", "Consortium size and client-level differential privacy (tabular MLP, DP-FedAvg with RDP "
          "accounting, 10 seeds). Publisher clients: all ten every round, $\\delta{=}10^{-3}$ (below $1/K$ for $K{=}10$ clients; a smaller "
          "$\\delta$ only lowers AUPRC further). Journal clients "
          "(every journal with $\\geq$ 60 training articles + one client per publisher for the rest): $q{=}0.1$, "
          "200 rounds, $\\delta{=}10^{-5}$.", ["Setting", "$\\sigma$", "$\\varepsilon$", "AUPRC", "ROC"],
          rows, "tab:b4", align="lrrrr")
    NUM["b4"] = b4.groupby(["exp", "sigma", "k"], dropna=False).agg(auprc=("auprc", "mean"), eps=("epsilon", "mean")).reset_index().round(4).replace({np.inf: "inf"}).to_dict("records")

    b7 = pd.read_csv(RES / "b7_certainty.csv")
    lab = {"pub_all": "All tabular", "pub_all+cert": "All tabular + model certainty", "text_only": "Abstract-lexical",
           "text_only+cert": "Abstract-lexical + model certainty", "cert_only": "Model certainty only"}
    rows = [[lab[f]] + [ms(b7[(b7.features == f) & (b7.model == mdl)].auprc) for mdl in ["central_xgb", "central_mlp", "fl_fedavg"]]
            for f in lab]
    table("a_b7", "Sentence-level certainty from a trained certainty model (pooled AUPRC, 10 seeds). The gain is small "
          "but positive in 10/10 seeds.", ["Features", "Central XGB", "Central MLP", "FedAvg"], rows, "tab:b7")
    NUM["b7"] = {f: {mdl: round(float(b7[(b7.features == f) & (b7.model == mdl)].auprc.mean()), 4)
                     for mdl in ["central_xgb", "central_mlp", "fl_fedavg"]} for f in lab}


def t_b6():
    b6 = pd.read_csv(RES / "b6_cross_corpus.csv")
    rows = []
    h = b6[b6.exp == "T4_his_protocol"]
    for _, r in h.iterrows():
        rows.append([f"Prior RF protocol, {r.arena}", r.features.replace("his_", "").replace("_", " "),
                     f"{r.accuracy:.3f}", f"{r.auprc:.3f}"])
    for _, r in b6[b6.exp == "T4_cv5x3"].iterrows():
        rows.append([f"5$\\times$3 CV, {r.arena}", r.features.replace("his_", "").replace("_", " "),
                     f"{r.accuracy:.3f} $\\pm$ {r.accuracy_sd:.3f}", f"{r.auprc:.3f}"])
    t1 = b6[b6.exp == "T1_to_ours"]
    for tr in ["curated_180", "ours_random_180", "ours_random_464"]:
        d = t1[t1.train == tr]
        rows.append([f"Train {tr.replace('_', ' ')} $\\rightarrow$ our test set", "lexical", "--", ms(d.auprc)])
    t23 = b6[b6.exp == "T23_ours_to_curated"]
    for feat in ["text_only", "meta_only", "all"]:
        for mdl in ["central_xgb", "fl_fedavg"]:
            d = t23[(t23.features == feat) & (t23.model == mdl)]
            rows.append([f"Ours $\\rightarrow$ curated corpus ({MODEL[mdl]})", feat.replace("_", " "),
                         ms(d.accuracy_at_0_5 if "accuracy_at_0_5" in d else d["accuracy_at_0.5"]), ms(d.auprc)])
    table("a_b6", "Comparison with the hand-curated corpus of Usman and Balke (WebSci 2025; ref.~[6] of the paper; "
          "232 retracted + 232 matched articles). "
          "\\emph{all 10} = the ten section-level readability and certainty features of the prior protocol; "
          "\\emph{abstract only} = the same features computed on the abstract. Subsets: all464 = all 464 articles; "
          "clean180 = articles none of whose section features is a repeated placeholder value (such values are "
          "unevenly distributed between the classes in our copy of the corpus and can reveal the label); "
          "audited172 = clean180 articles that passed a manual audit.",
          ["Experiment", "Features", "Accuracy", "AUPRC"], rows, "tab:b6", align="llrr")


def t_c2():
    c2 = pd.concat([pd.read_csv(f) for f in no_smoke(glob.glob(str(RES / "c2_heads/*_s*.csv")))], ignore_index=True)
    own = c2[c2.view == "own_silo"]
    c2 = pd.concat([pooled_rows(c2), c2[(c2.view == "prospective") & (c2.silo == "ALL")], own], ignore_index=True)
    lab = {"abstracts_modernbert": "ModernBERT-large, abstract", "abstracts_qwen3emb8b": "Qwen3-Emb-8B, abstract",
           "fulltext_modernbert_sections": "ModernBERT-large, full-text sections",
           "fulltext_qwen3emb4b_sections": "Qwen3-Emb-4B, full-text sections"}
    rows = []
    for emb, l in lab.items():
        for tab in [False, True]:
            d = c2[(c2.emb == emb) & (c2.with_tab == tab)]
            r = [l + (" + tabular" if tab else "")]
            for mdl, view in [("central_logreg", "pooled"), ("central_mlp", "pooled"), ("fl_fedavg", "pooled"),
                              ("local_mlp", "pooled"), ("fl_fedavg_ft10", "own_silo"), ("central_mlp", "prospective")]:
                dd = d[(d.model == mdl) & (d.view == view)]
                r.append(m(dd.groupby("seed").auprc.mean()) if view == "own_silo" else m(dd.auprc))
            rows.append(r)
    table("a_c2", "Frozen text embeddings + classification heads (10 seeds; mean AUPRC). Abstract corpus base rate "
          "0.29, full-text corpus 0.36. Own = FedAvg + 10 local fine-tuning epochs on each publisher's own slice; "
          "Prosp. = central MLP on the prospective set.",
          ["Embedding", "LogReg", "MLP", "FedAvg", "Local", "Own (FT)", "Prosp."], rows, "tab:c2", align="lrrrrrr")


def t_c3(c3: pd.DataFrame):
    a = c3[c3.silo == "ALL"]
    # main table (also used in the main text)
    fams = ["c3_lora_abstracts", "c3_lora_abstracts_modernbert-large", "c3_lora_abstracts_qwen3-1.7b"]
    NUM["lora"] = {}
    for fam in fams + ["c3_lora_ftabs", "c3_lora", "c3_lora_8192"]:
        d = a[a.fam == fam]
        g = d.groupby(["model", "view"]).agg(auprc=("auprc", "mean"), auprc_sd=("auprc", "std"), roc=("roc_auc", "mean"),
                                            roc_sd=("roc_auc", "std"), r5=("recall_at_5fpr", "mean"), n=("seed", "nunique"))
        NUM["lora"][fam] = {f"{k[0]}|{k[1]}": {kk: (None if pd.isna(vv) else round(float(vv), 4)) for kk, vv in v.items()}
                            for k, v in g.iterrows()}
    def own_macro(fam, mdl, view="own_silo"):
        d = c3[(c3.fam == fam) & (c3.model == mdl) & (c3.view == view)]
        return d.groupby("seed").auprc.mean()

    p4 = RES / "p4_review"
    pa = pd.read_csv(p4 / "prospective_auprc.csv") if (p4 / "prospective_auprc.csv").exists() else None
    oth = pd.read_csv(p4 / "other_publishers.csv") if (p4 / "other_publishers.csv").exists() else None
    one = "\\rlap{$^{\\ddagger}$}"

    def cell(x):
        return ms(x) + (one if x.dropna().size == 1 else "")

    def fmt_d(g):  # ROC-AUC with AUPRC in brackets (mean over seeds)
        return "--" if g.empty else f"{g.roc_auc.mean():.3f} ({g.auprc.mean():.3f})"

    tfidf = NUM["leak"]["Abstract corpus|title+abstract"]["tfidf_auprc"]
    fmt_js = RES / "p3_review/summary.json"
    fmt_only = json.loads(fmt_js.read_text())["format"]["format_only_auprc"] if fmt_js.exists() else np.nan
    rows = []
    for lab, mdl in [("Central (data pooled)", "central_lora"), ("FedAvg", "fl_fedavg_lora")]:
        rows.append([lab] + [cell(a[(a.fam == f) & (a.model == mdl) & (a.view == "pooled")].auprc) for f in fams])
    rows.append(["Single publisher's model$^{\\ast}$"] +
                [cell(a[(a.fam == f) & (a.model == "local_lora") & (a.view == "pooled")].auprc) for f in fams])
    rows.append(["No language model: TF-IDF words / formatting",
                 f"\\multicolumn{{3}}{{c}}{{{tfidf:.3f} / {fmt_only:.3f}}}"])
    for lab, mdl in [("Central (data pooled)", "central_lora"), ("Local only", "local_lora"),
                     ("FedAvg + local FT", "fl_fedavg_lora_ft")]:
        rows.append([lab] + [cell(own_macro(f, mdl)) for f in fams])
    if oth is not None:
        for lab in ["Central (data pooled)", "FedAvg"]:
            key = lab.split(" (")[0]
            rows.append([lab] + [cell(oth[(oth.model == FAM[f]) & (oth.regime == key)].other_auprc) for f in fams])
    for lab, mdl in [("Local only", "local_lora"), ("FedAvg + local FT", "fl_fedavg_lora_ft")]:
        rows.append([lab] + [cell(own_macro(f, mdl, "out_silo")) for f in fams])
    n_ctl = ""
    if pa is not None:
        w = pa[(pa.subset == "with abstract") & (pa.controls == "unweighted")]
        n_ctl = f"{int(w.n_neg.min()):,}--{int(w.n_neg.max()):,}"
        for lab, reg in [("Central (data pooled)", "central"), ("FedAvg", "fedavg")]:
            rows.append([lab] + [fmt_d(w[(w.model == FAM[f]) & (w.regime == reg)]) for f in fams])
    NUM["main_table"] = rows
    table("main_lora", "Abstract-based screening: AUPRC, mean $\\pm$ s.d. over 3 seeds (${}^{\\ddagger}$\\,one seed); "
          "base rate 0.290 (metadata-only XGBoost 0.512, topic only 0.455). (a) Pooled test set ($^{\\ast}$each "
          "publisher's model, averaged; no-LM baselines: 5-fold CV); (b) mean over publishers of the AUPRC on their own "
          "test articles; (c) the same on the other nine publishers' articles; (d) 61 later retractions with an "
          f"abstract vs.\\ {n_ctl} same-publisher controls with an abstract (base rate 0.014).",
          ["Training regime", "ModernBERT-base", "ModernBERT-large", "Qwen3-1.7B"], rows, "tab:main",
          section_rows={0: "(a) Pooled test set", 4: "(b) Each publisher scores its own articles",
                        7: "(c) Other publishers' articles",
                        7 + (2 if oth is not None else 0) + 2: "(d) Later retractions with an abstract: ROC-AUC (AUPRC)"},
          place="t")

    # appendix: every family x model x view
    rows = []
    breaks = []
    for fam in ["c3_lora_abstracts", "c3_lora_abstracts_modernbert-large", "c3_lora_abstracts_qwen3-1.7b",
                "c3_lora_ftabs", "c3_lora", "c3_lora_8192"]:
        d = a[a.fam == fam]
        breaks.append(len(rows))
        for (mdl, view), g in d.groupby(["model", "view"], sort=True):
            if mdl.startswith("dpsgd") or mdl == "fl_dp_lora":
                continue
            vlab = "pooled (silo avg.)" if view == "pooled" and mdl in ("local_lora", "fl_fedavg_lora_ft") else view.replace("_", " ")
            rows.append([FAM[fam], MODEL[mdl], vlab, ms(g.auprc), ms(g.roc_auc), m(g.recall_at_5fpr),
                         str(g.seed.nunique())])
    table("a_c3_all", "All LoRA fine-tuning results (pooled views; \\emph{pooled personal} = each publisher's own "
          "model on its own slice, scores pooled; \\emph{prospective} = articles retracted after the snapshot). "
          "Abstract corpus base rate 0.290; open-access (OA) full-text subset 0.359 ($K{=}8$ publisher silos); the OA "
          "subset's prospective view has only 29 articles (15 retracted) and is not interpreted. "
          "R@5\\% = recall at 5\\% false-positive rate; n = seeds; FT = local fine-tuning; \\emph{(silo avg.)} = "
          "each publisher's model scored on the whole pooled test set, averaged over publishers.",
          ["Model / corpus", "Regime", "View", "AUPRC", "ROC-AUC", "R@5\\%", "n"], rows, "tab:c3_all",
          align="lllrrrr", size="\\scriptsize", group_breaks=breaks)

    # per silo, abstracts, all three backbones
    o = c3[(c3.view == "own_silo") & c3.fam.isin(fams)]
    rows = []
    for s in sorted(o.silo.unique()):
        r = [esc(s)]
        for fam in fams:
            for mdl in ["central_lora", "fl_fedavg_lora", "local_lora", "fl_fedavg_lora_ft"]:
                r.append(m(o[(o.fam == fam) & (o.silo == s) & (o.model == mdl)].auprc))
        rows.append(r)
    table("a_c3_persilo", "Own-silo AUPRC per publisher for abstract LoRA (C = central, F = FedAvg, L = local only, "
          "P = FedAvg + local fine-tuning; mean over seeds).",
          ["Silo"] + [f"{x}" for x in ["C", "F", "L", "P"] * 3], rows, "tab:c3_persilo",
          align="l" + "r" * 12, size="\\scriptsize",
          spanner=" & \\multicolumn{4}{c}{ModernBERT-base} & \\multicolumn{4}{c}{ModernBERT-large} & "
                  "\\multicolumn{4}{c}{Qwen3-1.7B}\\\\ \\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}")
    # personalization
    d = a[(a.fam == "c3_lora_abstracts")]
    rows = []
    for mdl in ["fl_fedavg_lora_ft", "fl_ditto_lora", "local_lora", "fl_fedper_lora"]:
        g = d[(d.model == mdl) & (d.view == "pooled_personal")]
        own = c3[(c3.fam == "c3_lora_abstracts") & (c3.model == mdl) & (c3.view == "own_silo")].groupby("seed").auprc.mean()
        out = c3[(c3.fam == "c3_lora_abstracts") & (c3.model == mdl) & (c3.view == "out_silo")].groupby("seed").auprc.mean()
        rows.append([MODEL[mdl], ms(g.auprc), ms(g.roc_auc), ms(own), ms(out)])
    table("a_c3_personal", "Personalisation methods (ModernBERT-base, abstracts, 3 seeds). Ditto: $\\lambda{=}0.1$; "
          "FedPer: shared encoder, local classification heads.",
          ["Method", "Pooled-personal AUPRC", "ROC-AUC", "Own-silo (mean)", "Other-silo (mean)"], rows, "tab:c3_personal")


def t_privacy(c3: pd.DataFrame):
    a = c3[c3.silo == "ALL"]
    rows = []
    for fam, mdl in [("c3_lora_abstracts", "central_lora"), ("c3_lora_abstracts", "dpsgd_central_lora"),
                     ("c3_lora_abstracts", "fl_fedavg_lora"), ("c3_lora_abstracts", "dpsgd_fedavg_lora"),
                     ("c3_lora", "central_lora"), ("c3_lora", "fl_fedavg_lora"), ("c3_lora", "fl_dp_lora")]:
        d = a[(a.fam == fam) & (a.model == mdl)]
        for key, g in d.groupby(d.run.str.extract(r"(eps\d+|sig[\d.]+)")[0].fillna("--")):
            pooled = g[g.view == "pooled"]
            pro = g[g.view == "prospective"]
            eps = pooled.epsilon.mean() if "epsilon" in pooled and pooled.epsilon.notna().any() else np.nan
            sig = pooled.sigma.mean() if "sigma" in pooled and pooled.sigma.notna().any() else np.nan
            if mdl == "fl_dp_lora":
                sig = float(key.replace("sig", ""))
                eps = client_dp_eps(sig)
            sig_s = "per silo" if mdl == "dpsgd_fedavg_lora" else ("--" if np.isnan(sig) else f"{sig:.3f}")
            rows.append([FAM[fam], MODEL[mdl], sig_s,
                         "$\\infty$" if np.isnan(eps) else f"{eps:.1f}", ms(pooled.auprc), ms(pooled.roc_auc),
                         m(pro.roc_auc), str(pooled.seed.nunique())])
    table("a_dp_lora", "Differential privacy for LoRA fine-tuning. DP-SGD: record-level, exact per-example clipping "
          "($C{=}1$), Poisson sampling (lot 1,024), RDP accounting, $\\delta{=}10^{-5}$; federated DP-SGD gives each "
          "publisher its own $(\\varepsilon,\\delta)$ guarantee. Client-DP FedAvg: clipped, noised model updates over "
          "$K{=}8$ publishers ($\\varepsilon$ from RDP over 8 rounds).",
          ["Model", "Regime", "$\\sigma$", "$\\varepsilon$", "AUPRC", "ROC-AUC", "Prosp. ROC", "n"], rows, "tab:dp_lora",
          align="llrrrrrr", size="\\scriptsize")
    NUM["dp_lora"] = [dict(zip(["fam", "model", "sigma", "eps", "auprc", "roc", "pros_roc", "n"], r)) for r in rows]

    rec = pd.read_csv(RES / "b11_dp_heads/record.csv")
    rows = []
    for mdl in ["central_dpsgd_linear", "fl_dpsgd_linear"]:
        for eps in [np.inf, -1.0, 8.0, 4.0, 2.0, 1.0, 0.5]:
            d = rec[(rec.model == mdl) & (rec.epsilon == eps)]
            lab = "no clipping, no noise" if eps == np.inf else ("clipping only" if eps == -1 else f"{eps:g}")
            rows.append([MODEL[mdl], m(d.sigma, 2) if eps not in (np.inf, -1.0) else "0", lab, ms(d.auprc),
                         m(d.roc_auc), m(d.mia_auc), m(d.mia_adv)])
    table("a_dp_heads", "Record-level DP-SGD on a linear head over frozen ModernBERT-large abstract embeddings "
          "(federated PCA-128) + tabular features (5 seeds). Federated: per-publisher DP-SGD, FedAvg of the heads. "
          "$\\sigma$ = noise multiplier; clip norm $C{=}1$; \\emph{clipping only} = clipped per-example gradients "
          "without noise, the reference for the cost of the noise. MIA = loss-threshold membership inference on 5,000 "
          "training vs.\\ 5,000 test articles; MIA adv. = membership advantage, max(TPR $-$ FPR).",
          ["Regime", "$\\sigma$", "$\\varepsilon$", "AUPRC", "ROC", "MIA AUC", "MIA adv."], rows, "tab:dp_heads",
          align="llrrrrr", group_breaks=[7])
    NUM["dp_heads"] = rec.groupby(["model", "epsilon"]).auprc.mean().round(4).reset_index().replace({np.inf: "inf"}).to_dict("records")

    j = pd.read_csv(RES / "b11_dp_heads/journal.csv")
    rows = []
    for sg in sorted(j.sigma.unique()):
        d = j[j.sigma == sg]
        e = d.epsilon.mean()
        rows.append([f"{sg:g}", "$\\infty$" if not np.isfinite(e) else f"{e:.1f}", ms(d.auprc), m(d.roc_auc), m(d.mia_auc)])
    table("a_dp_journal", f"Client-level DP-FedAvg with journal clients ({int(j.n_clients.min())}--{int(j.n_clients.max())} "
          "clients depending on the split: every "
          "journal with $\\geq$ 60 training articles, plus one client per publisher for its remaining articles; "
          "sampling rate $q{=}0.1$, 200 rounds) on the same frozen-embedding features "
          "(5 seeds, $\\delta{=}10^{-5}$; the 10 seeds of Table~\\ref{tab:b4} give 222--232 clients). Updates are "
          "clipped to norm 1 in every row; $\\sigma{=}0$ = clipping without noise.", ["$\\sigma$", "$\\varepsilon$", "AUPRC", "ROC", "MIA AUC"], rows,
          "tab:dp_journal", align="rrrrr")
    NUM["dp_journal"] = j.groupby("sigma").agg(eps=("epsilon", "mean"), auprc=("auprc", "mean")).round(4).reset_index().replace({np.inf: "inf"}).to_dict("records")

    mia = pd.read_csv(RES / "b11_dp_heads/mia_lora.csv")
    mia = mia[~mia.run.str.contains("smoke")].copy()
    mia["fam"] = mia.run.str.split("/").str[0]
    mia["kind"] = mia.run.str.split("/").str[1].str.replace("scores_", "", regex=False).str.replace(r"_s4\d", "", regex=True)
    rows = []
    for (fam, kind), g in mia.groupby(["fam", "kind"]):
        kl = {"central": "Central", "fedavg": "FedAvg", "ditto": "Ditto", "local": "Local only",
              "dpsgd_central_eps3": "DP-SGD central, $\\varepsilon{=}3$",
              "dpsgd_central_eps8": "DP-SGD central, $\\varepsilon{=}8$",
              "dpsgd_fedavg_eps8": "DP-SGD federated, $\\varepsilon{=}8$"}.get(kind, kind.replace("_", " "))
        rows.append([FAM.get(fam, fam), kl, ms(g.mia_auc), ms(g.mia_adv), str(len(g))])
    table("a_mia", "Loss-threshold membership inference against the LoRA models (5,000 training vs.\\ 5,000 test "
          "articles; AUC 0.5 = no leakage; advantage = max TPR $-$ FPR).",
          ["Model", "Regime", "MIA AUC", "Advantage", "n"], rows, "tab:mia", align="llrrr")
    NUM["mia"] = {f"{r[0]}|{r[1]}": r[2] for r in rows}


def t_topic():
    s = pd.read_csv(RES / "b8_topic_summary.csv")
    c = pd.read_csv(RES / "b8_topic_concentration.csv")
    order = ["topic_only/logreg", "topic_only/xgb", "tabular/central_xgb", "tabular/central_mlp", "tabular/fl_fedavg"] + \
        sorted(x for x in s.model.unique() if x.startswith("c3_"))
    rows = []
    for mdl in order:
        d, dc = s[s.model == mdl], c[c.model == mdl]
        fam, _, reg = mdl.partition("/")
        fam = {"topic_only": "Topic only", "tabular": "Tabular"}.get(fam, fam)
        reg = {"logreg": "LogReg", "xgb": "XGBoost"}.get(reg, reg)
        name = (FAM.get(fam, fam.replace("_", " ")) + ", " + MODEL.get(reg.replace("__personal", ""), reg).replace("_", " ")
                + (" (personal)" if "__personal" in reg else ""))
        rows.append([name, m(d.auprc), m(d.roc_auc), m(d.within_auc), m(d.within_lift, 2),
                     m(dc.top5_share_of_flagged), m(dc.top5_share_of_false_alarms), str(d.seed.nunique())])
    table("a_topic", "Topic confound. Topic-only = one-hot OpenAlex field + subfield. Within-subfield ROC-AUC is the "
          "size-weighted mean over the 81 subfields with $\\geq$ 20 test articles of each class. Top-5 share = share of "
          "flags (at 5\\% FPR) falling in the five subfields with most retractions, which hold 25\\% of retracted and "
          "12\\% of control test articles; Top-5 FP = the same share among false positives. Lift = within-subfield "
          "AUPRC divided by the subfield's base rate (positive-weighted mean).",
          ["Model", "AUPRC", "ROC", "Within ROC", "Lift", "Top-5 flags", "Top-5 FP", "n"], rows, "tab:topic",
          align="lrrrrrrr", size="\\scriptsize", group_breaks=[2, 5])
    NUM["topic"] = {r[0]: {"auprc": r[1], "within": r[3], "top5_flags": r[5]} for r in rows}


def t_deploy():
    out = {}
    # text models (b10) and tabular (b5)
    p10 = pd.read_csv(RES / "b10_prevalence.csv")
    p5 = pd.read_csv(RES / "b5_prevalence.csv")
    p5["run"] = "tabular"
    p = pd.concat([p5, p10], ignore_index=True)
    rows = []
    for (run, mdl), g in p.groupby(["run", "model"], sort=False):
        name = FAM.get(run, "Tabular") + ", " + MODEL.get(mdl.replace("__personal", ""), mdl) + (" (pers.)" if "__personal" in mdl else "")
        r = [name]
        for pi in ["0.05", "0.01", "0.002"]:
            r += [m(g[f"prec@rec0.5_pi{pi}"]), m(g[f"prec@top1%_pi{pi}"])]
        rows.append(r)
        out[name] = {c: round(float(g[c].mean()), 4) for c in g.columns if c.startswith("prec@")}
    table("a_prev", "Precision under realistic retraction prevalence $\\pi$ (test scores re-weighted to $\\pi$). "
          "P@R50 = precision at 50\\% recall; P@1\\% = precision among the top 1\\% of articles. "
          "$\\pi{=}0.2\\%$ is close to the share of all papers that are retracted; 1\\% and 5\\% correspond to "
          "high-risk journals.",
          ["Model", "P@R50", "P@1\\%", "P@R50", "P@1\\%", "P@R50", "P@1\\%"], rows, "tab:prev",
          align="lrrrrrr", size="\\scriptsize",
          spanner=" & \\multicolumn{2}{c}{$\\pi=5\\%$} & \\multicolumn{2}{c}{$\\pi=1\\%$} & "
                  "\\multicolumn{2}{c}{$\\pi=0.2\\%$}\\\\")
    NUM["prevalence"] = out
    # country
    f10 = pd.read_csv(RES / "b10_fairness_country.csv")
    f5 = pd.read_csv(RES / "b5_fairness_country.csv")
    f5["run"] = "tabular"
    f = pd.concat([f5, f10], ignore_index=True)
    cs = ["CN", "IN", "IR", "KR", "JP", "IT", "US", "DE", "GB"]
    rows = []
    for (run, mdl), g in f.groupby(["run", "model"], sort=False):
        name = FAM.get(run, "Tabular") + ", " + MODEL.get(mdl.replace("__personal", ""), mdl) + (" (pers.)" if "__personal" in mdl else "")
        rows.append([name] + [m(g[g.country == c].fpr) for c in cs])
    table("a_country", "False-positive rate by first-author country at a threshold giving 5\\% overall FPR (control articles). "
          "CN = China, IN = India, IR = Iran, KR = South Korea, JP = Japan, IT = Italy, US = United States, "
          "DE = Germany, GB = United Kingdom; pers. = each publisher scores its own articles.",
          ["Model"] + cs, rows, "tab:country", align="l" + "r" * len(cs), size="\\scriptsize")
    # reasons
    r10 = pd.read_csv(RES / "b10_reasons.csv")
    r5 = pd.read_csv(RES / "b5_reasons.csv")
    r5["run"] = "tabular"
    rr = pd.concat([r5, r10], ignore_index=True)
    reasons = ["paper mill", "fake peer review", "image issues", "data/results errors", "plagiarism/duplication",
               "fabrication/falsification", "authorship/ethics", "other/unspecified"]
    rows = []
    for (run, mdl), g in rr.groupby(["run", "model"], sort=False):
        name = FAM.get(run, "Tabular") + ", " + MODEL.get(mdl.replace("__personal", ""), mdl) + (" (pers.)" if "__personal" in mdl else "")
        rows.append([name] + [m(g[g.reason == x]["recall@5%fpr"], 2) for x in reasons])
    table("a_reasons", "Recall at 5\\% FPR by Retraction Watch reason category. Mill = paper mill; Fake PR = fake peer review; "
          "Image = image problems; Errors = errors in data or results; Plag. = plagiarism or duplication; "
          "Fabr. = fabrication or falsification; Auth. = authorship or ethics; pers. = each publisher scores its own "
          "articles.",
          ["Model", "Mill", "Fake PR", "Image", "Errors", "Plag.", "Fabr.", "Auth.", "Other"], rows, "tab:reasons",
          align="l" + "r" * 8, size="\\scriptsize")
    # prospective tabular
    pr = pd.read_csv(RES / "b5_prospective.csv")
    NUM["prospective_tabular"] = pr.groupby("model")[["auprc", "roc_auc"]].mean().round(4).to_dict()


def t_compute(c3: pd.DataFrame):
    g = c3.drop_duplicates(["fam", "run"]).groupby("fam").agg(gpu_hours=("gpu_hours", "sum"), runs=("run", "nunique"))
    rows = [[FAM.get(k, k), str(int(v.runs)), f"{v.gpu_hours:.1f}"] for k, v in g.iterrows()]
    table("a_compute", "GPU time of the LoRA runs (NVIDIA Blackwell, one GPU per run). Embedding extraction and the "
          "CPU experiments are not included.", ["Model / corpus", "Runs", "GPU-hours"], rows, "tab:compute")
    NUM["gpu_hours_lora"] = round(float(g.gpu_hours.sum()), 1)


def t_review():
    """Robustness checks of scripts/v2/p3_review_checks.py."""
    R = RES / "p3_review"
    if not R.exists():
        return
    pm = pd.read_csv(R / "prospective_matched.csv")
    P4 = RES / "p4_review/prospective_auprc.csv"
    if P4.exists():
        pp = pd.read_csv(P4)
        rows = []
        for (mdl, reg, sub, wt), g in pp.groupby(["model", "regime", "subset", "controls"], sort=False):
            c = g[g.seed == 42].iloc[0]
            rows.append([mdl, {"central": "Central", "fedavg": "FedAvg"}[reg], sub, wt,
                         f"{int(g.n_pos.iloc[0])} / {int(g.n_neg.mean()):,}", f"{g.base_rate.mean():.3f}",
                         f"{g.roc_auc.mean():.3f} [{c.roc_lo:.2f}, {c.roc_hi:.2f}]",
                         f"{g.auprc.mean():.3f} [{c.auprc_lo:.2f}, {c.auprc_hi:.2f}]"])
        dg = pd.read_csv(RES / "p4_review/prospective_title_only_diagnostics.csv")
        d_t, d_a = dg[dg.subset == "title only"], dg[dg.subset == "with abstract"]
        table("a_p3_prosp", "Later-retraction test with matched inputs. 152 of the 213 articles retracted after "
              "19 July 2026 have no abstract in OpenAlex and are scored on the title; controls are the test articles of "
              "the same publishers (their number varies with the split; mean over seeds shown). \\emph{Year-matched}: "
              "controls re-weighted so that their publication years match those of the positives. Values: mean over 3 "
              "seeds; 95\\% bootstrap intervals from seed 42. Title-only positives are mostly recent Elsevier articles "
              f"({d_t.pos_share_elsevier.mean() * 100:.0f}\\% Elsevier vs.\\ {d_t.ctl_share_elsevier.mean() * 100:.0f}\\% of "
              f"title-only controls; median publication year {d_t.pos_median_year.mean():.0f} vs.\\ "
              f"{d_t.ctl_median_year.mean():.0f}), which partly explains their higher scores; within Elsevier alone the "
              f"central models reach ROC-AUC {d_t.roc_elsevier_only.min():.2f}--{d_t.roc_elsevier_only.max():.2f} (title "
              f"only) and {d_a.roc_elsevier_only.min():.2f}--{d_a.roc_elsevier_only.max():.2f} (with abstract).",
              ["Model", "Regime", "Input", "Controls", "Pos / neg", "Base", "ROC-AUC [95\\% CI]", "AUPRC [95\\% CI]"], rows,
              "tab:p3_prosp", align="llllrrrr", size="\\scriptsize", group_breaks=list(range(0, len(rows), 4)))
        NUM["prospective_auprc"] = pp.groupby(["model", "regime", "subset", "controls"])[["roc_auc", "auprc", "base_rate"]].mean().round(3).reset_index().to_dict("records")
    comp = pd.read_csv(R / "prospective_composition.csv")
    NUM["prospective_composition"] = comp.head(6).to_dict("records")
    NUM["prospective_matched"] = pm.groupby(["model", "regime", "subset"])[["roc_auc", "auprc"]].mean().round(3).reset_index().to_dict("records")

    wj = pd.read_csv(R / "within_journal.csv")
    lab = {"central_lora": "Central", "fl_fedavg_lora": "FedAvg", "fl_fedavg_lora_ft__personal": "FedAvg + local FT (own)",
           "local_lora__personal": "Local only (own)"}
    rows = []
    for (mdl, sc), g in wj.groupby(["model", "scores"], sort=False):
        j, sf = g[g.group == "journal"], g[g.group == "subfield"]
        rows.append([mdl, lab[sc], m(g.pooled_auc), m(j.within_auc), f"{j.n_groups.mean():.0f}", f"{j.pos_covered.mean():.2f}",
                     m(sf.within_auc)])
    table("a_p3_journal", "Within-journal ROC-AUC: the size-weighted mean over journals with $\\geq$5 retracted and "
          "$\\geq$5 control test articles (mean over seeds), next to the within-subfield value computed the same way. "
          "A journal shortcut would make the within-journal value fall towards 0.5. Pos. covered = share of the "
          "retracted test articles that lie in the journals used. This table requires $\\geq$5 articles of each class "
          "per group; the topic table (Section~\\ref{sec:topic}) requires $\\geq$20.",
          ["Model", "Scores", "Pooled ROC", "Within journal", "Journals", "Pos. covered", "Within subfield"], rows,
          "tab:p3_journal", align="llrrrrr", size="\\scriptsize")
    NUM["within_journal"] = wj.groupby(["model", "scores", "group"]).within_auc.mean().round(3).reset_index().to_dict("records")

    pr = pd.read_csv(R / "publisher_prior.csv")
    rows = []
    for (mdl, sc), g in pr.groupby(["model", "scores"], sort=False):
        rows.append([mdl, sc, ms(g.pooled_auprc), ms(g.own_silo_macro_auprc), str(g.seed.nunique())])
    table("a_p3_prior", "Does publisher-specific scoring only add a publisher prior? \\emph{Central + publisher prior} "
          "shifts the central model's log-odds by each publisher's training retraction rate. It does not help, so the "
          "gain of publisher-specific models comes from publisher-specific text patterns, not from base rates. "
          "Pooled = one AUPRC over all test articles; own = mean of the 10 per-publisher AUPRCs.",
          ["Model", "Scores", "Pooled AUPRC", "Own-publisher AUPRC", "n"], rows, "tab:p3_prior",
          align="llrrr", size="\\scriptsize")
    NUM["prior"] = pr.groupby(["model", "scores"])[["pooled_auprc", "own_silo_macro_auprc"]].mean().round(3).reset_index().to_dict("records")

    pa = pd.read_csv(R / "paired_bootstrap.csv")
    rows = []
    for (mdl, vs), g in pa.groupby(["model", "vs"], sort=False):
        sig = int(((g.lo > 0) | (g.hi < 0)).sum())
        rows.append([mdl, vs, f"{g.delta_auprc.mean():+.3f}", f"[{g.lo.mean():+.3f}, {g.hi.mean():+.3f}]",
                     f"{sig}/{len(g)}"])
    table("a_p3_paired", "Paired bootstrap (1,000 resamples of the test articles, same split) of the \\emph{pooled-personal} "
          "AUPRC (each article scored by its own publisher's model; one AUPRC over all test articles) of "
          "FedAvg + local fine-tuning minus each alternative (Table~\\ref{tab:p4_own} uses the metric of Table~1b); CI = mean of the per-seed 95\\% intervals; last column = "
          "seeds whose interval excludes 0.",
          ["Model", "FedAvg + local FT vs.", "$\\Delta$AUPRC", "95\\% CI", "Seeds"], rows, "tab:p3_paired",
          align="llrrr", size="\\scriptsize")
    NUM["paired"] = pa.groupby(["model", "vs"])[["delta_auprc", "lo", "hi"]].mean().round(4).reset_index().to_dict("records")

    js = json.loads((R / "summary.json").read_text())["format"]
    nd = pd.read_csv(R / "near_duplicates.csv")
    rows = [["Formatting-only classifier (lengths, casing, punctuation, markup), 5-fold CV",
             f"AUPRC {js['format_only_auprc']:.3f}, ROC {js['format_only_auc']:.3f}"],
            ["  same, within subfields", f"ROC {js['format_only_within_subfield_auc']:.3f}"],
            ["  strongest features", ", ".join(k.replace("_", " ") for k in list(js["top_features"])[:4])]]
    for mname, v in js["lm_within_format_deciles"].items():
        rows.append([f"{mname} (central): ROC overall / within formatting-score deciles",
                     f"{v['lm_auc']:.3f} / {v['lm_auc_within_format_deciles']:.3f}"])
    hb = js["has_abstract_by_label"]
    rows += [["Articles with an abstract, control / retracted", f"{hb['0']:.3f} / {hb['1']:.3f}"],
             ["Median abstract words, control / retracted",
              f"{js['abstract_words_median_by_label']['0']:.0f} / {js['abstract_words_median_by_label']['1']:.0f}"],
             ["Test abstracts with a near-duplicate (MinHash $J\\geq$0.5) in training, retracted / control",
              f"{nd.neardup_share_retracted.mean():.4f} / {nd.neardup_share_control.mean():.4f}"],
             ["ModernBERT-base central AUPRC, all test / without near-duplicates",
              f"{nd.auprc_all.mean():.3f} / {nd.auprc_without_neardups.mean():.3f}"]]
    table("a_p3_format", "Remaining-artefact checks (abstract corpus, base rate 0.290). Formatting carries some signal, "
          "mostly title style (colons, length, lowercase gene names such as \\emph{miR-}); HTML markup in raw titles is "
          "equally frequent in both classes (4.0\\% vs.\\ 3.7\\%). The language models keep most of their ranking "
          "among articles with near-identical formatting scores.", ["Check", "Result"], rows, "tab:p3_format",
          align="lr", size="\\scriptsize")
    NUM["format"] = js
    NUM["neardup"] = nd.mean(numeric_only=True).round(4).to_dict()


def t_review2():
    """Second-round checks of scripts/v2/p4_review_checks.py."""
    R = RES / "p4_review"
    if not R.exists():
        return
    f = R / "own_paired_bootstrap.csv"
    if f.exists():
        o = pd.read_csv(f)
        rows = []
        for (mdl, vs), g in o.groupby(["model", "vs"], sort=False):
            sig = int((g.lo > 0).sum())
            rows.append([mdl, vs, f"{g.delta_own_auprc.mean():+.3f}", f"[{g.lo.mean():+.3f}, {g.hi.mean():+.3f}]",
                         f"{sig}/{len(g)}", f"{g.publishers_gaining.mean():.1f} / {int(g.publishers.iloc[0])}"])
        table("a_p4_own", "Paired bootstrap on the metric of Table~1b: mean over the 10 publishers of the AUPRC on their "
              "own test articles, FedAvg + local fine-tuning minus each alternative (1,000 resamples drawn within each "
              "publisher and class). CI = mean of the per-seed 95\\% intervals; \\emph{Seeds} = seeds whose interval lies "
              "above 0; \\emph{Publishers} = publishers (mean over seeds) on which FedAvg + local FT is higher.",
              ["Model", "FedAvg + local FT vs.", "$\\Delta$AUPRC", "95\\% CI", "Seeds", "Publishers"], rows,
              "tab:p4_own", align="llrrrr", size="\\scriptsize")
        NUM["own_paired"] = o.groupby(["model", "vs"])[["delta_own_auprc", "lo", "hi", "publishers_gaining"]].mean().round(4).reset_index().to_dict("records")
    f = R / "without_hindawi.csv"
    if f.exists():
        h = pd.read_csv(f)
        rows = []
        for (mdl, sc), g in h.groupby(["model", "scores"], sort=False):
            rows.append([mdl, sc, m(g.pooled_auprc_all), m(g.pooled_auprc_no_hindawi), m(g.roc_no_hindawi),
                         m(g.own_macro_no_hindawi), str(g.seed.nunique())])
        table("a_p4_hindawi", "Results without the Hindawi silo, whose 2022--2023 special-issue retractions are templated "
              f"paper-mill batches (evaluation only; models unchanged; base rate without Hindawi {h.base_rate_no_hindawi.mean():.3f}). "
              "Personalised regimes (FedAvg + local FT, Local only, Ditto, FedPer) score each article with its own "
              "publisher's model. Own = mean over the nine other publishers of the AUPRC on their own articles.",
              ["Model", "Scores", "AUPRC (all)", "AUPRC (no Hindawi)", "ROC (no Hindawi)", "Own (no Hindawi)", "n"], rows,
              "tab:p4_hindawi", align="llrrrrr", size="\\scriptsize")
        NUM["no_hindawi"] = h.groupby(["model", "scores"]).mean(numeric_only=True).round(4).reset_index().to_dict("records")
    f = R / "embedding_splits.csv"
    if f.exists():
        e = pd.read_csv(f)
        lo = e[e.exp.str.startswith("held-out")]
        rows = []
        cols = ["own data only", "other nine publishers", "all ten publishers"]
        for q, g in lo.groupby("silo"):
            r = [esc(q), f"{g.base.mean():.3f}"]
            for c in cols:
                gg = g[g.exp == f"held-out publisher: {c}"]
                r += [m(gg.auprc), m(gg.roc_auc)]
            rows.append(r)
        r = ["\\textbf{Mean}", f"{lo.groupby('silo').base.mean().mean():.3f}"]
        for c in cols:
            gg = lo[lo.exp == f"held-out publisher: {c}"].groupby("silo")[["auprc", "roc_auc"]].mean()
            r += [f"{gg.auprc.mean():.3f}", f"{gg.roc_auc.mean():.3f}"]
        rows.append(r)
        table("a_p4_lopo", "A new consortium member, simulated on frozen Qwen3-Embedding-8B abstract embeddings with a "
              "logistic-regression head (3 seeds; same split as the paper). For each publisher, a head is trained on its "
              "own training articles only, on the other nine publishers' training articles only (the publisher never "
              "contributed data, as for a new member), or on all ten, and scored on the publisher's test articles.",
              ["Publisher", "Base", "AUPRC", "ROC", "AUPRC", "ROC", "AUPRC", "ROC"], rows, "tab:p4_lopo",
              align="lrrrrrrr", size="\\scriptsize", group_breaks=[len(rows) - 1],
              spanner=" & & \\multicolumn{2}{c}{Own data only} & \\multicolumn{2}{c}{Other nine only} & "
                      "\\multicolumn{2}{c}{All ten}\\\\ \\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}")
        NUM["lopo"] = {c: lo[lo.exp == f"held-out publisher: {c}"].groupby("silo")[["auprc", "roc_auc"]].mean().mean().round(4).to_dict()
                       for c in cols}
        sp = e[~e.exp.str.startswith("held-out")]
        lab = {"random split (reference)": "Random split (the paper's split)",
               "journal-grouped: unseen journals": "Journal-grouped: test journals unseen in training",
               "temporal reference: random train, same size": "Random training sample of the same size, same 2022--2025 test",
               "temporal: train <= 2020": "Temporal: train on articles up to 2020, test on 2022--2025"}
        rows = [[lab[k], m(sp[sp.exp == k].base), ms(sp[sp.exp == k].auprc), ms(sp[sp.exp == k].roc_auc)] for k in lab]
        table("a_p4_splits", "Split sensitivity of the same frozen-embedding head (3 seeds). The temporal test set is a random "
              "half of the 2022--2025 articles; 2021 is left out as a gap. The last two rows differ only in the years of "
              "the training articles.", ["Split", "Base", "AUPRC", "ROC-AUC"], rows, "tab:p4_splits", align="lrrr",
              size="\\scriptsize", group_breaks=[2])
        NUM["splits"] = sp.groupby("exp")[["auprc", "roc_auc", "base"]].mean().round(4).to_dict("index")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper/final"))
    a = ap.parse_args()
    OUT = Path(a.out)
    TAB = OUT / "tables"
    TAB.mkdir(parents=True, exist_ok=True)
    c3 = load_c3()
    for fn in [t_silos, t_leak, t_b2, t_b3_b4_b7, t_b6, t_c2, lambda: t_c3(c3), lambda: t_privacy(c3), t_topic,
               t_deploy, lambda: t_compute(c3), t_review, t_review2]:
        fn()
    (OUT / "numbers.json").write_text(json.dumps(NUM, indent=1, default=str), encoding="utf-8")
    print("tables:", sorted(p.name for p in TAB.glob("*.tex")))
    sys.exit(0)
