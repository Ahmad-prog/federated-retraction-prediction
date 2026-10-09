"""Write results_v2/STATUS.md: pipeline state + headline numbers of everything finished so far.
Regenerated every 30 minutes by status_loop.sh while the user is away."""
from __future__ import annotations

import glob
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.v2 import DATA2, RES2, ROOT  # noqa: E402


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"({e})"


def tbl(df: pd.DataFrame) -> str:
    return "```\n" + df.to_string() + "\n```"


def main() -> None:
    L = [f"# v2 run status — {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC", ""]
    L += ["## Pipeline", "```", sh("tail -4 logs/phaseA.log"), sh("tail -3 logs/data_chain.log"),
          "FT_READY: " + str((DATA2 / "FT_READY").exists()), "```", "",
          "## GPU queue (GPU 1)", "```", sh("tail -12 logs/gpu_queue.log"),
          "pending: " + sh("ls queue/pending | tr '\\n' ' '"),
          "failed: " + sh("ls queue/failed | tr '\\n' ' '"),
          sh("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n 2p"),
          "```", ""]
    gq = sh("grep -E 'START|DONE|FAIL' logs/gpu_queue.log")
    L += [f"GPU jobs started: {gq.count('START')}, done: {gq.count('DONE')}, failed: {gq.count('FAIL')}", ""]

    f = RES2 / "b2_main" / "results.csv"
    if f.exists():
        r = pd.read_csv(f)
        p = r[(r.view == "pooled") & (r.silo == "ALL")].groupby(["setting", "partition", "model"])["auprc"].agg(["mean", "std"]).round(4)
        L += ["## B2 main benchmark (pooled test, AUPRC, 10 seeds)", tbl(p), ""]
        o = r[r.view == "own_silo"].groupby(["setting", "partition", "model", "seed"])["auprc"].mean() \
            .groupby(["setting", "partition", "model"]).mean().round(4).unstack(level=2)
        L += ["## B2 own-silo view (mean over silos)", tbl(o), ""]
    for name, path, keys in [("B3 ablations", RES2 / "b3_ablations" / "results.csv", ["exp", "model"]),
                             ("B4 scale & DP", RES2 / "b4_scale_dp" / "results.csv",
                              ["exp", "k", "q", "sigma", "epsilon"])]:
        if path.exists():
            r = pd.read_csv(path)
            if "view" in r:
                r = r[r.view == "pooled"].groupby(keys + ["seed"])["auprc"].mean().reset_index()
            keys = [k for k in keys if k in r]
            L += [f"## {name} (pooled AUPRC)", tbl(r.groupby(keys, dropna=False)["auprc"].agg(["mean", "std", "count"]).round(4)), ""]
    for name, path in [("B5 prospective", RES2 / "b5_prospective.csv"), ("B6 cross-corpus", RES2 / "b6_cross_corpus.csv")]:
        if path.exists():
            r = pd.read_csv(path)
            keys = [c for c in ["exp", "arena", "features", "train", "model"] if c in r and r[c].notna().any()]
            cols = [c for c in ["accuracy", "accuracy_at_0.5", "auprc", "roc_auc"] if c in r]
            L += [f"## {name}", tbl(r.groupby(keys, dropna=False)[cols].mean().round(3)), ""]
    heads = sorted(glob.glob(str(RES2 / "c2_heads" / "*.csv")))
    if heads:
        r = pd.concat([pd.read_csv(h) for h in heads])
        r = r[r.view.isin(["pooled", "prospective"])]
        r = r[(r.model != "local_mlp")]
        L += ["## C2 embedding heads (AUPRC)", tbl(r.groupby(["source", "emb", "with_tab", "view", "model"])["auprc"].agg(["mean", "std", "count"]).round(4)), ""]
    for sub in ["c3_lora_abstracts", "c3_lora"]:
        fs = sorted(glob.glob(str(RES2 / sub / "results_*.csv")))
        fs = [x for x in fs if "smoke" not in x]
        if fs:
            r = pd.concat([pd.read_csv(x) for x in fs])
            v = r[r.view.isin(["pooled", "prospective"])]
            L += [f"## {sub} (LoRA fine-tuning; local_lora pooled = mean over silo models)",
                  tbl(v.groupby(["model", "view"])[["auprc", "roc_auc"]].agg(["mean", "count"]).round(4)), ""]
            o = r[r.view == "own_silo"].groupby(["model", "seed"])["auprc"].mean().groupby("model").mean().round(4)
            L += ["own-silo view:", tbl(o.to_frame()), ""]
    for audit in ["d1_leak_audit_abstracts.txt", "d1_leak_audit_fulltext.txt"]:
        a = RES2 / audit
        if a.exists():
            lines = [x for x in a.read_text(encoding="utf-8").splitlines()
                     if "keyword-only" in x or "TF-IDF" in x or "notice-like" in x]
            L += [f"## {audit}", "```", *[x[:300] for x in lines], "```", ""]
    (RES2 / "STATUS.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    import os
    os.chdir(ROOT)
    main()
