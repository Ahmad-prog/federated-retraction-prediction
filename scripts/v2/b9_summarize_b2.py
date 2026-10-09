"""Summarise B2: headline table, own-silo vs pooled views, paired tests across seeds."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.v2 import RES2  # noqa: E402

res = pd.read_csv(RES2 / "b2_main" / "results.csv")
pd.set_option("display.width", 200)

for setting in ["pub", "pub2", "v1leak"]:
    for part in ["publisher", "field"]:
        r = res[(res.setting == setting) & (res.partition == part)]
        print(f"\n===== setting={setting} partition={part} =====")
        # pooled view: global models on the whole test set; local = mean over silo models
        pooled = r[r.view == "pooled"].groupby(["seed", "model"])["auprc"].mean().unstack()
        print("POOLED test (local_* = mean over silo models):")
        print(pooled.agg(["mean", "std"]).T.round(4).to_string())
        # own-silo view: mean over silos of each model's AUPRC on that silo's slice
        own = r[r.view == "own_silo"].groupby(["seed", "model"])["auprc"].mean().unstack()
        print("OWN-SILO test (mean over silos):")
        print(own.agg(["mean", "std"]).T.round(4).to_string())
        # per-silo wins
        o = r[r.view == "own_silo"].groupby(["model", "silo"])["auprc"].mean().unstack(0)
        for a, b in [("fl_fedavg", "local_xgb"), ("fl_fedavg", "local_mlp"),
                     ("fl_fedavg_ft", "local_mlp"), ("fl_fedavg_ft", "local_xgb"),
                     ("central_mlp", "local_mlp")]:
            if a in o and b in o:
                print(f"  silos where {a} > {b}: {(o[a] > o[b]).sum()}/{len(o)}")
        for a, b in [("fl_fedavg", "central_mlp"), ("fl_fedavg", "local_mlp"),
                     ("fl_fedavg", "fl_fedprox")]:
            if a in pooled and b in pooled:
                t = stats.ttest_rel(pooled[a], pooled[b])
                print(f"  pooled {a} - {b}: {(pooled[a] - pooled[b]).mean():+.4f} "
                      f"(ratio {pooled[a].mean() / pooled[b].mean():.3f}), t={t.statistic:.1f}, p={t.pvalue:.2g}")
        if setting == "pub" and part == "publisher":
            print("\nper-silo own-slice AUPRC (mean over seeds):")
            print(o.round(3).to_string())
