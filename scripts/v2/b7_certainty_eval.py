"""B7: do model-based certainty features add signal? (M9) Same splits/protocol as B2/B3.
Feature sets: pub_all (v2 lexicon proxies) vs pub_all + model certainty vs text-only
(lexicon) vs text-only + model certainty. Central XGB/MLP and FedAvg, 10 seeds."""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.v2.b2_main_benchmark import META, TEXT, partition, split, xgb_model  # noqa: E402
from src.models.evaluate import evaluate  # noqa: E402
from src.v2 import DATA2, RES2  # noqa: E402

CERT = ["cert_mean", "cert_min", "cert_std", "cert_p10", "cert_low_share"]


def run(seed: int) -> list[dict]:
    import torch

    from src.v2.fl import FedStandardizer, Silo, run_fl, scores, train_central_mlp
    torch.set_num_threads(1)
    df = pd.read_parquet(DATA2 / "tabular_v2.parquet").reset_index(drop=True)
    cert = pd.read_parquet(DATA2 / "certainty_abstracts.parquet")
    assert cert["doi"].tolist() == df["doi"].tolist()
    df = pd.concat([df, cert[CERT]], axis=1)
    df["silo"] = partition(df, "publisher")
    df["part"] = split(df, seed)
    rows = []
    for name, feats in [("pub_all", META + TEXT), ("pub_all+cert", META + TEXT + CERT),
                        ("text_only", TEXT), ("text_only+cert", TEXT + CERT), ("cert_only", CERT)]:
        X = df[feats].to_numpy(np.float64)
        y = df["retracted"].to_numpy(np.float32)
        silos = sorted(df["silo"].unique())
        m = {p: (df["part"] == p).to_numpy() for p in ["train", "val", "test"]}
        sm = {s: (df["silo"] == s).to_numpy() for s in silos}
        Z = FedStandardizer.fit([X[m["train"] & sm[s]] for s in silos]).transform(X)
        Ztr, ytr, Zte, yte = Z[m["train"]], y[m["train"]], Z[m["test"]], y[m["test"]]
        spw = float((ytr == 0).sum() / (ytr == 1).sum())
        rows.append({"seed": seed, "features": name, "model": "central_xgb",
                     **evaluate(yte, xgb_model(seed, spw).fit(Ztr, ytr).predict_proba(Zte)[:, 1])})
        mlp = train_central_mlp(Ztr, ytr, Z[m["val"]], y[m["val"]], Z.shape[1], seed=seed)
        rows.append({"seed": seed, "features": name, "model": "central_mlp", **evaluate(yte, scores(mlp, Zte))})
        objs = [Silo(s, Z[m["train"] & sm[s]], y[m["train"] & sm[s]], Z[m["val"] & sm[s]],
                     y[m["val"] & sm[s]]) for s in silos]
        out = run_fl([o for o in objs if o.n and len(o.yva)], Z.shape[1], "fedavg", rounds=50, seed=seed)
        rows.append({"seed": seed, "features": name, "model": "fl_fedavg",
                     **evaluate(yte, scores(out["model"], Zte))})
    return rows


def main() -> None:
    with ProcessPoolExecutor(max_workers=10) as ex:
        rows = [r for rs in ex.map(run, range(42, 52)) for r in rs]
    r = pd.DataFrame(rows)
    r.to_csv(RES2 / "b7_certainty.csv", index=False)
    print(r.groupby(["features", "model"])["auprc"].agg(["mean", "std"]).round(4).unstack().to_string())


if __name__ == "__main__":
    main()
