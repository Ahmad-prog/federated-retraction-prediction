"""B11: privacy that can work for text (C9, R2.7) + membership-inference audit.

Frozen ModernBERT-large abstract embeddings are a public, per-record feature map, so a
private model only has to protect the small head trained on top of them.
Features: federated PCA (per-silo sums of x and x x^T only) of the 1024-d embedding to
PCA_DIM dims + the tabular features (META + TEXT), standardized federatedly.

  record   record-level DP-SGD (per-example clipping, Poisson sampling, RDP accounting),
           linear head, central and federated (each publisher runs DP-SGD locally;
           epsilon is per publisher, protecting each article), eps in EPS
  journal  client-level DP-FedAvg with journal-level clients (q = 0.1), same sigma grid
           and accounting as B4 (tabular-only) -> direct comparison
  mia      loss-threshold membership inference (members: 5,000 training articles,
           non-members: 5,000 test articles) for every head above and for the saved
           LoRA runs (c3_lora*/scores_*.npz)
Output: results_v2/b11_dp_heads/*.csv
"""
from __future__ import annotations

import argparse
import glob
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.v2.b2_main_benchmark import META, TEXT, partition, split  # noqa: E402
from src.models.evaluate import evaluate  # noqa: E402
from src.v2 import DATA2, RES2  # noqa: E402

OUT = RES2 / "b11_dp_heads"
PCA_DIM = 128
EPS = [0.5, 1.0, 2.0, 4.0, 8.0, -1.0, float("inf")]  # -1: clipping only, no noise (isolates the noise cost)
MIN_JOURNAL = 60
DELTA = 1e-5


def bce(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def mia(y_mem, s_mem, y_non, s_non, seed=0):
    """Loss-threshold attack: lower loss -> member. Returns AUC and max advantage."""
    rng = np.random.default_rng(seed)
    n = min(len(y_mem), len(y_non))
    i, j = rng.choice(len(y_mem), n, replace=False), rng.choice(len(y_non), n, replace=False)
    score = -np.r_[bce(y_mem[i], s_mem[i]), bce(y_non[j], s_non[j])]
    lab = np.r_[np.ones(n), np.zeros(n)]
    fpr, tpr, _ = roc_curve(lab, score)
    return {"mia_auc": float(roc_auc_score(lab, score)), "mia_adv": float(np.max(tpr - fpr))}


def features(df, m, silo_col="silo"):
    """Federated PCA of the embedding + federated standardization of all features."""
    E = np.load(DATA2 / "emb" / "abstracts_modernbert.npy").astype(np.float64)
    ids = (DATA2 / "emb" / "abstracts_modernbert.ids.txt").read_text().split("\n")
    assert ids == df["doi"].tolist()
    silos = sorted(df[silo_col].unique())
    tr = m["train"]
    n, s1, s2 = 0, np.zeros(E.shape[1]), np.zeros((E.shape[1], E.shape[1]))
    for s in silos:  # each silo shares only count, sum and sum of outer products
        Es = E[tr & (df[silo_col] == s).to_numpy()]
        n, s1, s2 = n + len(Es), s1 + Es.sum(0), s2 + Es.T @ Es
    mu = s1 / n
    cov = s2 / n - np.outer(mu, mu)
    vals, vecs = np.linalg.eigh(cov)
    W = vecs[:, ::-1][:, :PCA_DIM]
    X = np.hstack([(E - mu) @ W, df[META + TEXT].to_numpy(np.float64)])
    from src.v2.fl import FedStandardizer
    std = FedStandardizer.fit([X[tr & (df[silo_col] == s).to_numpy()] for s in silos])
    return std.transform(X)


# ------------------------------------------------------------ record-level DP-SGD (linear)
def dpsgd_linear(X, y, Xva, yva, epochs, eps, seed, lot=1024, clip=1.0, lr=0.05, w0=None, sigma=None):
    """Logistic regression with record-level DP-SGD; returns (w, best val loss).
    eps = inf -> same optimizer without clipping/noise (non-private reference)."""
    import torch
    from opacus.accountants.utils import get_noise_multiplier
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    n, d = X.shape
    q = min(1.0, lot / n)
    steps_per_epoch = max(1, int(round(1 / q)))
    if sigma is None:
        sigma = 0.0 if (np.isinf(eps) or eps < 0) else float(get_noise_multiplier(
            target_epsilon=eps, target_delta=DELTA, sample_rate=q, steps=steps_per_epoch * epochs,
            accountant="rdp"))
    pos_w = float((y == 0).sum() / max(1, (y == 1).sum()))
    Xt, yt = torch.tensor(np.hstack([X, np.ones((n, 1))]), dtype=torch.float32), torch.tensor(y)
    Xv = torch.tensor(np.hstack([Xva, np.ones((len(Xva), 1))]), dtype=torch.float32)
    w = torch.zeros(d + 1) if w0 is None else torch.tensor(w0, dtype=torch.float32)
    w.requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    best = (np.inf, w.detach().clone())
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            b = np.where(rng.random(n) < q)[0]
            if not len(b):
                continue
            xb, yb = Xt[b], yt[b]
            with torch.no_grad():
                p = torch.sigmoid(xb @ w)
                wt = torch.where(yb == 1, pos_w, 1.0)
                G = (wt * (p - yb))[:, None] * xb          # per-example gradients
                if not np.isinf(eps):
                    nrm = G.norm(dim=1, keepdim=True)
                    G = G * torch.clamp(clip / (nrm + 1e-12), max=1.0)
                gsum = G.sum(0) + torch.randn(d + 1, generator=g) * sigma * clip
            w.grad = gsum / (q * n)
            opt.step()
        with torch.no_grad():
            pv = torch.sigmoid(Xv @ w).numpy()
        wv = np.where(yva == 1, pos_w, 1.0)
        vl = float(np.average(bce(yva, pv), weights=wv))
        if vl < best[0]:
            best = (vl, w.detach().clone())
    return best[1].numpy(), best[0], sigma


def predict_linear(w, X):
    return 1 / (1 + np.exp(-(np.hstack([X, np.ones((len(X), 1))]) @ w)))


def run_record(seed: int, epochs: int) -> list[dict]:
    import torch
    torch.set_num_threads(2)
    df = pd.read_parquet(DATA2 / "tabular_v2.parquet").reset_index(drop=True)
    df["silo"] = partition(df, "publisher")
    df["part"] = split(df, seed)
    m = {p: (df["part"] == p).to_numpy() for p in ["train", "val", "test"]}
    Z = features(df, m)
    y = df["retracted"].to_numpy(np.float32)
    silos = sorted(df["silo"].unique())
    sm = {s: (df["silo"] == s).to_numpy() for s in silos}
    rng = np.random.default_rng(seed)
    mem = rng.choice(np.where(m["train"])[0], 5000, replace=False)
    yte = y[m["test"]]
    rows = []
    for eps in EPS:
        # central
        w, _, sig = dpsgd_linear(Z[m["train"]], y[m["train"]], Z[m["val"]], y[m["val"]], epochs, eps, seed)
        st, sm_ = predict_linear(w, Z[m["test"]]), predict_linear(w, Z[mem])
        rows.append({"seed": seed, "exp": "record", "model": "central_dpsgd_linear", "epsilon": eps,
                     "sigma": sig, **evaluate(yte, st), **mia(y[mem], sm_, yte, st, seed)})
        # federated: every silo runs DP-SGD locally for 1 epoch per round
        rounds = epochs
        sizes = np.array([(m["train"] & sm[s]).sum() for s in silos], float)
        wg = np.zeros(Z.shape[1] + 1)
        best = (np.inf, wg)
        sig_k = {s: 0.0 if (np.isinf(eps) or eps < 0) else _sigma(eps, int((m["train"] & sm[s]).sum()), rounds)
                 for s in silos}
        for r in range(rounds):
            locs, vls = [], []
            for k, s in enumerate(silos):
                tr, va = m["train"] & sm[s], m["val"] & sm[s]
                # per-silo budget: `rounds` local epochs in total -> accountant over rounds
                wl, _, _ = dpsgd_linear(Z[tr], y[tr], Z[va], y[va], 1, eps, seed * 1000 + r * 31 + k,
                                        w0=wg, sigma=sig_k[s])
                locs.append(wl)
            wg = np.average(np.stack(locs), axis=0, weights=sizes)
            pv = predict_linear(wg, Z[m["val"]])
            vl = float(bce(y[m["val"]], pv).mean())
            if vl < best[0]:
                best = (vl, wg.copy())
        st, sm_ = predict_linear(best[1], Z[m["test"]]), predict_linear(best[1], Z[mem])
        rows.append({"seed": seed, "exp": "record", "model": "fl_dpsgd_linear", "epsilon": eps,
                     "sigma": float(np.mean(list(sig_k.values()))),
                     **evaluate(yte, st), **mia(y[mem], sm_, yte, st, seed)})
    return rows


def _sigma(eps_total, n, rounds, lot=1024):
    """Noise multiplier giving (eps_total, DELTA) for a silo of n articles that runs one
    local DP-SGD epoch in each of `rounds` rounds (RDP accounting over all its steps)."""
    from opacus.accountants.utils import get_noise_multiplier
    q = min(1.0, lot / n)
    return float(get_noise_multiplier(target_epsilon=eps_total, target_delta=DELTA, sample_rate=q,
                                      steps=max(1, int(round(1 / q))) * rounds, accountant="rdp"))


# ------------------------------------------------------------ client-level DP, journal clients
def run_journal(seed: int, rounds: int) -> list[dict]:
    import torch

    from src.v2.fl import Silo, epsilon_gaussian, run_fl, scores
    torch.set_num_threads(2)
    df = pd.read_parquet(DATA2 / "tabular_v2.parquet").reset_index(drop=True)
    df["silo"] = partition(df, "publisher")
    df["part"] = split(df, seed)
    m = {p: (df["part"] == p).to_numpy() for p in ["train", "val", "test"]}
    tr_counts = df.loc[m["train"]].groupby("venue_id").size()
    big = set(tr_counts[tr_counts >= MIN_JOURNAL].index)
    df["jsilo"] = df["venue_id"].where(df["venue_id"].isin(big),
                                       "rest::" + df["publisher"].fillna("Unknown").astype(str))
    Z = features(df, m, "jsilo")
    y = df["retracted"].to_numpy(np.float32)
    objs = []
    for s in sorted(df["jsilo"].unique()):
        msk = (df["jsilo"] == s).to_numpy()
        o = Silo(str(s), Z[m["train"] & msk], y[m["train"] & msk], Z[m["val"] & msk], y[m["val"] & msk])
        if o.n >= 10:
            if len(o.yva) == 0:
                o.Xva, o.yva = o.Xtr[:5], o.ytr[:5]
                o.__post_init__()
            objs.append(o)
    rng = np.random.default_rng(seed)
    mem = rng.choice(np.where(m["train"])[0], 5000, replace=False)
    yte = y[m["test"]]
    rows = []
    for sigma in [0.0, 0.5, 1.0, 2.0, 4.0]:
        out = run_fl(objs, Z.shape[1], "fedavg", rounds=rounds, seed=seed, client_frac=0.1,
                     dp_sigma=sigma, dp_clip=1.0)
        st, sm_ = scores(out["model"], Z[m["test"]]), scores(out["model"], Z[mem])
        rows.append({"seed": seed, "exp": "journal", "model": "dp_fedavg_emb", "sigma": sigma,
                     "epsilon": epsilon_gaussian(sigma, rounds, DELTA, q=0.1), "n_clients": len(objs),
                     **evaluate(yte, st), **mia(y[mem], sm_, yte, st, seed)})
    return rows


# ------------------------------------------------------------ MIA on saved LoRA runs
def mia_lora() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(RES2 / "c3_lora*" / "scores_*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "mia_idx" not in z:
            continue
        for k in z.files:
            if not k.endswith("__mia_train"):
                continue
            name = k[: -len("__mia_train")]
            st = z.get(f"{name}__ALL__test")
            if st is None:
                continue
            rows.append({"run": Path(f).parent.name + "/" + Path(f).stem, "model": name,
                         **mia(z["y_mia"], z[k], z["y_test"], st)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--only", choices=["record", "journal", "mia", "all"], default="all")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.only in ("mia", "all"):
        r = mia_lora()
        r.to_csv(OUT / "mia_lora.csv", index=False)
        if len(r):
            print(r.groupby("model")[["mia_auc", "mia_adv"]].agg(["mean", "count"]).round(4).to_string())
    for exp, fn, arg in [("record", run_record, a.epochs), ("journal", run_journal, a.rounds)]:
        if a.only not in (exp, "all"):
            continue
        rows = []
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            futs = {ex.submit(fn, 42 + i, arg): 42 + i for i in range(a.seeds)}
            for f in as_completed(futs):
                try:
                    rows.extend(f.result())
                    print(f"{exp} seed {futs[f]} done", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"{exp} seed {futs[f]} FAILED: {e!r}", flush=True)
                pd.DataFrame(rows).to_csv(OUT / f"{exp}.csv", index=False)
        res = pd.DataFrame(rows)
        keys = [c for c in ["model", "epsilon"] if c in res]
        print(res.groupby(keys)[["auprc", "roc_auc", "mia_auc"]].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
