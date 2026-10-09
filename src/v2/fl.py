"""v2 cross-silo FL engine for tabular / embedding features.

Differences from src/fl/engine.py (v1):
  * federated preprocessing: silos share only (count, sum, sum of squares) per feature;
    missing values -> aggregated mean plus a missingness indicator (C12)
  * strategies: fedavg, fedprox, scaffold, fedbal (fixed: focal alpha on the POSITIVE
    class = 1 - silo prevalence, i.e. up-weights the minority; M1)
  * personalisation: fine-tune the global model on each silo's own data (C2, M14)
  * round selection on the federated validation loss (sample-weighted mean of silo
    validation losses), never on the test set (M7)
  * client-level DP-FedAvg (McMahan et al. 2018): clipped updates, uniform weights with a
    fixed denominator, Poisson client sampling; sigma = 0 is the matching non-private
    baseline ("clipped uniform FedAvg"), reported next to plain size-weighted FedAvg (C9)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)


# ---------------------------------------------------------------- preprocessing
@dataclass
class FedStandardizer:
    mean: np.ndarray
    std: np.ndarray
    miss_cols: np.ndarray  # indices of features that have missing values anywhere

    @classmethod
    def fit(cls, silo_X: list[np.ndarray]) -> "FedStandardizer":
        """Each silo contributes only per-feature count, sum and sum of squares."""
        n = sum(np.sum(~np.isnan(X), axis=0) for X in silo_X)
        s = sum(np.nansum(X, axis=0) for X in silo_X)
        ss = sum(np.nansum(X.astype(np.float64) ** 2, axis=0) for X in silo_X)
        mean = s / np.maximum(n, 1)
        var = ss / np.maximum(n, 1) - mean ** 2
        anymiss = sum(np.isnan(X).sum(axis=0) for X in silo_X) > 0
        return cls(mean, np.sqrt(np.maximum(var, 1e-12)), np.where(anymiss)[0])

    def transform(self, X: np.ndarray) -> np.ndarray:
        miss = np.isnan(X)
        Z = (np.where(miss, self.mean, X) - self.mean) / self.std
        ind = miss[:, self.miss_cols].astype(np.float32)
        return np.hstack([Z, ind]).astype(np.float32)


# ---------------------------------------------------------------- model
class MLP(nn.Module):
    def __init__(self, d: int, hidden=(64, 32), dropout: float = 0.2):
        super().__init__()
        layers, prev = [], d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def focal_loss(logits, y, alpha_pos: float, gamma: float = 2.0):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p_t = torch.exp(-bce)
    a = alpha_pos * y + (1 - alpha_pos) * (1 - y)
    return (a * (1 - p_t) ** gamma * bce).mean()


def params(m: nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in m.parameters()]


def set_params(m: nn.Module, ps: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for p, q in zip(m.parameters(), ps):
            p.copy_(q)


@dataclass
class Silo:
    name: str
    Xtr: np.ndarray
    ytr: np.ndarray
    Xva: np.ndarray
    yva: np.ndarray

    def __post_init__(self):
        self.Xtr_t, self.ytr_t = torch.tensor(self.Xtr), torch.tensor(self.ytr)
        self.Xva_t, self.yva_t = torch.tensor(self.Xva), torch.tensor(self.yva)
        self.n = len(self.ytr)
        self.prev = float(self.ytr.mean()) if self.n else 0.0
        # clamp so single-class silos (e.g. a journal with no retractions) still train
        pc = min(max(self.prev, 0.05), 0.95)
        self.pos_weight = torch.tensor((1 - pc) / pc)

    def loss_fn(self, strategy: str):
        if strategy == "fedbal":
            return lambda lo, y: focal_loss(lo, y, alpha_pos=1 - self.prev)
        bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        return bce

    def val_loss(self, model: nn.Module) -> float:
        model.eval()
        with torch.no_grad():
            lo = model(self.Xva_t)
            return float(nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(lo, self.yva_t))


def local_sgd(model, silo: Silo, strategy: str, epochs: int, lr: float, batch: int,
              seed: int, mu: float = 0.0, global_ps=None, c_glob=None, c_loc=None):
    """Train `model` in place on silo data. Returns number of optimizer steps."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0 if strategy == "scaffold" else 0.9)
    lf = silo.loss_fn(strategy)
    steps = 0
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(silo.n, generator=g)
        for i in range(0, silo.n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lf(model(silo.Xtr_t[idx]), silo.ytr_t[idx])
            if strategy == "fedprox" and mu > 0:
                loss = loss + (mu / 2) * sum(((p - q) ** 2).sum()
                                             for p, q in zip(model.parameters(), global_ps))
            loss.backward()
            if strategy == "scaffold":
                with torch.no_grad():
                    for p, cg, cl in zip(model.parameters(), c_glob, c_loc):
                        p.grad += cg - cl
            opt.step()
            steps += 1
    return steps


def run_fl(silos: list[Silo], d: int, strategy: str = "fedavg", rounds: int = 50,
           local_epochs: int = 2, lr: float = 0.05, batch: int = 128, mu: float = 0.01,
           seed: int = 42, dp_sigma: float | None = None, dp_clip: float = 1.0,
           client_frac: float = 1.0) -> dict:
    """Returns the global model at the round with the lowest federated validation loss."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLP(d)
    gps = params(model)
    sizes = np.array([s.n for s in silos], dtype=float)
    c_glob = [torch.zeros_like(p) for p in gps]
    c_loc = {s.name: [torch.zeros_like(p) for p in gps] for s in silos}
    best = (np.inf, 0, copy.deepcopy(gps))
    hist = []
    for r in range(1, rounds + 1):
        if client_frac >= 1.0:
            chosen = list(range(len(silos)))
        else:  # Poisson sampling (each client independently with prob q): needed for RDP amplification
            chosen = [k for k in range(len(silos)) if rng.random() < client_frac]
            if not chosen:
                continue
        deltas, w = [], []
        dc = [torch.zeros_like(p) for p in gps]
        for k in chosen:
            s = silos[k]
            local = MLP(d)
            set_params(local, gps)
            steps = local_sgd(local, s, strategy, local_epochs, lr, batch,
                              seed * 100003 + r * 101 + k, mu=mu, global_ps=gps,
                              c_glob=c_glob, c_loc=c_loc[s.name])
            new = params(local)
            delta = [a - b for a, b in zip(new, gps)]
            if strategy == "scaffold":
                ci_new = [cl - cg + (g0 - n1) / (steps * lr)
                          for cl, cg, g0, n1 in zip(c_loc[s.name], c_glob, gps, new)]
                for j in range(len(dc)):
                    dc[j] += (ci_new[j] - c_loc[s.name][j]) / len(silos)
                c_loc[s.name] = ci_new
            if dp_sigma is not None:
                norm = torch.sqrt(sum((x ** 2).sum() for x in delta))
                delta = [x * min(1.0, dp_clip / (float(norm) + 1e-12)) for x in delta]
            deltas.append(delta)
            w.append(sizes[k])
        if dp_sigma is None:
            w = np.array(w) / np.sum(w)
        else:
            # standard DP-FedAvg (McMahan et al. 2018): uniform weights with a FIXED
            # denominator (expected number of participants), so one client's clipped
            # update moves the average by at most C / m_exp in L2
            m_exp = client_frac * len(silos)
            w = np.full(len(deltas), 1.0 / m_exp)
        agg = [sum(float(wk) * dl[j] for wk, dl in zip(w, deltas)) for j in range(len(gps))]
        if dp_sigma is not None and dp_sigma > 0:
            sens = dp_clip / (client_frac * len(silos))
            gen = torch.Generator().manual_seed(seed * 7919 + r)
            agg = [a + torch.randn(a.shape, generator=gen) * dp_sigma * sens for a in agg]
        gps = [g + a for g, a in zip(gps, agg)]
        if strategy == "scaffold":
            c_glob = [cg + x for cg, x in zip(c_glob, dc)]
        set_params(model, gps)
        vl = float(np.average([s.val_loss(model) for s in silos], weights=sizes))
        hist.append({"round": r, "val_loss": vl})
        if vl < best[0]:
            best = (vl, r, copy.deepcopy(gps))
    set_params(model, best[2])
    return {"model": model, "best_round": best[1], "history": hist,
            "client_weight_max": float(np.max(sizes / sizes.sum()))}


def personalize(global_model: nn.Module, silo: Silo, d: int, epochs: int = 3,
                lr: float = 0.01, seed: int = 0) -> nn.Module:
    """Fine-tune a copy of the global model on one silo; keep the epoch with best val loss."""
    m = MLP(d)
    set_params(m, params(global_model))
    best = (silo.val_loss(m), params(m))
    for e in range(epochs):
        local_sgd(m, silo, "fedavg", 1, lr, 128, seed + e)
        vl = silo.val_loss(m)
        if vl < best[0]:
            best = (vl, params(m))
    set_params(m, best[1])
    return m


def train_central_mlp(Xtr, ytr, Xva, yva, d: int, epochs: int = 100, lr: float = 0.05,
                      batch: int = 128, seed: int = 42) -> nn.Module:
    """Same architecture AND optimizer as FL (SGD, momentum 0.9); epoch picked on val loss."""
    s = Silo("central", Xtr, ytr, Xva, yva)
    torch.manual_seed(seed)
    m = MLP(d)
    best = (np.inf, params(m))
    for e in range(epochs):
        local_sgd(m, s, "fedavg", 1, lr, batch, seed * 1009 + e)
        vl = s.val_loss(m)
        if vl < best[0]:
            best = (vl, params(m))
    set_params(m, best[1])
    return m


def scores(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X))).numpy()


def epsilon_gaussian(sigma: float, rounds: int, delta: float, q: float = 1.0) -> float:
    """(eps, delta) for `rounds` compositions of the Gaussian mechanism with noise
    multiplier sigma. q < 1 (client sampling) uses Opacus' subsampled-Gaussian RDP."""
    if sigma <= 0:
        return float("inf")
    if q < 1.0:
        from opacus.accountants import RDPAccountant
        acc = RDPAccountant()
        acc.history = [(sigma, q, rounds)]
        return float(acc.get_epsilon(delta=delta))
    alphas = np.arange(1.1, 400, 0.1)
    rdp = rounds * alphas / (2 * sigma ** 2)
    eps = rdp + np.log1p(-1 / alphas) - np.log(delta * alphas) / (alphas - 1)
    return float(np.min(eps))
