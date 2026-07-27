"""Client-level DP for the FL engine: per-update clipping + Gaussian noise.

We add DP at the server aggregation step (central DP over client updates),
the standard DP-FedAvg recipe (McMahan et al., 2018): clip each client's
model delta to C in L2, average, then add N(0, (sigma*C)^2 / K) per coordinate.
Epsilon is accounted with the RDP accountant over rounds.
"""
from __future__ import annotations

import copy

import numpy as np

from src.fl.engine import Client, aggregate
from src.models.evaluate import evaluate
from src.models.mlp import MLP, get_weights, set_weights
import torch


def _flat(ws: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([w.ravel() for w in ws])


def _unflat(vec: np.ndarray, like: list[np.ndarray]) -> list[np.ndarray]:
    out, i = [], 0
    for w in like:
        out.append(vec[i : i + w.size].reshape(w.shape).astype(w.dtype))
        i += w.size
    return out


def run_dp_fl(clients: list[Client], Xte, yte, sigma: float, clip: float = 1.0,
              rounds: int = 50, local_epochs: int = 2, lr: float = 0.05,
              batch: int = 128, seed: int = 42, eval_every: int = 5) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLP(clients[0].X.shape[1])
    global_weights = get_weights(model)
    Xte_t = torch.tensor(Xte)
    history = []
    K = len(clients)
    for rnd in range(1, rounds + 1):
        gflat = _flat(global_weights)
        deltas = []
        for c in clients:
            w = c.local_train(copy.deepcopy(global_weights), "fedavg",
                              local_epochs, lr, batch, 0.0, seed * 1000 + rnd)
            d = _flat(w) - gflat
            norm = np.linalg.norm(d)
            deltas.append(d * min(1.0, clip / max(norm, 1e-12)))
        avg = np.mean(deltas, axis=0)
        if sigma > 0:
            avg = avg + rng.normal(0, sigma * clip / K, size=avg.shape)
        global_weights = _unflat(gflat + avg, global_weights)
        if rnd % eval_every == 0 or rnd == rounds:
            set_weights(model, global_weights)
            model.eval()
            with torch.no_grad():
                scores = torch.sigmoid(model(Xte_t)).numpy()
            history.append({"round": rnd, **evaluate(yte, scores)})
    return {"final": history[-1], "history": history}


def epsilon_for(sigma: float, rounds: int, delta: float = 1e-3) -> float:
    """RDP accounting for the Gaussian mechanism composed over `rounds`
    (every client participates every round, so no subsampling amplification)."""
    if sigma <= 0:
        return float("inf")
    alphas = np.arange(1.1, 200, 0.1)
    rdp = rounds * alphas / (2 * sigma**2)
    eps = rdp + np.log1p(-1 / alphas) - np.log(delta * alphas) / (alphas - 1)
    return float(np.min(eps))
