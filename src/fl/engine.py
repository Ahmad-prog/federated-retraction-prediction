"""Lightweight in-process cross-silo FL engine (FedAvg / FedProx / imbalance-aware).

Runs clients sequentially per round — mathematically identical to a FedAvg
simulation, fully deterministic, and avoids Ray overhead on WSL. Strategies:

- fedavg:    weight aggregation by client sample count (McMahan et al., 2017)
- fedprox:   fedavg + proximal term mu/2 * ||w - w_global||^2 (Li et al., 2020)
- fedbal:    local focal loss + aggregation weighted by effective positive
             count per silo (our imbalance-aware variant)
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from src.models.evaluate import evaluate
from src.models.mlp import MLP, FocalLoss, get_weights, set_weights


class Client:
    def __init__(self, cid: str, Xtr, ytr):
        self.cid = cid
        self.X = torch.tensor(Xtr)
        self.y = torch.tensor(ytr)
        self.n = len(ytr)
        self.n_pos = int(self.y.sum().item())

    def local_train(self, global_weights, strategy: str, epochs: int, lr: float,
                    batch: int, mu: float, seed: int) -> list[np.ndarray]:
        torch.manual_seed(seed)
        model = MLP(self.X.shape[1])
        set_weights(model, global_weights)
        if strategy == "fedbal":
            loss_fn = FocalLoss()
        else:
            pos_weight = torch.tensor(
                (self.n - self.n_pos) / max(self.n_pos, 1), dtype=torch.float32)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        global_params = [torch.tensor(w) for w in global_weights]
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.X, self.y),
            batch_size=batch, shuffle=True,
            generator=torch.Generator().manual_seed(seed))
        model.train()
        for _ in range(epochs):
            for xb, yb in dl:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                if strategy == "fedprox" and mu > 0:
                    # state_dict() detaches, so iterate parameters() to keep grads
                    prox = sum(((p - g) ** 2).sum()
                               for p, g in zip(model.parameters(), global_params))
                    loss = loss + (mu / 2) * prox
                loss.backward()
                opt.step()
        return get_weights(model)


def aggregate(updates: list[list[np.ndarray]], weights: np.ndarray) -> list[np.ndarray]:
    weights = weights / weights.sum()
    return [np.sum([u[i] * w for u, w in zip(updates, weights)], axis=0)
            for i in range(len(updates[0]))]


def run_fl(clients: list[Client], Xte, yte, strategy: str = "fedavg",
           rounds: int = 50, local_epochs: int = 2, lr: float = 0.05,
           batch: int = 128, mu: float = 0.01, seed: int = 42,
           eval_every: int = 5) -> dict:
    torch.manual_seed(seed)
    model = MLP(clients[0].X.shape[1])
    global_weights = get_weights(model)
    Xte_t = torch.tensor(Xte)
    history = []
    for rnd in range(1, rounds + 1):
        updates = [c.local_train(copy.deepcopy(global_weights), strategy,
                                 local_epochs, lr, batch, mu, seed * 1000 + rnd)
                   for c in clients]
        if strategy == "fedbal":
            # effective positive count: rare-class evidence drives the average
            agg_w = np.array([np.sqrt(c.n) * (1 + c.n_pos / max(c.n, 1)) for c in clients])
        else:
            agg_w = np.array([float(c.n) for c in clients])
        global_weights = aggregate(updates, agg_w)
        if rnd % eval_every == 0 or rnd == rounds:
            set_weights(model, global_weights)
            model.eval()
            with torch.no_grad():
                scores = torch.sigmoid(model(Xte_t)).numpy()
            m = evaluate(yte, scores)
            history.append({"round": rnd, **m})
    return {"final": history[-1], "history": history, "weights": global_weights}
