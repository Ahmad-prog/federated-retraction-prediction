"""Small MLP for tabular retraction features (CPU-friendly)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...] = (64, 32), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FocalLoss(nn.Module):
    """Binary focal loss for extreme class imbalance (Lin et al., 2017)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


def get_weights(model: nn.Module) -> list[np.ndarray]:
    return [p.detach().cpu().numpy() for p in model.state_dict().values()]


def set_weights(model: nn.Module, weights: list[np.ndarray]) -> None:
    state = model.state_dict()
    for k, w in zip(state.keys(), weights):
        state[k] = torch.tensor(w)
    model.load_state_dict(state)
