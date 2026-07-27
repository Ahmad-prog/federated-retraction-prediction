"""Centralized and local-only baselines: LogReg, XGBoost, MLP."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features.build_features import NUMERIC_FEATURES
from src.models.evaluate import evaluate
from src.models.mlp import MLP


def make_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[NUMERIC_FEATURES].to_numpy(dtype=np.float32)
    y = df["retracted"].to_numpy(dtype=np.float32)
    return X, y


def preprocessor() -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())])


def train_logreg(Xtr, ytr, Xte, yte, seed: int) -> dict:
    pipe = Pipeline([*preprocessor().steps,
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                random_state=seed))])
    pipe.fit(Xtr, ytr)
    return evaluate(yte, pipe.predict_proba(Xte)[:, 1])


def train_rf(Xtr, ytr, Xte, yte, seed: int) -> dict:
    """Random Forest on readability/certainty/metadata features — the direct
    reproduction of Usman & Balke's WebSci'25 architecture on our corpus."""
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 n_jobs=4, random_state=seed)
    imp = SimpleImputer(strategy="median")
    clf.fit(imp.fit_transform(Xtr), ytr)
    return evaluate(yte, clf.predict_proba(imp.transform(Xte))[:, 1])


def train_xgb(Xtr, ytr, Xte, yte, seed: int) -> dict:
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    clf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                        subsample=0.9, colsample_bytree=0.9, n_jobs=4,
                        scale_pos_weight=spw, random_state=seed,
                        eval_metric="aucpr")
    imp = SimpleImputer(strategy="median")
    clf.fit(imp.fit_transform(Xtr), ytr)
    return evaluate(yte, clf.predict_proba(imp.transform(Xte))[:, 1])


def train_mlp(Xtr, ytr, Xte, yte, seed: int, epochs: int = 30, lr: float = 1e-3,
              batch: int = 256, loss_fn=None) -> dict:
    torch.manual_seed(seed)
    pre = preprocessor()
    Xtr_, Xte_ = pre.fit_transform(Xtr).astype(np.float32), pre.transform(Xte).astype(np.float32)
    model = MLP(Xtr_.shape[1])
    if loss_fn is None:
        pos_weight = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), dtype=torch.float32)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = torch.utils.data.TensorDataset(torch.tensor(Xtr_), torch.tensor(ytr))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.tensor(Xte_))).numpy()
    return evaluate(yte, scores)
