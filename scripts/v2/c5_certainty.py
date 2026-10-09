"""C5: model-based sentence certainty (R2.8; M9), the measure used by Usman & Balke.

pedropei/sentence-level-certainty (Pei & Jurgens 2021) scores every abstract sentence;
per abstract we keep mean, min, std, 10th percentile and the share of low-certainty
sentences (bottom quartile of all sentences). These replace the v1 word-list proxies.
Output: data_v2/certainty_abstracts.parquet (doi + features)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.v2 import DATA2  # noqa: E402

NAME = "pedropei/sentence-level-certainty"
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


@torch.no_grad()
def main() -> None:
    df = pd.read_parquet(DATA2 / "tabular_v2.parquet", columns=["doi", "abstract"])
    sents, owner = [], []
    for i, a in enumerate(df["abstract"].fillna("")):
        for s in SENT.split(a):
            if len(s.split()) >= 4:
                sents.append(s[:1000])
                owner.append(i)
    print(f"{len(sents)} sentences from {len(df)} abstracts", flush=True)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NAME, dtype=torch.bfloat16).cuda().eval()
    order = np.argsort([len(s) for s in sents])
    scores = np.zeros(len(sents), dtype=np.float32)
    bs = 512
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        enc = tok([sents[j] for j in idx], padding=True, truncation=True, max_length=128,
                  return_tensors="pt").to("cuda")
        out = model(**enc).logits.float()
        scores[idx] = (out[:, 0] if out.shape[1] == 1 else out.softmax(-1)[:, -1]).cpu().numpy()
        if k % (bs * 200) == 0:
            print(f"  {k}/{len(sents)}", flush=True)
    low = np.quantile(scores, 0.25)
    g = pd.DataFrame({"owner": owner, "c": scores})
    agg = g.groupby("owner")["c"].agg(cert_mean="mean", cert_min="min", cert_std="std",
                                      cert_p10=lambda x: np.quantile(x, 0.1),
                                      cert_low_share=lambda x: float(np.mean(x <= low)))
    out = df[["doi"]].join(agg, how="left")
    out.to_parquet(DATA2 / "certainty_abstracts.parquet", index=False)
    print(out.describe().round(3).to_string())


if __name__ == "__main__":
    main()
