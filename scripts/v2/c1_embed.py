"""C1: frozen-encoder embeddings (R2.2, R2.3). One GPU (CUDA_VISIBLE_DEVICES=1).

Inputs
  --source abstracts : data_v2/tabular_v2.parquet  (cleaned title + abstract, 138K)
  --source fulltext  : data_v2/fulltext_clean.parquet (cleaned sections)
Models
  --model modernbert : answerdotai/ModernBERT-large, mean pooling (8K context)
  --model qwen3emb   : Qwen/Qwen3-Embedding-8B (or -4B), last-token pooling
Output: data_v2/emb/<source>_<model>[_<field>].npy (+ .ids.txt), float16

`--benchmark` embeds 512 documents and reports tokens/s (hour-1 checkpoint).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.v2 import DATA2  # noqa: E402
from src.v2.cleaning import clean_title  # noqa: E402,F401

MODELS = {
    "modernbert": ("answerdotai/ModernBERT-large", "mean"),
    "qwen3emb8b": ("Qwen/Qwen3-Embedding-8B", "last"),
    "qwen3emb4b": ("Qwen/Qwen3-Embedding-4B", "last"),
}
INSTRUCT = ("Instruct: Represent this scientific article for assessing its "
            "methodological reliability and research integrity\nQuery: ")


def load_texts(source: str, field: str) -> tuple[list[str], list[str]]:
    if source == "abstracts":
        df = pd.read_parquet(DATA2 / "tabular_v2.parquet", columns=["doi", "title", "abstract"])
        texts = [((t or "") + ". " + (a or "")).strip(". ") for t, a in
                 zip(df["title"], df["abstract"])]
        return df["doi"].tolist(), texts
    df = pd.read_parquet(DATA2 / "fulltext_clean.parquet")
    df = df[df["parse_ok"] & ~df["is_notice"]]
    if field == "sections":
        parts = ["title", "abstract", "introduction", "methods", "results", "discussion",
                 "conclusion"]
        texts = ["\n\n".join(f"{p.upper()}: {r[p]}" for p in parts if r.get(p))
                 for r in df[parts].to_dict("records")]
        # fall back to the body when no section headings were recognised
        texts = [t if len(t) > 500 else f"TITLE: {r['title']}\n\n{r['body']}"
                 for t, r in zip(texts, df[["title", "body"]].to_dict("records"))]
    else:
        texts = df[field].fillna("").tolist()
    return df["pmcid"].tolist(), texts


@torch.no_grad()
def embed(texts, model_key, max_len, batch_tokens, device="cuda"):
    name, pool = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(name, padding_side="left" if pool == "last" else "right")
    model = AutoModel.from_pretrained(name, torch_dtype=torch.bfloat16,
                                      attn_implementation="sdpa").to(device).eval()
    if pool == "last":
        texts = [INSTRUCT + t for t in texts]
    enc = tok(texts, truncation=True, max_length=max_len, add_special_tokens=True)["input_ids"]
    order = np.argsort([-len(x) for x in enc])
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float16)
    i, n_tok, t0 = 0, 0, time.time()
    while i < len(order):
        L = len(enc[order[i]])
        bs = max(1, batch_tokens // max(L, 1))
        idx = order[i:i + bs]
        batch = tok.pad({"input_ids": [enc[j] for j in idx]}, return_tensors="pt").to(device)
        h = model(**batch).last_hidden_state
        msk = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
        if pool == "mean":
            e = (h * msk).sum(1) / msk.sum(1).clamp(min=1)
        else:
            e = h[:, -1]  # left padding: last position is the final real token
        out[idx] = F.normalize(e.float(), dim=-1).cpu().numpy().astype(np.float16)
        n_tok += int(batch["attention_mask"].sum())
        i += len(idx)
        if (i // max(bs, 1)) % 50 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(order)} docs, {n_tok / el:,.0f} tok/s, {el / 60:.1f} min", flush=True)
    return out, n_tok / (time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["abstracts", "fulltext"], required=True)
    ap.add_argument("--field", default="sections")
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--batch-tokens", type=int, default=65536)
    ap.add_argument("--benchmark", action="store_true")
    a = ap.parse_args()
    ids, texts = load_texts(a.source, a.field)
    if a.benchmark:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(texts), size=min(512, len(texts)), replace=False)
        ids, texts = [ids[i] for i in sel], [texts[i] for i in sel]
    print(f"{a.source}/{a.field}: {len(texts)} docs with {a.model}, max_len {a.max_len}", flush=True)
    emb, tps = embed(texts, a.model, a.max_len, a.batch_tokens)
    print(f"throughput: {tps:,.0f} tokens/s", flush=True)
    if a.benchmark:
        return
    od = DATA2 / "emb"
    od.mkdir(parents=True, exist_ok=True)
    stem = f"{a.source}_{a.model}" + (f"_{a.field}" if a.source == "fulltext" else "")
    np.save(od / f"{stem}.npy", emb)
    (od / f"{stem}.ids.txt").write_text("\n".join(ids))
    print(f"saved {emb.shape} -> {od / stem}.npy")


if __name__ == "__main__":
    main()
