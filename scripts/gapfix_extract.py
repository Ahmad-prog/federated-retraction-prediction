"""Gap-fix feature extraction on the Usman WebSci'25 corpus (464 papers).

Produces data/processed/gapfix_features.parquet with, per paper and section:
  - certainty profile (Gap 3+4): PySBD segmentation -> certainty-estimator BERT
      -> mean (his statistic) + min, var, q10, frac_low, and trajectory deltas
  - tortured-phrase counts (Gap 7a)
  - GPT-2 perplexity (Gap 7b)
And caches SciBERT section embeddings for the Gap-1 model:
  data/processed/gapfix_scibert.npz  (sections x 768, chunk-averaged)

Everything cached per stage; safe to rerun.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SECTIONS = {"abstract": "abstract", "methods": "Materials_and_Methods",
            "results": "Results", "conclusion": "Conclusion"}
CACHE = ROOT / "data/cache/gapfix"
CACHE.mkdir(parents=True, exist_ok=True)

# Cabanac et al. tortured-phrases seed list (fingerprints of paraphrased text)
TORTURED = [
    "counterfeit consciousness", "man-made brainpower", "profound learning",
    "fake neural organization", "counterfeit neural", "irregular woodland",
    "arbitrary backwoods", "choice tree", "gullible bayes", "innocent bayes",
    "support vector machine", "colossal information", "huge information",
    "bosom malignancy", "bosom disease", "kidney disappointment",
    "flag to commotion", "sign to clamor", "mean square blunder",
    "root mean square mistake", "subterranean insect province",
    "microorganisms settlement", "wellspring of data", "watchword extraction",
    "execution measurements", "profound convolutional",
]


def load_corpus() -> pd.DataFrame:
    df = pd.read_csv(
        ROOT / "data/raw/Scientific-Accountability-main/Retracted_NonRetracted Articles.csv",
        sep=";", engine="python", quotechar='"', on_bad_lines="skip")
    df["y"] = (df["Label"].str.strip() == "R").astype(int)
    return df.reset_index(drop=True)


def segment(df: pd.DataFrame) -> dict:
    """Gap 3: proper sentence segmentation, cached."""
    f = CACHE / "sentences.json"
    if f.exists():
        return json.loads(f.read_text())
    import pysbd
    seg = pysbd.Segmenter(language="en", clean=False)
    out = {}
    for i, row in df.iterrows():
        for tag, col in SECTIONS.items():
            txt = str(row[col]) if pd.notna(row[col]) else ""
            sents = [s.strip() for s in seg.segment(txt) if len(s.strip()) > 15] if len(txt) > 30 else []
            out[f"{i}_{tag}"] = sents[:80]
    f.write_text(json.dumps(out))
    print("segmentation done", flush=True)
    return out


def certainty_scores(sent_map: dict) -> dict:
    """Gap 4: his certainty model over clean sentences, cached incrementally."""
    f = CACHE / "certainty.json"
    done = json.loads(f.read_text()) if f.exists() else {}
    todo = {k: v for k, v in sent_map.items() if k not in done and v}
    if todo:
        # direct port of certainty_estimator.predict_certainty (the released
        # package targets transformers 4.x): same checkpoint, same encoding
        # (max_length 60, padded), same regression head output
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        ckpt = "pedropei/sentence-level-certainty"
        tok = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=1)
        model.eval()

        def predict(sents: list[str]) -> list[float]:
            ids = [tok.encode(s, add_special_tokens=True, max_length=60,
                              truncation=True) for s in sents]
            ids = [x + [tok.pad_token_id] * (60 - len(x)) for x in ids]
            with torch.no_grad():
                out = model(torch.tensor(ids).long())
            return [float(v) for v in out.logits.cpu().numpy().flatten()]

        keys = list(todo.keys())
        for j, k in enumerate(keys):
            try:
                done[k] = predict(todo[k])
            except Exception:
                done[k] = []
            if j % 50 == 0:
                f.write_text(json.dumps(done))
                print(f"  certainty {j}/{len(keys)}", flush=True)
        f.write_text(json.dumps(done))
    print("certainty done", flush=True)
    return done


def profile(scores: list[float]) -> dict:
    if not scores:
        return {"mean": np.nan, "min": np.nan, "var": np.nan, "q10": np.nan, "frac_low": np.nan}
    a = np.array(scores, dtype=float)
    return {"mean": float(a.mean()), "min": float(a.min()), "var": float(a.var()),
            "q10": float(np.quantile(a, 0.1)),
            "frac_low": float((a < np.quantile(a, 0.25).clip(min=a.mean() - a.std())).mean()
                              if len(a) > 3 else 0.0)}


def perplexity_and_phrases(df: pd.DataFrame) -> pd.DataFrame:
    """Gap 7: GPT-2 perplexity + tortured-phrase counts per section, cached."""
    f = CACHE / "gap7.parquet"
    if f.exists():
        return pd.read_parquet(f)
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    lm = GPT2LMHeadModel.from_pretrained("gpt2")
    lm.eval()
    rows = []
    for i, row in df.iterrows():
        r = {}
        for tag, col in SECTIONS.items():
            txt = (str(row[col]) if pd.notna(row[col]) else "").strip()
            low = txt.lower()
            r[f"{tag}_tortured"] = sum(low.count(p) for p in TORTURED)
            if len(txt) > 100:
                ids = tok(txt, truncation=True, max_length=512, return_tensors="pt").input_ids
                with torch.no_grad():
                    loss = lm(ids, labels=ids).loss
                r[f"{tag}_ppl"] = float(torch.exp(loss))
            else:
                r[f"{tag}_ppl"] = np.nan
        rows.append(r)
        if i % 50 == 0:
            print(f"  gap7 {i}/{len(df)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(f, index=False)
    print("gap7 done", flush=True)
    return out


def scibert_embeddings(df: pd.DataFrame) -> None:
    """Gap 1: chunk-averaged SciBERT embeddings per section, cached npz."""
    f = ROOT / "data/processed/gapfix_scibert.npz"
    if f.exists():
        print("scibert cached", flush=True)
        return
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    enc = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    enc.eval()
    embs = np.zeros((len(df), len(SECTIONS), 768), dtype=np.float32)
    mask = np.zeros((len(df), len(SECTIONS)), dtype=np.float32)
    with torch.no_grad():
        for i, row in df.iterrows():
            for si, (tag, col) in enumerate(SECTIONS.items()):
                txt = (str(row[col]) if pd.notna(row[col]) else "").strip()
                if len(txt) < 50:
                    continue
                ids = tok(txt, return_tensors="pt", truncation=False).input_ids[0]
                chunks = [ids[j:j + 510] for j in range(0, min(len(ids), 2040), 510)]
                vecs = []
                for ch in chunks:
                    ch = torch.cat([torch.tensor([tok.cls_token_id]), ch,
                                    torch.tensor([tok.sep_token_id])]).unsqueeze(0)
                    vecs.append(enc(ch).last_hidden_state[:, 0].squeeze(0).numpy())
                embs[i, si] = np.mean(vecs, axis=0)
                mask[i, si] = 1.0
            if i % 25 == 0:
                print(f"  scibert {i}/{len(df)}", flush=True)
    np.savez_compressed(f, embs=embs, mask=mask, y=df["y"].to_numpy())
    print("scibert done", flush=True)


def main() -> None:
    df = load_corpus()
    print(f"corpus: {len(df)} papers, {int(df['y'].sum())} retracted", flush=True)
    sent_map = segment(df)
    cert = certainty_scores(sent_map)

    rows = []
    for i in range(len(df)):
        r = {}
        means = {}
        for tag in SECTIONS:
            p = profile(cert.get(f"{i}_{tag}", []))
            means[tag] = p["mean"]
            r.update({f"{tag}_cert_{k}": v for k, v in p.items()})
        # trajectory: confidence drop from abstract to methods/results
        r["traj_abs_minus_methods"] = (means["abstract"] - means["methods"]
                                       if pd.notna(means["abstract"]) and pd.notna(means["methods"]) else np.nan)
        r["traj_abs_minus_results"] = (means["abstract"] - means["results"]
                                       if pd.notna(means["abstract"]) and pd.notna(means["results"]) else np.nan)
        rows.append(r)
    certdf = pd.DataFrame(rows)

    g7 = perplexity_and_phrases(df)
    out = pd.concat([df[["pmcid", "y"]], certdf, g7], axis=1)
    out.to_parquet(ROOT / "data/processed/gapfix_features.parquet", index=False)
    print(f"features saved: {out.shape}", flush=True)

    scibert_embeddings(df)
    print("EXTRACTION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
