"""Rebuild the cleaned abstracts that FedRetract does not redistribute (no open licence).

For every article whose `abstract_file` is "rebuild", the abstract is downloaded from OpenAlex (abstract_inverted_index),
reconstructed in word order and cleaned with exactly the rules used to build the dataset (src/v2/cleaning.py). The SHA-256
of the result is compared with `abstract_sha256` in the release, so users can see which abstracts were rebuilt
byte-identically (OpenAlex records can change after the snapshot).

Usage (from the release folder):
  pip install -r code/requirements.txt
  python code/scripts/v2/rebuild_abstracts.py --articles benchmark/articles.parquet --out rebuilt_abstracts.parquet
  [--mailto you@example.org]  (optional, OpenAlex polite pool)  [--limit N]  [--all]
Output columns: doi, openalex_id, abstract, sha256_match
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.v2.cleaning import clean_abstract  # noqa: E402

API = "https://api.openalex.org/works"


def abstract_from_inv(inv: dict | None) -> str:
    """Same reconstruction as scripts/v2/b1_tabular_features.py."""
    if not inv:
        return ""
    pos = {}
    for w, idx in inv.items():
        for i in idx:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))


def cleaned(raw: str) -> str:
    """Cleaned abstract as stored in the dataset ('' when the abstract is missing, too short or a notice)."""
    text = clean_abstract(raw)[0]
    return text if len(text) >= 40 else ""


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch(ids: list[str], mailto: str | None) -> dict[str, str]:
    q = {"filter": "openalex:" + "|".join(i.rsplit("/", 1)[-1] for i in ids),
         "select": "id,abstract_inverted_index", "per-page": "100"}
    if mailto:
        q["mailto"] = mailto
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(q)}", timeout=60) as r:
                res = json.loads(r.read())["results"]
            return {w["id"]: abstract_from_inv(w.get("abstract_inverted_index")) for w in res}
        except Exception:  # noqa: BLE001 - network hiccups: back off and retry
            time.sleep(2 ** attempt)
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", required=True, help="benchmark/articles.parquet or prospective/articles.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mailto", default=None)
    ap.add_argument("--limit", type=int, default=None, help="rebuild only the first N articles (testing)")
    ap.add_argument("--all", action="store_true", help="also re-download abstracts that are redistributed")
    a = ap.parse_args()
    art = pd.read_parquet(a.articles, columns=["doi", "openalex_id", "abstract_file", "abstract_sha256"])
    todo = art if a.all else art[art["abstract_file"] == "rebuild"]
    todo = todo.head(a.limit) if a.limit else todo
    rows = []
    ids = todo["openalex_id"].tolist()
    for i in range(0, len(ids), 50):
        got = fetch(ids[i:i + 50], a.mailto)
        for _, r in todo.iloc[i:i + 50].iterrows():
            text = cleaned(got.get(r.openalex_id, ""))
            rows.append({"doi": r.doi, "openalex_id": r.openalex_id, "abstract": text,
                         "sha256_match": sha256(text) == r.abstract_sha256})
        time.sleep(0.1)
    out = pd.DataFrame(rows)
    out.to_parquet(a.out, index=False)
    print(f"rebuilt {len(out)} abstracts; identical to the release: {out['sha256_match'].mean():.1%}")


if __name__ == "__main__":
    main()
