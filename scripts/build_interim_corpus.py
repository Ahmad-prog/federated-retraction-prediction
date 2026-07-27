"""Assemble an interim corpus purely from cached OpenAlex JSON (no API calls).

Keeps positives whose (venue_id, year) cell has cached controls, so the pilot
corpus has the same matched-control design as the full one. Used while the
OpenAlex daily budget is exhausted; the full run replaces it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.openalex import flatten  # noqa: E402

CACHE = ROOT / "data" / "cache"
OUT = ROOT / "data" / "processed"


def load_dir(d: Path) -> list[dict]:
    rows = []
    for f in sorted(d.glob("*.json")):
        rows.extend(flatten(w) for w in json.loads(f.read_text()))
    return rows


def main() -> None:
    pos = pd.DataFrame(load_dir(CACHE / "positives"))
    pos["retracted"] = 1
    ctl_rows = []
    cells = set()
    for f in sorted((CACHE / "controls").glob("*.json")):
        vid, year = f.stem.rsplit("_", 1)
        cells.add((f"https://openalex.org/{vid}", int(year)))
        ctl_rows.extend(flatten(w) for w in json.loads(f.read_text()))
    neg = pd.DataFrame(ctl_rows)
    neg["retracted"] = 0

    pos_keep = pos[[(v, y) in cells for v, y in zip(pos["venue_id"], pos["year"])]]
    pos_dois = set(pos["doi"])
    neg = neg[~neg["doi"].isin(pos_dois)]

    corpus = pd.concat([pos_keep, neg], ignore_index=True).drop_duplicates(subset="doi")
    OUT.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(OUT / "corpus_interim.parquet", index=False)
    print(f"cells with controls: {len(cells)}")
    print(f"positives kept: {len(pos_keep)} / {len(pos)}, controls: {len(neg)}")
    print(f"Saved {len(corpus)} rows -> {OUT / 'corpus_interim.parquet'}")


if __name__ == "__main__":
    main()
