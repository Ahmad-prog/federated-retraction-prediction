"""A1: refresh Retraction Watch (RW) and derive v2 label sets.

Outputs (data_v2/):
  rw_all_dois.txt            every original-paper DOI in RW, any notice nature
                             (used to keep ALL RW papers out of the controls; M12)
  prospective_positives.parquet
                             research-article retractions added to RW after the
                             v1 snapshot (2026-07-19) and retracted after it:
                             a true prospective test set (R0.7)
  rw_stats.json              counts used in the paper text (M15, m1)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.v2 import DATA2, ROOT  # noqa: E402

NEW = DATA2 / "raw" / "retraction_watch_2026-10-07.csv"
OLD = ROOT / "data" / "raw" / "retraction_watch.csv"
V1_SNAPSHOT = pd.Timestamp("2026-07-19")


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, encoding="utf-8", encoding_errors="replace")
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c]]
    df["doi"] = df["OriginalPaperDOI"].astype(str).str.strip().str.lower()
    df.loc[df["doi"].isin(["", "nan", "unavailable", "none"]), "doi"] = None
    df["rdate"] = pd.to_datetime(df["RetractionDate"], errors="coerce", format="mixed")
    df["odate"] = pd.to_datetime(df["OriginalPaperDate"], errors="coerce", format="mixed")
    df["nature"] = df["RetractionNature"].astype(str).str.strip().str.lower()
    return df


def main() -> None:
    new, old = load(NEW), load(OLD)
    print(f"RW new snapshot: {len(new)} records; v1 snapshot: {len(old)}")

    all_dois = sorted(set(new["doi"].dropna()) | set(old["doi"].dropna()))
    (DATA2 / "rw_all_dois.txt").write_text("\n".join(all_dois))
    print(f"all RW original DOIs (any nature): {len(all_dois)}")

    is_ret = (new["nature"] == "retraction") & new["ArticleType"].str.contains(
        "Research Article", case=False, na=False) & new["doi"].notna()
    added = ~new["Record ID"].isin(old["Record ID"])
    prosp = new[is_ret & added & (new["rdate"] > V1_SNAPSHOT)].drop_duplicates("doi")
    old_ret_dois = set(old.loc[old["nature"] == "retraction", "doi"].dropna())
    prosp = prosp[~prosp["doi"].isin(old_ret_dois)]
    prosp[["Record ID", "doi", "Title", "Journal", "Publisher", "Reason",
           "rdate", "odate"]].to_parquet(DATA2 / "prospective_positives.parquet", index=False)
    print(f"prospective positives (added + retracted after {V1_SNAPSHOT.date()}): {len(prosp)}")
    print(prosp["Publisher"].value_counts().head(10).to_string())

    ret = new[new["nature"] == "retraction"]
    lag_years = ((ret["rdate"] - ret["odate"]).dt.days / 365.25).dropna()
    lag_years = lag_years[(lag_years >= 0) & (lag_years < 60)]
    stats = {
        "snapshot": "2026-10-07",
        "n_records": int(len(new)),
        "n_retraction_notices": int(len(ret)),
        "retractions_by_year": {int(y): int(n) for y, n in
                                ret["rdate"].dt.year.value_counts().sort_index().items()
                                if pd.notna(y) and y >= 2015},
        "lag_years_mean": round(float(lag_years.mean()), 2),
        "lag_years_median": round(float(lag_years.median()), 2),
        "n_prospective_positives": int(len(prosp)),
        "reason_top": new.loc[is_ret, "Reason"].str.split(";").explode().str.strip()
                         .str.lstrip("+").value_counts().head(15).to_dict(),
    }
    (DATA2 / "rw_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "reason_top"}, indent=1))


if __name__ == "__main__":
    main()
