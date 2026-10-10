"""R3: licence-aware FedRetract release for the Scientific Data Data Descriptor.

Starts from the R1 package (release/FedRetract) and changes how article text is shared:
  * abstracts are redistributed only where the article's licence allows it (OpenAlex primary_location.license):
      benchmark/abstracts_cc_by.parquet     CC BY, CC0, public domain
      benchmark/abstracts_cc_other.parquet  CC BY-SA, CC BY-NC, CC BY-NC-SA (each row keeps its own licence)
    all other abstracts (no stated licence, publisher-specific, -ND licences) are released as an SHA-256 checksum and
    rebuilt from OpenAlex with code/scripts/v2/rebuild_abstracts.py;
  * the same for the prospective set;
  * abstract embeddings are kept only for rows whose abstract is redistributed or missing (title only), as already done
    for the full-text embeddings;
  * a sample of withheld abstracts is rebuilt from the live OpenAlex API to measure how many come back byte-identical.
Titles, metadata, labels, features and splits are unchanged.

Output: release_scidata/FedRetract/ + FedRetract_data.zip, FedRetract_embeddings.zip, SHA256SUMS.txt, MD5SUMS.txt
Usage: python scripts/v2/r3_build_scidata_release.py [--sample 1000]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.v2.rebuild_abstracts import abstract_from_inv, cleaned, sha256  # noqa: E402

SRC = ROOT / "release" / "FedRetract"
OUTD = ROOT / "release_scidata"
OUT = OUTD / "FedRetract"
OPEN = {"cc-by": "CC BY", "cc0": "CC0", "public-domain": "public domain"}
OTHER = {"cc-by-sa": "CC BY-SA", "cc-by-nc": "CC BY-NC", "cc-by-nc-sa": "CC BY-NC-SA"}
NAMES = {**OPEN, **OTHER, "cc-by-nd": "CC BY-ND", "cc-by-nc-nd": "CC BY-NC-ND",
         "publisher-specific-oa": "publisher-specific", "other-oa": "other open access", "implied-oa": "implied open access"}


def licences_and_raw(files: list[str]) -> pd.DataFrame:
    rows = []
    for f in files:
        works = json.loads(Path(f).read_text())
        works = works if isinstance(works, list) else works.get("results", list(works.values()))
        for w in works:
            if not isinstance(w, dict):
                continue
            loc = w.get("primary_location") or {}
            rows.append({"doi": (w.get("doi") or "").replace("https://doi.org/", "").lower(),
                         "oa_id": w.get("id"), "lic": loc.get("license"), "raw": abstract_from_inv(w.get("abstract_inverted_index"))})
    return pd.DataFrame(rows).drop_duplicates("doi")


def split_text(art: pd.DataFrame, lic: pd.DataFrame, folder: Path, stats_key: str, stats: dict) -> pd.DataFrame:
    art = art.merge(lic, on="doi", how="left")
    if "openalex_id" not in art:
        art.insert(1, "openalex_id", art["oa_id"])
    re_clean = art["raw"].fillna("").map(cleaned)
    same = (re_clean == art["abstract"].fillna("")).mean()
    art["abstract_licence"] = art["lic"].map(lambda x: NAMES.get(x, x) if isinstance(x, str) else "none stated")
    has = art["abstract"].fillna("").str.len() > 0
    art["abstract_file"] = np.where(~has, "", np.where(art["lic"].isin(OPEN), "abstracts_cc_by",
                                                      np.where(art["lic"].isin(OTHER), "abstracts_cc_other", "rebuild")))
    art["abstract_sha256"] = np.where(has, art["abstract"].fillna("").map(sha256), "")
    for name in ["abstracts_cc_by", "abstracts_cc_other"]:
        art.loc[art.abstract_file == name, ["doi", "openalex_id", "abstract_licence", "abstract"]].to_parquet(
            folder / f"{name}.parquet", index=False)
    y = art["retracted"] if "retracted" in art else pd.Series(1, index=art.index)
    stats[stats_key] = {
        "cache_reproduces_released_abstract": float(same),
        "abstract_file_counts": art["abstract_file"].replace("", "no abstract").value_counts().to_dict(),
        "abstract_file_by_class": art.assign(y=y).groupby(["abstract_file", "y"]).size().unstack(fill_value=0)
        .rename(index={"": "no abstract"}).to_dict(),
        "licence_counts": art.loc[has, "abstract_licence"].value_counts().to_dict(),
    }
    return art.drop(columns=["lic", "raw", "abstract", "oa_id"])


def zipdir(zpath: Path, base: Path, members: list[Path]) -> None:
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for m in members:
            for p in sorted([m] if m.is_file() else m.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(base))


def digest(p: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=1000, help="withheld abstracts rebuilt from the live API (0 = skip)")
    a = ap.parse_args()
    if OUTD.exists():
        shutil.rmtree(OUTD)
    shutil.copytree(SRC, OUT, ignore=shutil.ignore_patterns("code"))
    stats = json.loads((OUT / "stats.json").read_text())

    # ------------------------------------------------------------ benchmark + prospective abstracts
    lic = licences_and_raw(sorted(glob.glob(str(ROOT / "data/cache/positives/*.json"))) +
                           sorted(glob.glob(str(ROOT / "data/cache/controls_pub/*.json"))))
    art = pd.read_parquet(OUT / "benchmark/articles.parquet")
    art = split_text(art, lic, OUT / "benchmark", "benchmark_text", stats)
    art.to_parquet(OUT / "benchmark/articles.parquet", index=False)

    plic = licences_and_raw([str(ROOT / "data_v2/cache/openalex_prospective.json")])
    pro = pd.read_parquet(OUT / "prospective/articles.parquet")
    pro = split_text(pro, plic, OUT / "prospective", "prospective_text", stats)
    pro.to_parquet(OUT / "prospective/articles.parquet", index=False)

    # ------------------------------------------------------------ embeddings of withheld abstracts are dropped
    keep = set(art.loc[art.abstract_file != "rebuild", "doi"])
    for stem in ["abstracts_modernbert", "abstracts_qwen3emb8b"]:
        ids = (OUT / "embeddings" / f"{stem}.ids.txt").read_text().split("\n")
        E = np.load(OUT / "embeddings" / f"{stem}.npy")
        m = np.array([i in keep for i in ids])
        np.save(OUT / "embeddings" / f"{stem}.npy", E[m])
        (OUT / "embeddings" / f"{stem}.ids.txt").write_text("\n".join(i for i, k in zip(ids, m) if k))
    stats["embeddings"] = {f.stem: list(np.load(f, mmap_mode="r").shape) for f in sorted((OUT / "embeddings").glob("*.npy"))}

    # ------------------------------------------------------------ code (current versions) + documents
    code = OUT / "code"
    for rel in ["src/__init__.py", "src/v2", "src/features/build_features.py", "src/features/__init__.py",
                "src/models/evaluate.py", "src/models/__init__.py", "scripts/__init__.py", "scripts/v2"]:
        p = ROOT / rel
        if not p.exists():
            continue
        dst = code / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if p.is_dir():
            shutil.copytree(p, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.log"))
        else:
            shutil.copy(p, dst)
    (code / "scripts/__init__.py").touch()
    # corpus construction (src/data): the R1 copy, whose OpenAlex client reads the contact e-mail from OPENALEX_MAILTO
    shutil.copytree(SRC / "code/src/data", code / "src/data", ignore=shutil.ignore_patterns("__pycache__"))
    if (SRC / "code/requirements.txt").exists():
        shutil.copy(SRC / "code/requirements.txt", code / "requirements.txt")
    docs = ROOT / "release_docs"  # README.md, DATASHEET.md, LICENSE.md, CITATION.cff written for this release
    for f in docs.glob("*"):
        shutil.copy(f, OUT / f.name)

    # ------------------------------------------------------------ rebuild check on a sample of withheld abstracts
    if a.sample:
        smp = art[art.abstract_file == "rebuild"].sample(a.sample, random_state=0)
        tmp = OUTD / "rebuild_sample_articles.parquet"
        smp.to_parquet(tmp, index=False)
        subprocess.run([sys.executable, str(ROOT / "scripts/v2/rebuild_abstracts.py"), "--articles", str(tmp),
                        "--out", str(OUTD / "rebuild_sample.parquet")], check=True)
        r = pd.read_parquet(OUTD / "rebuild_sample.parquet")
        stats["rebuild_check"] = {"n": len(r), "identical": float(r.sha256_match.mean()),
                                  "rebuilt_nonempty": float((r.abstract.str.len() > 0).mean())}
        tmp.unlink()
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1, default=str))

    # ------------------------------------------------------------ archives + checksums
    top = [p for p in OUT.iterdir() if p.name != "embeddings"]
    zipdir(OUTD / "FedRetract_data.zip", OUTD, top)
    zipdir(OUTD / "FedRetract_embeddings.zip", OUTD, [OUT / "embeddings"])
    sums = [p for p in [OUTD / "FedRetract_data.zip", OUTD / "FedRetract_embeddings.zip",
                        OUT / "README.md", OUT / "DATASHEET.md"]]
    (OUTD / "SHA256SUMS.txt").write_text("".join(f"{digest(p, 'sha256')}  {p.name}\n" for p in sums))
    (OUTD / "MD5SUMS.txt").write_text("".join(f"{digest(p, 'md5')}  {p.name}\n" for p in sums))
    print(json.dumps({k: stats[k] for k in ["benchmark_text", "prospective_text", "embeddings"] +
                      (["rebuild_check"] if a.sample else [])}, indent=1, default=str))
    for p in sums:
        print(p.name, round(p.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
