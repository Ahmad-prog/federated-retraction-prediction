"""R2: deposit FedRetract on Zenodo (REST API). Three explicit steps, nothing is public until
`publish`:

  create   create a draft deposition with full metadata and a RESERVED DOI (prints the DOI)
  upload   upload release/FedRetract_*.zip, README.md, DATASHEET.md, SHA256SUMS.txt to the draft
  publish  publish the draft (irreversible: published records cannot be deleted)

The access token is read from ~/.zenodo_token (scopes: deposit:write, deposit:actions).
State (deposition id, bucket, DOI) is kept in release/zenodo_state.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.v2 import ROOT  # noqa: E402

API = "https://zenodo.org/api"
REL = ROOT / "release"
STATE = REL / "zenodo_state.json"
FILES = ["FedRetract_data.zip", "FedRetract_embeddings.zip", "FedRetract/README.md",
         "FedRetract/DATASHEET.md", "SHA256SUMS.txt"]

DESCRIPTION = """
<p><b>FedRetract</b> is a leakage-audited benchmark for predicting scientific retractions, with federated (publisher-silo)
splits, a journal&ndash;year matched open-access full-text corpus, and a prospective test set.</p>
<ul>
<li><b>Benchmark:</b> 121,594 English-language research articles (35,244 retracted, controls matched on publisher and
year). Cleaned title and abstract, bibliographic and authorship metadata, text features, time-aware citation features,
retraction date and reason. Federated splits: 10 publisher silos, field silos, 10 seeds.</li>
<li><b>Full-text corpus:</b> 14,172 PubMed Central open-access research articles (5,090 retracted; controls matched on
journal and publication year), parsed into sections. Cleaned text is included for the 12,549 articles whose licence
allows redistribution (CC BY / CC0 / CC BY-NC / CC BY-NC-SA); the rest are given as identifiers and labels with a
rebuild script.</li>
<li><b>Prospective test set:</b> 213 articles retracted after the 2026-07-19 snapshot.</li>
<li><b>Embeddings:</b> ModernBERT-large and Qwen3-Embedding vectors for abstracts and full text.</li>
<li><b>Leakage audit:</b> retraction notices, notice sentences, "RETRACTED" title residues, non-research items,
non-English items and notice-only bodies are removed identically from both classes. A keyword-only classifier scores at
chance after cleaning.</li>
</ul>
<p>Labels are derived from the Retraction Watch database (The Center for Scientific Integrity), made openly available by
Crossref. Metadata comes from OpenAlex (CC0), full text from Europe PMC / PubMed Central (per-article licences). See
DATASHEET.md for construction, known biases (topic skew towards paper-mill areas, country disparities in false
positives) and intended use. <b>Not for judging individual authors or articles without expert review.</b></p>
"""

METADATA = {
    "upload_type": "dataset",
    "title": "FedRetract: A Leakage-Audited Benchmark and Matched Full-Text Corpus for Federated Retraction Prediction",
    "creators": [
        {"name": "Ahmad, Muhammad", "affiliation": "Shifa Tameer-e-Millat University, Islamabad, Pakistan",
         "orcid": "0009-0005-1403-8928"},
        {"name": "Afzal, Muhammad Tanvir", "affiliation": "Gisma University of Applied Sciences, Potsdam, Germany"},
    ],
    "description": DESCRIPTION.strip(),
    "access_right": "open",
    "license": "cc-by-4.0",
    "keywords": ["retraction prediction", "research integrity", "federated learning", "scientometrics",
                 "benchmark", "leakage audit", "paper mills", "full text"],
    "notes": "Labels: Retraction Watch database (Center for Scientific Integrity) via Crossref. Metadata: OpenAlex (CC0). "
             "Full text: PubMed Central open-access subset via Europe PMC; each article keeps its own licence "
             "(see fulltext/index.parquet). Code: MIT licence.",
    "prereserve_doi": True,
}


def session() -> requests.Session:
    tok = (Path.home() / ".zenodo_token").read_text().strip()
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {tok}"
    return s


def check(r: requests.Response) -> dict:
    if r.status_code >= 400:
        raise SystemExit(f"Zenodo error {r.status_code}: {r.text[:1000]}")
    return r.json() if r.content else {}


def create(s) -> None:
    if STATE.exists():
        raise SystemExit(f"draft already exists: {STATE.read_text()}")
    d = check(s.post(f"{API}/deposit/depositions", json={"metadata": METADATA}))
    st = {"id": d["id"], "bucket": d["links"]["bucket"], "html": d["links"]["html"],
          "doi": d["metadata"]["prereserve_doi"]["doi"]}
    STATE.write_text(json.dumps(st, indent=1))
    print(json.dumps(st, indent=1))


def upload(s) -> None:
    st = json.loads(STATE.read_text())
    for f in FILES:
        p = REL / f
        with open(p, "rb") as fh:
            r = check(s.put(f"{st['bucket']}/{p.name}", data=fh, timeout=7200))
        print(f"uploaded {p.name}: {r.get('size')} bytes, checksum {r.get('checksum')}", flush=True)


def publish(s) -> None:
    st = json.loads(STATE.read_text())
    d = check(s.post(f"{API}/deposit/depositions/{st['id']}/actions/publish"))
    st.update(published=True, doi=d.get("doi"), conceptdoi=d.get("conceptdoi"), record=d["links"].get("record_html"))
    STATE.write_text(json.dumps(st, indent=1))
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["create", "upload", "publish", "status"])
    a = ap.parse_args()
    s = session()
    if a.step == "status":
        st = json.loads(STATE.read_text())
        d = check(s.get(f"{API}/deposit/depositions/{st['id']}"))
        print(json.dumps({"state": d["state"], "submitted": d["submitted"],
                          "files": [(f["filename"], f["filesize"]) for f in d.get("files", [])],
                          "doi": st["doi"]}, indent=1))
    else:
        {"create": create, "upload": upload, "publish": publish}[a.step](s)
