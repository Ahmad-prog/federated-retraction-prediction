"""Fetch full-text section features for the corpus from Europe PMC.

Stage 1: map DOIs -> PMCIDs (batched search queries, cached per batch).
Stage 2: download fullTextXML per PMCID (cached per article).
Stage 3: parse <sec> blocks into abstract/methods/results/conclusion text and
         compute section-level readability/certainty features -> parquet.

Europe PMC has no daily budget; polite rate only. Fully resumable.
"""
from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.build_features import text_features

EPMC = "https://www.ebi.ac.uk/europepmc/webservice/rest"
MAP_CACHE = ROOT / "data/cache/pmcid_map"
XML_CACHE = ROOT / "data/cache/fulltext_xml"
MAP_CACHE.mkdir(parents=True, exist_ok=True)
XML_CACHE.mkdir(parents=True, exist_ok=True)

SEC_PATTERNS = {
    "methods": re.compile(r"method|material|experimental", re.I),
    "results": re.compile(r"result|finding", re.I),
    "conclusion": re.compile(r"conclusion|discussion|summary", re.I),
}


def get(url, params=None, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return None


def map_batch(job):
    bi, dois = job
    bfile = MAP_CACHE / f"batch_{bi:06d}.json"
    if bfile.exists():
        return json.loads(bfile.read_text())
    q = " OR ".join(f'DOI:"{d}"' for d in dois)
    r = get(f"{EPMC}/search", {"query": q, "format": "json", "pageSize": len(dois) * 2,
                               "resultType": "idlist"})
    out = {}
    if r is not None:
        for res in (r.json().get("resultList") or {}).get("result", []):
            doi = (res.get("doi") or "").lower()
            pmcid = res.get("pmcid")
            if doi and pmcid:
                out[doi] = pmcid
    bfile.write_text(json.dumps(out))
    time.sleep(0.1)
    return out


def fetch_xml(pmcid: str) -> str | None:
    f = XML_CACHE / f"{pmcid}.xml"
    if f.exists():
        return str(f)
    r = get(f"{EPMC}/{pmcid}/fullTextXML")
    if r is None or not r.text.strip().startswith("<"):
        f.with_suffix(".missing").touch()
        return None
    f.write_text(r.text, encoding="utf-8")
    time.sleep(0.1)
    return str(f)


def parse_sections(xml_path: str) -> dict[str, str]:
    try:
        root = ElementTree.parse(xml_path).getroot()
    except ElementTree.ParseError:
        return {}
    out = {"abstract": " ".join(t for t in
           (" ".join(a.itertext()) for a in root.iter("abstract")) if t)}
    for sec in root.iter("sec"):
        title_el = sec.find("title")
        title = " ".join(title_el.itertext()) if title_el is not None else ""
        body = " ".join(sec.itertext())
        for key, pat in SEC_PATTERNS.items():
            if pat.search(title or ""):
                out[key] = out.get(key, "") + " " + body
    return out


def main() -> None:
    corpus = pd.read_parquet(ROOT / "data/processed/corpus.parquet")[["doi", "retracted"]]
    dois = sorted({d for d in corpus["doi"] if isinstance(d, str) and d})
    batches = [dois[i:i + 20] for i in range(0, len(dois), 20)]

    print(f"Stage 1: mapping {len(dois)} DOIs in {len(batches)} batches", flush=True)
    doi2pmc = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for bi, res in enumerate(ex.map(map_batch, enumerate(batches))):
            doi2pmc.update(res)
            if bi % 500 == 0:
                print(f"  mapped batch {bi}/{len(batches)}, hits {len(doi2pmc)}", flush=True)
    print(f"PMCIDs found: {len(doi2pmc)}", flush=True)

    pmcids = sorted(set(doi2pmc.values()))
    print(f"Stage 2: fetching {len(pmcids)} full texts", flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        done = 0
        for _ in ex.map(fetch_xml, pmcids):
            done += 1
            if done % 2000 == 0:
                print(f"  fetched {done}/{len(pmcids)}", flush=True)

    print("Stage 3: parsing + features", flush=True)
    pmc2doi = {v: k for k, v in doi2pmc.items()}
    rows = []
    for i, pmcid in enumerate(pmcids):
        f = XML_CACHE / f"{pmcid}.xml"
        if not f.exists():
            continue
        secs = parse_sections(str(f))
        row = {"doi": pmc2doi[pmcid], "pmcid": pmcid}
        got = 0
        for tag in ["abstract", "methods", "results", "conclusion"]:
            txt = secs.get(tag, "")
            tf = text_features(txt)
            got += 1 if len(txt) > 200 else 0
            row.update({f"ft_{tag}_{k}": v for k, v in tf.items()})
        row["ft_sections_found"] = got
        rows.append(row)
        if i % 5000 == 0:
            print(f"  parsed {i}/{len(pmcids)}", flush=True)
    ft = pd.DataFrame(rows)
    ft.to_parquet(ROOT / "data/processed/fulltext_features.parquet", index=False)
    print(f"FULLTEXT_FEATURES_READY rows={len(ft)} "
          f"with>=3sections={int((ft['ft_sections_found']>=3).sum())}")


if __name__ == "__main__":
    main()
