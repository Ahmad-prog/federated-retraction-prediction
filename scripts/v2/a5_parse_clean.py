"""A5: parse JATS full text into sections and remove label leakage (C3).

Leakage removed identically for retracted articles and controls (rules in src/v2/cleaning.py):
  * <related-article> links (e.g. retraction-forward), <notes>, <fn-group>, <ack>,
    <back> (references, funding, COI), <processing-meta>, <custom-meta-group>
  * "RETRACTED:" / "[Retracted]" / "WITHDRAWN:" title prefixes
  * every sentence mentioning retraction, withdrawal, expression of concern,
    erratum/corrigendum/correction notice
  * documents that are themselves notices (article-type retraction/correction/...)
The number of removed sentences is kept per document for the leakage audit (R0.4).

Output: data_v2/fulltext_clean.parquet
"""
from __future__ import annotations

import gzip
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.v2 import DATA2  # noqa: E402
from src.v2.cleaning import clean_text, clean_title, is_nonresearch  # noqa: E402

XML = DATA2 / "fulltext_xml"
DROP_TAGS = ["related-article", "notes", "fn-group", "ack", "back", "processing-meta",
             "custom-meta-group", "funding-group", "author-notes", "permissions",
             "table-wrap-foot", "supplementary-material"]
NOTICE_TYPES = {"retraction", "correction", "expression-of-concern", "addendum",
                "editorial", "letter", "article-commentary", "reply", "news", "obituary"}
SECTIONS = {
    "introduction": re.compile(r"introduc|background", re.I),
    "methods": re.compile(r"method|material|experiment|procedure|patients|participants|"
                          r"study design|data collection", re.I),
    "results": re.compile(r"result|finding", re.I),
    "discussion": re.compile(r"discussion", re.I),
    "conclusion": re.compile(r"conclu|summary|outlook", re.I),
}


def _text(el) -> str:
    return re.sub(r"\s+", " ", " ".join(el.itertext())).strip() if el is not None else ""


def _clean(text: str) -> tuple[str, int]:
    return clean_text(text)


def parse(path: Path) -> dict:
    pmcid = path.name.split(".")[0]
    try:
        with gzip.open(path, "rb") as f:
            root = etree.fromstring(f.read(), etree.XMLParser(recover=True, huge_tree=True))
    except Exception:  # noqa: BLE001
        return {"pmcid": pmcid, "parse_ok": False}
    if root is None:
        return {"pmcid": pmcid, "parse_ok": False}
    for el in root.iter():  # namespace-agnostic (NCBI OAI JATS uses namespaces)
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    art = root if root.tag == "article" else root.find(".//article")
    art = art if art is not None else root
    atype = (art.get("article-type") or "").lower()
    for tag in DROP_TAGS:
        for el in list(art.iter(tag)):
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)

    raw_title = _text(art.find(".//title-group/article-title"))
    title = clean_title(raw_title)
    abstract = " ".join(_text(a) for a in art.iter("abstract")
                        if (a.get("abstract-type") or "") not in ("teaser", "graphical"))
    sec_text = {k: [] for k in SECTIONS}
    body = art.find(".//body")
    for sec in (body.findall("sec") if body is not None else []):
        head = _text(sec.find("title"))
        txt = _text(sec)
        for k, pat in SECTIONS.items():
            if pat.search(head):
                sec_text[k].append(txt)
                break
    body_txt = _text(body)

    # research articles only, identically for both classes (article-type shortcut)
    out = {"pmcid": pmcid, "parse_ok": True, "article_type": atype,
           "is_notice": (atype in NOTICE_TYPES or atype not in ("research-article", "")
                         or is_nonresearch(raw_title, abstract))}
    removed = 0
    out["title"] = title
    for name, txt in [("abstract", abstract), ("body", body_txt)] + [
            (k, " ".join(v)) for k, v in sec_text.items()]:
        out[name], r = _clean(txt)
        removed += r
    out["n_leak_sentences_removed"] = removed
    out["n_body_words"] = len(out["body"].split())
    out["n_sections_found"] = sum(bool(out[k]) for k in SECTIONS)
    return out


def main() -> None:
    files = sorted(XML.glob("*.xml.gz"))
    print(f"parsing {len(files)} documents", flush=True)
    with ProcessPoolExecutor(max_workers=40) as ex:
        rows = list(ex.map(parse, files, chunksize=64))
    df = pd.DataFrame(rows)
    df.to_parquet(DATA2 / "fulltext_clean.parquet", index=False)
    ok = df[df["parse_ok"]]
    print(f"parsed ok: {len(ok)}/{len(df)}; notices: {int(ok['is_notice'].sum())}")
    print(f"median body words: {ok['n_body_words'].median():.0f}; "
          f">=3 sections: {(ok['n_sections_found'] >= 3).mean():.3f}")
    print(f"docs with leak sentences removed: {(ok['n_leak_sentences_removed'] > 0).mean():.3f}")


if __name__ == "__main__":
    main()
