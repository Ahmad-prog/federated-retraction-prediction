"""Europe PMC REST client (v2).

Fixes bug B1/B2 of the v1 pipeline: the base URL is `webservices` (plural), and an
HTTP failure is never cached as an empty result. Callers get either a parsed
response or an exception; only successful responses are written to cache.
"""
from __future__ import annotations

import gzip
import json
import random
import threading
import time
from pathlib import Path

import requests

BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
_local = threading.local()


class EPMCError(RuntimeError):
    pass


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.headers["User-Agent"] = "fedretract-research/2.0"
        _local.s = s
    return _local.s


def get(path: str, params: dict | None = None, retries: int = 6,
        allow_404: bool = False) -> requests.Response | None:
    """GET BASE/path. Returns the response, None on 404 if allow_404, else raises."""
    url = f"{BASE}/{path.lstrip('/')}"
    last = None
    for attempt in range(retries):
        try:
            r = _session().get(url, params=params, timeout=90)
            if r.status_code == 200:
                return r
            if r.status_code == 404 and allow_404:
                return None
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except requests.RequestException as e:
            last = repr(e)
        time.sleep(min(60, 2 ** attempt + random.random()))
    raise EPMCError(f"{url} params={params} failed: {last}")


def search(query: str, page_size: int = 1000, cursor: str = "*",
           result_type: str = "lite") -> dict:
    r = get("search", {"query": query, "format": "json", "pageSize": page_size,
                       "cursorMark": cursor, "resultType": result_type})
    return r.json()


def cached_json(path: Path, fn):
    """Return cached JSON at `path` or compute it with fn() and cache it.
    fn() must raise on failure so that failures are never cached."""
    if path.exists():
        return json.loads(path.read_text())
    out = fn()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(out))
    tmp.replace(path)
    return out


NCBI_OAI = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"
_ncbi_lock = threading.Lock()
_ncbi_last = [0.0]


def _ncbi_oai(pmcid: str) -> bytes | None:
    """Fallback source: NCBI PMC OAI-PMH (JATS inside <metadata>). <= 3 requests/s."""
    for attempt in range(4):
        with _ncbi_lock:
            wait = 0.4 - (time.time() - _ncbi_last[0])
            if wait > 0:
                time.sleep(wait)
            _ncbi_last[0] = time.time()
        try:
            r = _session().get(NCBI_OAI, params={
                "verb": "GetRecord", "metadataPrefix": "pmc",
                "identifier": f"oai:pubmedcentral.nih.gov:{pmcid.replace('PMC', '')}"}, timeout=90)
            if r.status_code == 200 and b"<article" in r.content:
                return r.content
            if r.status_code in (400, 404) or (r.status_code == 200 and b"idDoesNotExist" in r.content):
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return None


def fetch_fulltext_xml(pmcid: str, out_dir: Path) -> str:
    """Download full-text JATS XML for a PMCID to out_dir/<PMCID>.xml.gz.
    Europe PMC first; on a server error, NCBI PMC OAI. Returns 'ok', 'ok_ncbi',
    'cached' or 'missing' (no full text from either source)."""
    f = out_dir / f"{pmcid}.xml.gz"
    miss = out_dir / f"{pmcid}.missing"
    if f.exists():
        return "cached"
    if miss.exists():
        return "missing"
    try:
        r = get(f"{pmcid}/fullTextXML", allow_404=True, retries=3)
        content = r.content if r is not None else None
        src = "ok"
    except EPMCError:
        content, src = None, "ok_ncbi"
    if content is None or not content.lstrip().startswith(b"<"):
        content, src = _ncbi_oai(pmcid), "ok_ncbi"
    if content is None:
        miss.touch()
        return "missing"
    tmp = f.with_suffix(".tmp")
    with gzip.open(tmp, "wb") as g:
        g.write(content)
    tmp.replace(f)
    return src
