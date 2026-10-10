"""R4: replace the files and description of the FedRetract Zenodo DRAFT with the licence-aware release
(release_scidata/, built by r3_build_scidata_release.py). Never publishes.

Steps: read release/zenodo_state.json (draft id, bucket) -> delete draft files that differ from the release ->
upload the missing ones (resumable, 3 attempts per file):
FedRetract_data.zip, FedRetract_embeddings.zip, README.md, DATASHEET.md, SHA256SUMS.txt, MD5SUMS.txt ->
update the description -> verify the uploaded MD5 checksums. Token: ~/.zenodo_token.
Usage: python scripts/v2/r4_zenodo_update_draft.py [--dry-run | --metadata-only]
--metadata-only: update the description and check the MD5s of files uploaded by hand on the web page; no file changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.v2.r2_zenodo_upload import METADATA  # noqa: E402

API = "https://zenodo.org/api"
REL = ROOT / "release_scidata"
STATE = ROOT / "release" / "zenodo_state.json"
FILES = [REL / "FedRetract_data.zip", REL / "FedRetract_embeddings.zip", REL / "FedRetract" / "README.md",
         REL / "FedRetract" / "DATASHEET.md", REL / "SHA256SUMS.txt", REL / "MD5SUMS.txt"]

DESCRIPTION = """
<p><b>FedRetract</b> contains retracted and matched control research articles, cleaned of retraction notices and
partitioned by publisher for federated retraction prediction.</p>
<ul>
<li><b>Benchmark:</b> 121,594 English-language research articles (35,244 retracted according to the Retraction Watch
database; 86,350 controls sampled from OpenAlex within the same publisher and publication year). Identifiers, bibliographic
and authorship metadata, cleaned titles, text features, retraction date and reason, ten publisher silos and field silos,
train/validation/test splits for seeds 42&ndash;51.</li>
<li><b>Abstracts</b> are included where the article's licence allows redistribution (49,863 articles, CC BY or public
domain in one file and CC BY-SA / CC BY-NC / CC BY-NC-SA in another). For the other 34,396 abstracts the release gives a
SHA-256 checksum and a script that rebuilds them from OpenAlex with the same cleaning.</li>
<li><b>Full-text corpus:</b> 14,172 PubMed Central open-access research articles (5,090 retracted; controls matched on
journal and publication year), parsed into sections; cleaned text for the 12,549 articles whose licence allows
redistribution, identifiers and a rebuild script for the rest.</li>
<li><b>Prospective set:</b> 213 research articles retracted between 19 July and 7 October 2026, after the label snapshot.</li>
<li><b>Embeddings:</b> ModernBERT-large and Qwen3-Embedding vectors for articles whose text is redistributed or absent.</li>
<li><b>Leakage audit, documentation and code</b> (cleaning, rebuild, benchmark).</li>
</ul>
<p>Labels are derived from the Retraction Watch database (The Center for Scientific Integrity), openly distributed by
Crossref. Metadata come from OpenAlex (CC0), full text from Europe PMC / PubMed Central (per-article licences). Our files
are under CC BY 4.0 and the code under MIT; third-party text keeps its own licence. See DATASHEET.md for construction,
known biases and intended use. <b>Not for judging individual authors or articles without expert review.</b></p>
"""


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--metadata-only", action="store_true")
    a = ap.parse_args()
    st = json.loads(STATE.read_text())
    tok = (Path.home() / ".zenodo_token").read_text().strip()
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {tok}"
    dep = s.get(f"{API}/deposit/depositions/{st['id']}").json()
    assert not dep.get("submitted"), "the record is published; create a new version instead"
    old = s.get(f"{API}/deposit/depositions/{st['id']}/files").json()
    print("draft", st["id"], "state", dep.get("state"), "files now:", [f["filename"] for f in old])
    if a.dry_run:
        return
    want = {p.name: md5(p) for p in FILES}
    send = [] if a.metadata_only else FILES  # --metadata-only: files were uploaded by hand, leave them alone
    have = {f["filename"]: (f["id"], f["checksum"].replace("md5:", "")) for f in old}
    for name, (fid, chk) in (have.items() if send else []):  # remove files that are not part of the release or differ from it
        if want.get(name) != chk:
            s.delete(f"{API}/deposit/depositions/{st['id']}/files/{fid}").raise_for_status()
            print("deleted", name)
    bucket = dep["links"]["bucket"]
    for p in send:
        if have.get(p.name, (None, None))[1] == want[p.name]:
            print("already uploaded", p.name)
            continue
        for attempt in range(1, 4):  # large files can drop on slow links: retry the whole file
            try:
                with open(p, "rb") as fh:
                    r = s.put(f"{bucket}/{p.name}", data=fh, timeout=7200)
                r.raise_for_status()
                print("uploaded", p.name, r.json().get("checksum"), flush=True)
                break
            except requests.RequestException as e:
                print(f"attempt {attempt} failed for {p.name}: {type(e).__name__}", flush=True)
                if attempt == 3:
                    raise
    meta = dict(METADATA)
    meta["description"] = DESCRIPTION.strip()
    r = s.put(f"{API}/deposit/depositions/{st['id']}", json={"metadata": meta})
    r.raise_for_status()
    now = {f["filename"]: f["checksum"] for f in s.get(f"{API}/deposit/depositions/{st['id']}/files").json()}
    ok = all(now.get(p.name, "").replace("md5:", "") == md5(p) for p in FILES)
    print("files in draft:", sorted(now), "md5 verified:", ok, "| still a draft (not published)")


if __name__ == "__main__":
    main()
