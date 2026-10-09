"""Shared text cleaning for every v2 script (titles, abstracts, full text).

Found by the leakage audit (scripts/v2/d1_leak_audit.py) after a first cleaning pass:
  * titles such as "RETRACTED ARTICLE: ..." kept the residue "ARTICLE:" -> label leak
  * notice abstracts were missed when they said "removed", "the publisher",
    "Wiley Online Library", "Editor-in-Chief", "duplicate", "published in error", ...
  * article-type mismatch: retracted items are research articles, but OpenAlex
    type:article controls include reviews, case reports, letters, editorials and
    CheminForm abstracts -> a shortcut unrelated to integrity
All rules are applied identically to retracted articles and controls.
"""
from __future__ import annotations

import html
import re

# ---- titles
_PREFIX = re.compile(
    r"^\s*[\[(]?\s*(?:retracted(?:\s+article)?|retraction(?:\s+note)?|withdrawn(?:\s+article)?|"
    r"notice\s+of\s+retraction|retraction\s+notice|article\s+removed|removed|"
    r"expression\s+of\s+concern|duplicate(?:\s+publication)?|article|erratum|corrigendum)"
    r"\s*[\])]?\s*[:.\-–—]+\s*", re.I)
_TITLE_LEAK = re.compile(r"retract|withdr|removed|notice of|erratum|corrigend|"
                         r"expression of concern|duplicate publication", re.I)

# ---- notices (an abstract/section that is, or is contaminated by, an editorial notice)
_NOTICE = re.compile(
    r"retract|withdr[ae]wn|withdraw|has been removed|been removed|article removed|"
    r"policy on article withdrawal|above article|editor[- ]in[- ]chief|by agreement between|"
    r"at the request of|erroneously|published in error|in error\b|duplicate publication|"
    r"concerns? (?:was|were|have been|has been) raised|systematic manipulation|"
    r"peer[- ]review process|this (?:article|paper) (?:was|has been|is)|the publisher|"
    r"wiley online library|elsevier policy|expression of concern|erratum|corrigend|"
    r"since (?:its )?publication|after (?:the )?publication|discovered (?:an? |two |several )?errors?|"
    r"error(?:s)? (?:was|were|has been|have been) (?:found|identified|discovered|noted)|"
    r"this publication is (?:available|free)|original article|has been corrected|"
    r"(?:the )?authors? (?:would like|wish) to (?:correct|apologi[sz]e|retract|notify)|"
    r"doi(?:\.org/|:)?\s*10\.\d{4,}|"
    r"(?:peer review|review process)\W+(?:\w+\W+){0,6}compromised|"
    r"(?:research|publication|scientific|academic) (?:integrity|misconduct)|misconduct", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")

# ---- article types that are not research articles (symmetric exclusion)
_NONRESEARCH = re.compile(
    r"^\s*(?:review|a review|mini[- ]review|systematic review|case report|case series|"
    r"letter|editorial|commentary|comment on|reply|response to|correspondence|erratum|"
    r"corrigendum|book review|obituary|interview|news|in brief|cheminform|abstracts?\b|"
    r"proceedings|meeting|conference|introduction\s*$|preface|foreword)|"
    r"\b(?:in this review|this review|we review|we report an? (?:rare )?case|we present an? (?:rare )?case|"
    r"case of an? \d+[- ]year|\d+[- ]year[- ]old (?:male|female|man|woman|boy|girl|patient))\b|"
    r"\b(?:a (?:systematic |narrative |scoping |literature )?review|case report|"
    r"meta-analysis|cheminform abstract)\b", re.I)


_TAGS = re.compile(r"<[^>]{1,400}>")


def strip_markup(text: str | None) -> str:
    """Remove XML/MathML/HTML tags and entities (publisher formatting artefacts)."""
    t = html.unescape(html.unescape(text or ""))  # twice: handles &amp;lt; double encoding
    t = _TAGS.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def clean_title(t: str | None) -> str:
    t = strip_markup(t)
    for _ in range(3):  # strip stacked prefixes, e.g. "RETRACTED ARTICLE: Retraction Note: ..."
        t2 = _PREFIX.sub("", t)
        if t2 == t:
            break
        t = t2
    return "" if _TITLE_LEAK.search(t) else t


def is_notice(text: str | None) -> bool:
    return bool(_NOTICE.search(text or ""))


def clean_text(text: str | None) -> tuple[str, int]:
    """Drop every sentence matching the notice pattern. Returns (text, n_removed)."""
    kept, removed = [], 0
    for s in _SENT.split((text or "").strip()):
        if _NOTICE.search(s):
            removed += 1
        elif s:
            kept.append(s)
    return " ".join(kept), removed


def clean_abstract(text: str | None, max_removed_share: float = 0.34) -> tuple[str, bool, int]:
    """Return (clean_abstract, contaminated, n_removed). If more than a third of the
    sentences are notice-like, the whole abstract is treated as a notice and dropped."""
    text = strip_markup(text)
    if not text:
        return "", False, 0
    n = max(1, len(_SENT.split(text)))
    clean, removed = clean_text(text)
    if removed / n > max_removed_share or len(clean) < 40:
        return "", removed > 0, removed
    return clean, removed > 0, removed


def is_nonresearch(title: str | None, abstract: str | None = None) -> bool:
    """Reviews, case reports, letters, editorials, ... (from the title or the abstract)."""
    return bool(_NONRESEARCH.search(title or "")) or bool(_NONRESEARCH.search(abstract or ""))

_EN_STOP = set("the of and to in a is that for with as on are by this was be from at an or which were we "
               "have has can these it its their between not also using results based study".split())
_WORD = re.compile(r"[a-z]+")


def is_english(text: str | None, min_share: float = 0.12) -> bool:
    """Heuristic: share of common English function words among tokens (titles alone are
    short, so callers should pass title + abstract)."""
    toks = _WORD.findall((text or "").lower())
    if len(toks) < 6:
        return True  # too short to judge; keep
    return sum(t in _EN_STOP for t in toks) / len(toks) >= min_share
