# IP&MC 2026 Submission Checklist — Federated Retraction Prediction

**Target:** Information Processing & Management Conference 2026 (IP&MC2026),
Wuhan. Deadline **31 July 2026**. Conference→journal pathway: accepted full
papers eligible for a special issue in *IP&M* (IF 8.1) or *DIM* (IF 8.3).

**Files here** (`paper/ipm/`):
- `main.tex` — Elsevier `elsarticle` format (review mode, single column, line
  numbers), Vancouver-numbered references. Compiles clean with pdflatex.
- `main.pdf` — compiled proof (18 pp in review/double-spaced mode ≈ 10–11 pp
  final two-column).
- `figures/` — all 7 figures.

## What's DONE
- [x] Converted from LNCS (DASFAA) to IP&M `elsarticle` journal format.
- [x] Reframed for the information-science audience (digital libraries /
      scholarly communication / integrity of the scientific record); notes that
      it builds on an IP&M-published bibliometric-retraction paper.
- [x] Added Elsevier-required sections: Data availability, Declaration of
      competing interest, Declaration of generative-AI use.
- [x] All numbers, tables, figures carried over from the finished study.
- [x] Compiles with no undefined references/citations.

## RESOLVED this round
- [x] **Authorship** — Muhammad Ahmad (corresponding) + **Muhammad Tanvir Afzal,
      Gisma University of Applied Sciences GmbH, Potsdam, Germany**
      (tanvir.afzal@gisma.com) added. NOTE: Ahmad is still listed as
      "Independent Researcher, Pakistan" — if you are a Gisma student, tell me
      and I'll move you to the Gisma affiliation.
- [x] **Repository URL** — public repo created and inserted in Data-availability:
      https://github.com/Ahmad-prog/federated-retraction-prediction
- [x] **Abstract** trimmed to 243 words (IP&M limit: 250).

## STILL TO DECIDE before submitting
1. **Track selection** — pick the best-fit thematic track on the portal
   (scholarly information / digital libraries / responsible-IS / FIRM).
2. **Page/length limits** — I researched this (DYOR): the IP&MC/IP&M guidance
   points to the IP&M journal guide, which sets an **abstract ≤ 250 words** (done)
   but **no hard page limit** (it's a journal). No per-track page cap was
   published at the time of writing — confirm on the submission portal. If a cap
   appears, switch `\documentclass[review,12pt]` →
   `\documentclass[final,5p,times,twocolumn]` for the compact two-column form.

## How to compile
- **Overleaf (recommended for final):** upload `main.tex` + `figures/`, pick
  "Elsevier (elsarticle)" — it has the class built in. Compile with pdfLaTeX.
- **Locally (TinyTeX, already set up):**
  `export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH" && pdflatex main && pdflatex main`

## Note on venue strategy
This is the **FL paper** → IP&M. The **certainty-profiles paper**
(`paper/unified/`) remains free for another venue. The two share a research
line and should cross-cite; keep them at different venues (no double
submission of the same paper).
