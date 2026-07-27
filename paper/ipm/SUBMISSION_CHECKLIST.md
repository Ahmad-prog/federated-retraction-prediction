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

## YOU MUST DECIDE before submitting (in `main.tex`)
1. **Authorship** — currently `Muhammad Ahmad` (Independent Researcher, Pakistan)
   only. This work reproduces/extends Usman & Balke. Confirm with your
   supervisor whether **Usman / Wolf-Tilo Balke should be co-authors** and add
   an affiliation (university) if applicable. Edit the `\author`/`\affiliation`
   block near the top of `main.tex`.
2. **Repository URL** — the Data-availability section says "URL to be inserted
   upon acceptance." If you have a public GitHub repo, put it in now (IP&MC is
   not double-blind — it's a named submission).
3. **Track selection** — on the submission site, pick the best-fit thematic
   track. Recommended: a track under IP&M covering scholarly information /
   digital libraries / responsible information systems (e.g. **FIRM** or a
   data-intelligence / responsible-IS track). Confirm exact track list on the
   portal.
4. **Page/length limits** — IP&M is a journal (no hard page cap), but the
   conference track may specify one. Check the portal; if a limit applies, we
   can switch `\documentclass[review,12pt]` → `\documentclass[final,5p,times,twocolumn]`
   for the compact two-column form.

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
