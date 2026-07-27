# Experiment Summary — Federated Retraction Prediction (2026-07-20)

Corpus: 138,537 papers (37,648 retracted via Retraction Watch + 100,889 publisher/year-matched controls, OpenAlex metadata). AUPRC random baseline ≈ 0.27 (our corpus) / 0.50 (Usman's balanced corpus). 3 seeds unless noted.

| # | Experiment | Result (AUPRC) | Interpretation | Simple terms |
|---|---|---|---|---|
| 1 | Main benchmark, publisher silos | local 0.315 · **FedAvg 0.558** · central MLP 0.609 · central XGB 0.633 | FL recovers 92% of centralized (same architecture), +77% over local | Publishers alone: weak. Sharing only model weights: nearly as good as impossible full pooling. **The headline.** |
| 2 | Main benchmark, discipline silos | FedAvg 0.585, local 0.326 | Conclusion robust to silo definition | Works whether data is split by company or by science field |
| 3 | Usman-style RF at scale | central RF 0.598 | His WebSci'25 architecture validated on 180× data | His recipe holds up at scale — respectful, citable |
| 4 | FedProx / FedBal | 0.556 / 0.542 | FedProx ≈ FedAvg; FedBal unneeded (matched design equalizes imbalance) | Honest negative on our own variant |
| 5 | DP privacy sweep | σ=0.3: 0.514 (95% of 0.540) · σ=1.0: collapse | Meaningful protection ≈ 5% cost; formal ε weak with only 10 clients (discussed honestly) | Privacy "static" is affordable until the dial goes too far |
| 6 | Forward transfer → his 232+232 | full features 0.51 (≈chance) · **text-only 0.691** | Metadata is domain-specific; writing-style signal transfers | Our big model recognizes his hand-picked cases from writing style alone |
| 7 | Reverse transfer: his RF → our 138k | 0.283 (≈chance 0.27) | 464-paper models don't generalize | Small hand-made models collapse in the real world — the scale argument in one number |
| 8 | His corpus, abstract-only CV | 0.525 (≈chance 0.50) | **His 0.88 was almost entirely full-text value** | We measured what the paywalled text is worth: nearly everything. The #1 justification for federating. |
| 9 | Scale curve (500→110k train) | 0.403 → 0.498 → 0.573 → 0.618 → 0.633, monotonic | Performance rises with data throughout | More data keeps helping — why 138k beats 225 |
| 10 | Feature ablation | metadata 0.593 · text 0.381 · all 0.633 | In-domain, metadata dominates; text adds +0.04 but is what generalizes (see #6) | Two feature kinds do two jobs: metadata = raw accuracy, writing style = portability |
| 11 | Temporal split (≤2018 → ≥2019) | central 0.436 · FL 0.407 (93% of central) | Future prediction harder (retraction lag = label noise) but FL gap unchanged | Predicting tomorrow is harder than explaining yesterday — true for everyone; FL costs no extra |
| 12 | Per-silo: local vs federated | big silos: local wins in-silo (Elsevier 0.708 vs 0.578); small/weak silos: FL wins (PLoS, Spandidos) | Local specializes to own distribution but fails globally (see #1); federation helps data-poor silos most → motivates personalized FL (future work) | On home turf big publishers do fine; the shared model is what covers everyone everywhere — and small publishers gain most |
| 13 | K sweep (5→20 silos) | 0.574 / 0.558 / 0.560 / 0.558 | Stable as consortium grows | Federation doesn't degrade when more publishers join |
| 14 | Feature importance | citations (0.136), early citations (0.128), year, refs, authors; best text: Flesch-Kincaid | Citation dynamics top, echoing Usman's citation-centric findings | What the model looks at, and it matches prior intuition |

## The paper's argument in 4 numbers
0.88 → 0.525: his accuracy collapses without paywalled full text (its value, measured).
0.283: his small-corpus model fails on the open world (scale needed).
0.315 → 0.558: federation fixes what isolation breaks (our contribution).
0.558 vs 0.609: the price of not sharing data ≈ 8% (why it's deployable).

## Assets ready
- results/*.csv (14 experiments), paper/figures/*.png (5 main figures)
- paper/draft.md (narrative), Usman's 4 datasets+code in data/raw/
- Venue: DASFAA 2027 (LNCS 16pp, ~late-Oct-2026 deadline, double-blind)
