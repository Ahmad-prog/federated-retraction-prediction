# Overnight Gap-Fix Report — Usman WebSci'25 corpus (2026-07-22)

Arenas: HP = his protocol (clean-180, single 80/20 split seed 42; locked baseline 0.889)
        CV = audited-172, 5-fold stratified CV x 3 seeds (statistical armor)

## Full ladder

| Model | HP acc | CV acc (±std) | Verdict |
|---|---|---|---|
| A  baseline, his features, his RF config (faithful) | **0.889** | 0.925 ±0.008 | reproduces locked baseline exactly |
| B  + certainty profiles (Gaps 3+4) | **0.917** | 0.919 ±0.008 | +2.8 on HP; CV flat vs A |
| C  + tortured phrases & perplexity (Gap 7) | 0.889 | 0.921 ±0.015 | no additional gain |
| D0 abstract-only SciBERT head (control) | 0.50–0.53 | 0.55–0.56 | chance |
| D  section-aware SciBERT + attention (Gap 1) | 0.50–0.58 | 0.53–0.55 | chance |
| E  unified (D + all features) | 0.50–0.56 | 0.51–0.56 | chance (head drowns features) |
| Fine-tuned SciBERT flat-128 (2 seeds) | 0.42–0.47 | — | at/below chance |
| Fine-tuned SciBERT full-512 (2 seeds) | 0.47 | — | at/below chance |

(Embedding variants tried: CLS, mean-pool; heads: 80 and 800 steps, standardized;
fine-tuning: 4 epochs, lr 2e-5, batch 4. All configurations at chance.)

## Per-gap impact attribution

| Gap | Impact on accuracy | Conclusion |
|---|---|---|
| Gap 3+4 (certainty profiles) | **+2.8 pts on his protocol** (0.889→0.917); flat under CV | suggestive, not statistically confirmed |
| Gap 7 (paper-mill features) | ~0 | no measurable signal on this corpus |
| Gap 1 (section-aware deep encoding) | ~0 (chance in every configuration) | **hypothesis refuted** — see below |
| LOO from E | removing any block leaves ≈ chance | neural head is the bottleneck, not features |

## Scientific findings (the real output of the night)

1. **Complete reproduction of the WebSci'25 paper achieved** — both halves:
   engineered features 0.889 (his 0.88) AND transformer failure 0.42–0.55
   (his 0.42–0.55). Their published results are genuine on the clean subset.
2. **Our Gap-1 hypothesis is refuted by our own controlled experiment.**
   Truncation was NOT why transformers failed: with full 512-token sections,
   mean-pooling, proper training, and fine-tuning, deep models remain at
   chance. The cause is corpus size (144 training documents cannot fine-tune
   a 110M-parameter encoder). Their design choice (features over deep models)
   was CORRECT at this scale.
3. **The corpus is saturated.** His feature set already reaches 0.92 CV with
   ±0.01 noise on 172 papers. No architecture change can demonstrate a
   statistically robust gain inside a ±1–2 point noise band. The bottleneck
   is N, not method.
4. Certainty profiles beat the locked 0.889 on his exact protocol (0.917)
   but the CV armor does not confirm → per our pre-agreed decision rule,
   NOT claimable as a robust improvement.

## Implication

Any honest "we improved retraction prediction" claim requires a larger
evaluation corpus — architectural gains cannot be measured at N=464.
Options: (a) scale the corpus with our RW+PMC full-text pipeline (partially
cached already); (b) switch to his DASFAA'25 concerning-citations task
(N=4,034, baselines F1 0.5–0.7, large genuine headroom).
