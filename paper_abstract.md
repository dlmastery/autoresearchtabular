# Paper abstract — autoresearchtabular

## TL;DR

We run 25 experiments × 14 backbones (= 350 experiments) on the **Higgs UCI
tabular benchmark** (Baldi 2014, Nat. Comm., arXiv:1402.4735) with the frozen
10 M / 500 k / 500 k Baldi split. Every experiment passes through three
programmatic gates — a 7-auditor data-split audit, a Citation Rigor gate
that rejects undocumented hyperparameter changes, and a Reasoning Blob
Completeness gate that requires a 7-section research blob (`diagnose / cite /
hypothesize / predict / run / analyze / checkpoint`). The composite metric
`min(test_auc, val_auc) − 0.1·|test_auc − val_auc|` is SHA-256 fingerprinted
at runner boot.

## Headline result

*Populated after the campaign completes.*

The published baselines we are racing against:
- **XGBoost (Chen 2016 reproduction):** test AUROC ≈ 0.864
- **FT-Transformer (Gorishniy 2021):** test AUROC ≈ 0.880
- **SAINT (Somepalli 2022):** test AUROC ≈ 0.881
- **TabM (Gorishniy 2024):** test AUROC ≈ 0.886

Headroom from XGBoost to deep tabular SOTA: ~2.2 AUROC points,
spread over 10 + years and dozens of papers.

## Why this is honest

- The **Baldi 2014 split is contiguous and frozen**: rows
  `[0, 10M) / [10M, 10.5M) / [10.5M, 11M)`. Every paper since 2014 uses
  this exact split. Our audit re-derives the SHA-256 fingerprint of the
  union and fails the run if it ever drifts.
- **No tuning on test.** All 25 per-backbone experiments are scored on
  `val`; only the per-backbone winner is rerun once on `test` to record
  the leaderboard row. The composite formula combines val and test such
  that aggressive val-overfitting *hurts* the composite (large gap term).
- **Citation Rigor is enforced in code.** Every HP change cites a paper
  with section / table / figure pointer. The reasoning gate rejects bare
  URLs and "best practice" folklore.
- **All artefacts are committed.** Per-experiment recipe, reasoning blob,
  per-row prediction CSV (val + test), and the data-split fingerprint
  are pushed to GitHub at every checkpoint. A third-party can replay the
  full audit from `autoresearch_results/`.

## What this is NOT

- **Not** a claim of state-of-the-art on Higgs. The campaign is
  publishable because of the *audit discipline*, not because the absolute
  number is novel.
- **Not** a single-architecture deep dive. We compare 14 backbones with a
  fixed budget per backbone; this is a horizontal sweep, not a vertical
  one.
- **Not** a hyperparameter optimisation paper. Each backbone gets 25
  paper-cited HP changes, not 1000 random samples; the goal is *grounded*
  selection, not exhaustive search.
