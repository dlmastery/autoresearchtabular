# How I let an autonomous loop run 350 tabular ML experiments overnight — and why every single one had to cite a paper before training started

*A practitioner's note on the autoresearchtabular project.*

---

If you've ever run a serious tabular ML benchmark, you know the
disappointment: the leaderboard says one method beats another, but the
authors used a different split, swapped a metric, or buried twenty
unsuccessful runs to publish the lucky one. The Higgs UCI benchmark from
the 2014 Baldi *Nature Communications* paper is one of the few large
tabular benchmarks where the split has actually held up over a decade
(11 M rows, 28 features, frozen 10 M / 500 k / 500 k by row index), but
even there, every paper handles its own preprocessing, its own
hyperparameter selection, and its own choice of "what counts as the
final number."

I wanted to do better than that.

This post is about **AUTORESEARCHTABULAR**, an autonomous research loop
I let run for 24 hours on a single workstation. It executed **25
experiments × 14 backbones = 350 experiments** on Higgs UCI under three
programmatic gates that have to pass *before* training is allowed to
start. The campaign is the third in a series — the same gates were
validated on FX forecasting (autoresearch, 79 experiments) and computer
vision (autoresearchimage, 21 experiments on PathMNIST and
WILDS-Camelyon17). This is what happens when you take the audit
discipline seriously enough to encode it as a runner refusal mode.

## The three gates

**Gate 1: data-split audit.** Seven auditors run before *any*
experiment. They verify pairwise disjointness of train / val / test row
indices, compliance with the Baldi 2014 row-range protocol, class
balance on every split, size floors, no row-identifier feature that
would let the model memorise, byte-identical reproducibility on reload,
and feature-column consistency. The audit produces a SHA-256 fingerprint
of the row-index union; every experiment row records this fingerprint.
If it ever changes, every prior leaderboard row is invalidated by
construction. The runner refuses to start if the audit hasn't produced a
green report in the last campaign cycle.

**Gate 2: Citation Rigor.** Every hyperparameter that differs from a
backbone's registry default has to cite a paper. Author + year + venue +
section/table/figure + a *reason this should help on Higgs*. Bare URLs
are rejected. "Folk best practice" is rejected. If you want to set
`lightgbm.num_leaves = 256`, you cite Ke 2017 §3.2 with a sentence about
why the default 31 underfits at 10 M rows. The reasoning module parses
the citation block and refuses the experiment if the format is wrong.

**Gate 3: Reasoning Blob Completeness.** Every experiment ships a
7-section reasoning blob: `diagnose` (what's wrong with prior runs),
`cite` (Gate 2), `hypothesize` (what the change will do, why),
`predict` (a quantitative prediction in AUROC delta and sign), `run`
(the recipe id and diff), `analyze` (post-run: what happened vs
predicted, with surprise budget), `checkpoint` (what the experiment
adds to the campaign). Each section has a word floor. The runner
refuses to start without a passing pre-run blob (sections 1-5) and
refuses to write the result row without a passing post-run blob
(sections 1-7).

## What an experiment actually looks like

```yaml
experiment: 1
backbone: lightgbm
recipe_id: lightgbm_default
predecessor: null
diff_vs_predecessor: registry default

reasoning:
  diagnose: |
    No prior experiments. We need a Tier-1 baseline so subsequent
    experiments have something to be measured against. LightGBM is the
    most-cited gradient-boosted-trees baseline in tabular ML
    competitions over the last decade and is the natural starting
    point for the GBM family.
  cite: |
    Ke et al. 2017 NeurIPS "LightGBM: A Highly Efficient Gradient
    Boosting Decision Tree" §3 ("LightGBM Algorithm") establishes the
    histogram-based learner with leaf-wise tree growth. Default
    `num_leaves=31, learning_rate=0.1, num_iterations=100` per §3.4.
    Reason for using these on Higgs: the paper's experimental section
    §4 demonstrates these defaults perform within 0.5% of tuned values
    on benchmarks of similar scale (1M-10M rows).
  hypothesize: |
    LightGBM with default params on Higgs should reach val/test AUROC
    in the [0.74, 0.78] range based on Chen 2016 XGBoost-Higgs
    reproduction giving 0.810 with deeper tuning. The default 100
    iterations is intentionally conservative — we expect modest
    underfitting — but this anchors the campaign and exposes the
    natural baseline gap.
  predict: |
    val AUROC ≈ 0.76 ± 0.01, test AUROC ≈ 0.76 ± 0.01, val/test gap
    < 0.005 (gradient boosting is notoriously stable on this kind of
    held-out split). Composite ≈ 0.755.
```

The runner reads the blob, runs `validate_reasoning_blob()`, computes a
hash of the recipe, and only then is the experiment allowed to call
`backbone.fit()`. After training, the post-run sections (`analyze` and
`checkpoint`) are written, the blob is re-validated, and only if every
section meets its word floor does the row land in
`autoresearch_results/all_runs.csv`.

## The Goodhart fingerprint

The composite metric is

```python
composite = min(test_auc, val_auc) - 0.1 * abs(test_auc - val_auc)
```

That formula is hashed (SHA-256 of the literal string) at runner boot
and the hash is stored in `composite.fingerprint`. Every result row
embeds the hash. If anyone — me, future-me, a contributor —
silently swaps the formula mid-campaign, the runner refuses to start
because the new hash doesn't match the existing one, and the entire
leaderboard is flagged as invalidated. This is one of the most
important pieces of audit hygiene I've added to the project. A
benchmark whose metric can be silently changed isn't a benchmark.

## Why Higgs UCI specifically?

The honest answer is: it's one of the few large tabular benchmarks with
genuine publishable headroom in 2026 *and* a frozen split that nobody
has cheated on. Adult Income, MNIST, CIFAR-10 are all saturated.
Forest CoverType has a weird non-i.i.d. structure. The 2024 TabPFN-v2
paper showed that small tabular benchmarks are mostly at ceiling.
Higgs is large enough (11M rows) to discriminate methods cleanly,
the headroom from XGBoost (0.864) to recent SOTA (TabM ~0.886) is
real, and the LHC physics community uses this exact problem class in
production trigger systems — so the result has industrial relevance.

## What the loop produced overnight

_Live snapshot — last updated 2026-04-26T21:29:49Z._

- **Total experiments completed:** 94
- **Backbones with results:** 5 (`catboost, lightgbm, logistic_regression, tabm, xgboost`)
- **Global champion:** experiment #92 on `tabm` — composite **0.8675** (val_auc 0.8676, test_auc 0.8679, val/test gap 0.0003)
- **Data-split fingerprint (Baldi 2014, subset_train_n=1M):** `eee12999eeae3950c0c27295…`
- **Composite formula fingerprint:** SHA-256 of `min(test_auc, val_auc) - 0.1 * abs(test_auc - val_auc)`

### Per-backbone leaderboard

| backbone | n_exps | best composite | best test_auc | best val_auc | within-backbone gap (max−min composite) |
|---|---:|---:|---:|---:|---:|
| `catboost` | 16 | **0.8355** | 0.8355 | 0.8357 | 0.0250 |
| `lightgbm` | 25 | **0.8401** | 0.8401 | 0.8403 | 0.0145 |
| `logistic_regression` | 25 | **0.6846** | 0.6846 | 0.6849 | 0.0001 |
| `tabm` | 3 | **0.8675** | 0.8679 | 0.8676 | 0.0320 |
| `xgboost` | 25 | **0.8403** | 0.8403 | 0.8403 | 0.0238 |

### Global top 10

| rank | exp | backbone | composite | val_auc | test_auc | val/test gap | description |
|---|---:|---|---:|---:|---:|---:|---|
| 1 | 92 | `tabm` | **0.8675** | 0.8676 | 0.8679 | 0.0003 | exp92 [tabm#1] paper Higgs default k=32 h=512x3 lr=2e-3 |
| 2 | 69 | `xgboost` | **0.8403** | 0.8403 | 0.8403 | 0.0001 | exp69 [xgboost#19] depth 10 lr 0.01 iters 5000 |
| 3 | 47 | `lightgbm` | **0.8401** | 0.8403 | 0.8401 | 0.0002 | exp47 [lightgbm#22] leaves 511 lr 0.01 iters 5000 |
| 4 | 54 | `xgboost` | **0.8374** | 0.8374 | 0.8374 | 0.0000 | exp54 [xgboost#4] depth 10 |
| 5 | 94 | `tabm` | **0.8374** | 0.8374 | 0.8374 | 0.0000 | exp94 [tabm#3] @ train=1M k=16 ensemble |
| 6 | 55 | `xgboost` | **0.8373** | 0.8376 | 0.8373 | 0.0003 | exp55 [xgboost#5] depth 12 |
| 7 | 30 | `lightgbm` | **0.8370** | 0.8373 | 0.8371 | 0.0002 | exp30 [lightgbm#5] leaves 511 |
| 8 | 68 | `xgboost` | **0.8368** | 0.8370 | 0.8369 | 0.0002 | exp68 [xgboost#18] depth 8 lr 0.02 iters 3000 |
| 9 | 73 | `xgboost` | **0.8368** | 0.8370 | 0.8369 | 0.0002 | exp73 [xgboost#23] seed 42 best-zone |
| 10 | 74 | `xgboost` | **0.8368** | 0.8370 | 0.8369 | 0.0002 | exp74 [xgboost#24] seed 7 best-zone |

## What I'd change

- The 25-experiment-per-backbone budget is fine for Tier 1 (GBMs) but
  too small for the deep architectures, where 25 paper-cited HP changes
  barely scratch the search space. Future runs should adapt the budget
  to the architecture's natural HP dimensionality.
- Citation Rigor is great at filtering out folklore but is a *human*
  bottleneck — even with paper-grounded HP changes pre-baked into the
  catalog, writing 350 distinct reasoning blobs is non-trivial. The
  next iteration will reuse blobs across mechanically-identical changes
  (e.g., learning-rate sweeps with `lr ∈ {1e-3, 5e-4, 2e-4}`) instead
  of duplicating prose.
- The runner's per-experiment audit gate is currently a hard refusal;
  it should arguably be a "yellow" warning state for cases where the
  blob is slightly under word-floor but otherwise clearly grounded.

If you want to see the actual audit reports, the per-experiment
reasoning blobs, or the per-row prediction CSVs, everything is at
https://github.com/dlmastery/autoresearchtabular and the live dashboard
is at https://dlmastery.github.io/autoresearchtabular/.
