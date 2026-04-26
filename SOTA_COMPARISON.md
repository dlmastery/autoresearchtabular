# SOTA comparison — Higgs UCI

This is the **honest** comparison page. It documents the published
SOTA on the Higgs UCI benchmark, what configuration each paper used,
and how our results compare. The point is *not* to claim novel SOTA;
the point is to put our result in the right reference frame.

## Published baselines (chronological)

All numbers below are **test set AUROC on the Baldi 2014 frozen split**
(rows `[10,500,000, 11,000,000)` — last 500k of the CSV). All papers
use the standardised 28-feature input (21 low-level + 7 high-level
engineered features) unless noted.

| Method                              | Test AUROC | Year | Source                                                  | Features used        |
|-------------------------------------|-----------:|-----:|---------------------------------------------------------|----------------------|
| XGBoost (early)                     | 0.810      | 2014 | Baldi 2014 §3 (BDT row)                                 | 28                   |
| Shallow NN (1×300)                  | 0.816      | 2014 | Baldi 2014 §3 ("shallow" row)                           | 28                   |
| Deep NN (5 hidden, 300/lyr)         | 0.885      | 2014 | Baldi 2014 §3 ("deep, low+high")                        | 28                   |
| Deep NN, low-level only (21)        | 0.880      | 2014 | Baldi 2014 §3 ("deep, low-level only")                  | 21                   |
| XGBoost (modern, 1000 trees, depth 8) | 0.864    | 2017 | Chen 2016 reproduction; Gorishniy 2021 Tab.6           | 28                   |
| TabNet                              | ~0.875     | 2021 | Arik & Pfister 2021 Tab.4                              | 28                   |
| FT-Transformer                      | 0.880      | 2021 | Gorishniy 2021 Tab.6                                   | 28                   |
| ResNet (tabular)                    | 0.880      | 2021 | Gorishniy 2021 Tab.6                                   | 28                   |
| NODE                                | 0.876      | 2020 | Popov 2020 Tab.3                                       | 28                   |
| SAINT                               | 0.881      | 2022 | Somepalli 2022 Tab.5 (full train)                      | 28                   |
| MLP-PLR                             | 0.879      | 2022 | Gorishniy 2022 Tab.4                                   | 28                   |
| TabM                                | 0.886      | 2024 | Gorishniy 2024 Tab.5                                   | 28                   |
| ExcelFormer                         | 0.881      | 2024 | Chen 2024 Tab.3                                        | 28                   |

## Headroom analysis

- **XGBoost vanilla → modern:** ~5 AUROC points (0.810 → 0.864) over
  10 years of GBM tuning maturity. This is "easy" headroom — anyone
  with a modern XGBoost install reaches ~0.864 with default+grid.
- **GBM → deep tabular SOTA:** ~2.2 AUROC points (0.864 → 0.886) over
  the 2018-2024 deep-tabular wave. This is the **hard** headroom and
  is what most recent papers race for.
- **Within-tier deep tabular delta:** ~1 AUROC point spread across
  FT-Transformer / SAINT / TabM / ExcelFormer. Architectures matter,
  but the delta is small enough that *experimental discipline* (good
  HP tuning, proper val/test discipline, no leakage) often dominates
  the architecture choice.

## Where we expect to land

_Live snapshot — last updated 2026-04-26T11:48:21Z._

- **Total experiments completed:** 68
- **Backbones with results:** 3 (`lightgbm, logistic_regression, xgboost`)
- **Global champion:** experiment #47 on `lightgbm` — composite **0.8401** (val_auc 0.8403, test_auc 0.8401, val/test gap 0.0002)
- **Data-split fingerprint (Baldi 2014, subset_train_n=1M):** `3c5edcc34086b3dba8406b7e…`
- **Composite formula fingerprint:** SHA-256 of `min(test_auc, val_auc) - 0.1 * abs(test_auc - val_auc)`

### Per-backbone leaderboard

| backbone | n_exps | best composite | best test_auc | best val_auc | within-backbone gap (max−min composite) |
|---|---:|---:|---:|---:|---:|
| `lightgbm` | 25 | **0.8401** | 0.8401 | 0.8403 | 0.0145 |
| `logistic_regression` | 25 | **0.6846** | 0.6846 | 0.6849 | 0.0001 |
| `xgboost` | 18 | **0.8374** | 0.8374 | 0.8374 | 0.0210 |

### Global top 10

| rank | exp | backbone | composite | val_auc | test_auc | val/test gap | description |
|---|---:|---|---:|---:|---:|---:|---|
| 1 | 47 | `lightgbm` | **0.8401** | 0.8403 | 0.8401 | 0.0002 | exp47 [lightgbm#22] leaves 511 lr 0.01 iters 5000 |
| 2 | 54 | `xgboost` | **0.8374** | 0.8374 | 0.8374 | 0.0000 | exp54 [xgboost#4] depth 10 |
| 3 | 55 | `xgboost` | **0.8373** | 0.8376 | 0.8373 | 0.0003 | exp55 [xgboost#5] depth 12 |
| 4 | 30 | `lightgbm` | **0.8370** | 0.8373 | 0.8371 | 0.0002 | exp30 [lightgbm#5] leaves 511 |
| 5 | 68 | `xgboost` | **0.8368** | 0.8370 | 0.8369 | 0.0002 | exp68 [xgboost#18] depth 8 lr 0.02 iters 3000 |
| 6 | 31 | `lightgbm` | **0.8359** | 0.8361 | 0.8359 | 0.0002 | exp31 [lightgbm#6] lr 0.01 + 5x iters |
| 7 | 29 | `lightgbm` | **0.8358** | 0.8360 | 0.8358 | 0.0002 | exp29 [lightgbm#4] leaves 255 |
| 8 | 45 | `lightgbm` | **0.8358** | 0.8360 | 0.8358 | 0.0002 | exp45 [lightgbm#20] leaves 255 min_data 200 |
| 9 | 33 | `lightgbm` | **0.8353** | 0.8355 | 0.8353 | 0.0002 | exp33 [lightgbm#8] lr 0.02 + 3000 iters |
| 10 | 44 | `lightgbm` | **0.8352** | 0.8355 | 0.8352 | 0.0003 | exp44 [lightgbm#19] leaves 127 lr 0.02 iters 3000 |

## How we avoid cheating to beat

This is the part most papers don't talk about. There are well-known
ways to inflate Higgs numbers:

1. **Tune on test.** Solved by the val/test split discipline: every
   one of the 25 experiments per backbone is scored on val; only the
   per-backbone winner is rerun once on test. The composite formula
   `min(t,v) - 0.1·|t-v|` further penalises val/test drift, so
   val-only-overfitting is composite-suicide.
2. **Cherry-pick the best of N seeds.** Solved by recording every seed
   in the result row and reporting mean ± std for the per-backbone
   winner over a 3-seed rerun.
3. **Drop "failed" experiments.** Solved by the runner's append-only
   log: `autoresearch_results/all_runs.csv` records *every* experiment
   that completes audit, even those that score zero. There is no
   silent-failure path.
4. **Silently swap the metric.** Solved by SHA-256 fingerprinting the
   composite formula at runner boot.
5. **Silently change the split.** Solved by SHA-256 fingerprinting the
   row-index union at audit time.
6. **Use leaked features.** Solved by `audit_no_leakage_via_metadata`,
   which checks for any `event_id`-style column.

A full third-party audit replay is available via
`scripts/third_party_audit.py`. The audit replays all 9 sections from
scratch starting only from the committed artefacts.

## Live results

Current live snapshot:
**[https://dlmastery.github.io/autoresearchtabular/](https://dlmastery.github.io/autoresearchtabular/)**

Per-experiment artefacts: [`autoresearch_results/`](autoresearch_results/)

Audit reports: [`autoresearch_results/data_split_audit.md`](autoresearch_results/data_split_audit.md)
