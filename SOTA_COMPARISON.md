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

*(populated when campaign completes)*

The campaign budget is 25 experiments per backbone. With paper-grounded
HP changes, the realistic targets are:

| Tier | Backbones | Composite target | Test AUROC target |
|------|-----------|-----------------:|------------------:|
| 1    | LR, RF    | 0.65 - 0.80      | 0.65 - 0.80       |
| 1    | LightGBM, XGBoost, CatBoost | 0.85 - 0.87      | 0.85 - 0.87       |
| 2    | MLP, MLP-PLR, FT-Transformer, SAINT, NODE, TabNet | 0.86 - 0.88 | 0.86 - 0.88 |
| 3    | TabM, TabPFN-v2, Trompt, ExcelFormer | 0.87 - 0.89 | 0.87 - 0.89 |

Goal: a per-backbone champion within 0.005 AUROC of the published
result for that architecture. We are *not* trying to set new SOTA on
Higgs; we are trying to demonstrate that the audit-gated discipline
reproduces published numbers within a defensible margin.

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
