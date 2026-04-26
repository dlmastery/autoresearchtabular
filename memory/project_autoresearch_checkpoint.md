# AUTORESEARCHTABULAR — autoresearch checkpoint

_Last updated: 2026-04-26T22:31:31.576360Z_

## Session start instructions

1. Read this file (you are here).
2. Read `CLAUDE.md`.
3. Read `autoresearch_results/experiment_log.jsonl` (last 3 entries).
4. Run the audit if it's stale (> 24 h).
5. Resume the loop with the command below.

```
"C:/Users/evija/anaconda3/python.exe" -m core.runner --config configs/higgs.yaml --backbone resnet_tabular --description "exp98: <DESCRIBE>"
```

## Current champion
- Backbone: `ft_transformer`
- Experiment: #95
- Composite: **0.8723**
- test_auc: 0.8726
- val_auc: 0.8723
- Description: PAPER_DEFAULT_FULL_10M ft_transformer recipe#1: paper Higgs default 3blk d=192 h=8

## Last experiment
- #97 backbone=`resnet_tabular` composite=0.8678 status=KEEP
- Description: PAPER_DEFAULT_FULL_10M resnet_tabular recipe#1: paper Higgs default 2blk d=256 mult=2

## Experiment history

| # | backbone | composite | test_auc | val_auc | train_auc | status |
|---|---|---|---|---|---|---|
| 68 | `xgboost` | 0.8368 | 0.8369 | 0.8370 | 0.8971 | KEEP |
| 69 | `xgboost` | 0.8403 | 0.8403 | 0.8403 | 0.9531 | KEEP/CHAMPION |
| 70 | `xgboost` | 0.8302 | 0.8302 | 0.8304 | 0.8491 | KEEP |
| 71 | `xgboost` | 0.8354 | 0.8354 | 0.8356 | 0.8827 | KEEP |
| 72 | `xgboost` | 0.8286 | 0.8286 | 0.8288 | 0.8449 | KEEP |
| 73 | `xgboost` | 0.8368 | 0.8369 | 0.8370 | 0.8971 | KEEP |
| 74 | `xgboost` | 0.8368 | 0.8369 | 0.8370 | 0.8971 | KEEP |
| 75 | `xgboost` | 0.8368 | 0.8369 | 0.8370 | 0.8971 | KEEP |
| 76 | `catboost` | 0.8223 | 0.8223 | 0.8227 | 0.8249 | KEEP |
| 77 | `catboost` | 0.8105 | 0.8105 | 0.8111 | 0.8114 | KEEP |
| 78 | `catboost` | 0.8303 | 0.8303 | 0.8306 | 0.8386 | KEEP |
| 79 | `catboost` | 0.8355 | 0.8355 | 0.8357 | 0.8610 | KEEP |
| 80 | `catboost` | 0.8223 | 0.8224 | 0.8227 | 0.8250 | KEEP |
| 81 | `catboost` | 0.8292 | 0.8292 | 0.8294 | 0.8353 | KEEP |
| 82 | `catboost` | 0.8244 | 0.8244 | 0.8248 | 0.8277 | KEEP |
| 83 | `catboost` | 0.8222 | 0.8222 | 0.8227 | 0.8248 | KEEP |
| 84 | `catboost` | 0.8223 | 0.8223 | 0.8226 | 0.8248 | KEEP |
| 85 | `catboost` | 0.8221 | 0.8222 | 0.8226 | 0.8247 | KEEP |
| 86 | `catboost` | 0.8223 | 0.8223 | 0.8227 | 0.8249 | KEEP |
| 87 | `catboost` | 0.8223 | 0.8223 | 0.8227 | 0.8249 | KEEP |
| 88 | `catboost` | 0.8223 | 0.8223 | 0.8227 | 0.8249 | KEEP |
| 89 | `catboost` | 0.8223 | 0.8223 | 0.8227 | 0.8249 | KEEP |
| 90 | `catboost` | 0.8223 | 0.8223 | 0.8227 | 0.8249 | KEEP |
| 91 | `catboost` | 0.8223 | 0.8223 | 0.8227 | 0.8249 | KEEP |
| 92 | `tabm` | 0.8675 | 0.8679 | 0.8676 | 0.8705 | KEEP/CHAMPION |
| 93 | `tabm` | 0.8355 | 0.8355 | 0.8356 | 0.8370 | KEEP |
| 94 | `tabm` | 0.8374 | 0.8374 | 0.8374 | 0.8405 | KEEP |
| 95 | `ft_transformer` | 0.8723 | 0.8726 | 0.8723 | 0.8769 | KEEP/CHAMPION |
| 96 | `mlp_plr` | 0.8623 | 0.8623 | 0.8623 | 0.8674 | KEEP |
| 97 | `resnet_tabular` | 0.8678 | 0.8685 | 0.8679 | 0.8715 | KEEP |

## Next experiment command
```
"C:/Users/evija/anaconda3/python.exe" -m core.runner --config configs/higgs.yaml --backbone resnet_tabular --description "exp98: <DESCRIBE>"
```