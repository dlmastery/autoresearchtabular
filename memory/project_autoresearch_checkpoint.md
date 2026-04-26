# AUTORESEARCHTABULAR — autoresearch checkpoint

_Last updated: 2026-04-26T11:36:32.092838Z_

## Session start instructions

1. Read this file (you are here).
2. Read `CLAUDE.md`.
3. Read `autoresearch_results/experiment_log.jsonl` (last 3 entries).
4. Run the audit if it's stale (> 24 h).
5. Resume the loop with the command below.

```
"C:/Users/evija/anaconda3/python.exe" -m core.runner --config configs/higgs.yaml --backbone xgboost --description "exp69: <DESCRIBE>"
```

## Current champion
- Backbone: `lightgbm`
- Experiment: #47
- Composite: **0.8401**
- test_auc: 0.8401
- val_auc: 0.8403
- Description: exp47 [lightgbm#22] leaves 511 lr 0.01 iters 5000

## Last experiment
- #68 backbone=`xgboost` composite=0.8368 status=KEEP
- Description: exp68 [xgboost#18] depth 8 lr 0.02 iters 3000

## Experiment history

| # | backbone | composite | test_auc | val_auc | train_auc | status |
|---|---|---|---|---|---|---|
| 39 | `lightgbm` | 0.8296 | 0.8296 | 0.8298 | 0.8495 | KEEP |
| 40 | `lightgbm` | 0.8299 | 0.8300 | 0.8303 | 0.8502 | KEEP |
| 41 | `lightgbm` | 0.8306 | 0.8306 | 0.8309 | 0.8497 | KEEP |
| 42 | `lightgbm` | 0.8298 | 0.8298 | 0.8301 | 0.8488 | KEEP |
| 43 | `lightgbm` | 0.8299 | 0.8299 | 0.8304 | 0.8501 | KEEP |
| 44 | `lightgbm` | 0.8352 | 0.8352 | 0.8355 | 0.8817 | KEEP |
| 45 | `lightgbm` | 0.8358 | 0.8358 | 0.8360 | 0.9070 | KEEP |
| 46 | `lightgbm` | 0.8341 | 0.8341 | 0.8343 | 0.8702 | KEEP |
| 47 | `lightgbm` | 0.8401 | 0.8401 | 0.8403 | 0.9557 | KEEP/CHAMPION |
| 48 | `lightgbm` | 0.8352 | 0.8352 | 0.8355 | 0.8817 | KEEP |
| 49 | `lightgbm` | 0.8352 | 0.8352 | 0.8355 | 0.8817 | KEEP |
| 50 | `lightgbm` | 0.8352 | 0.8352 | 0.8355 | 0.8817 | KEEP |
| 51 | `xgboost` | 0.8286 | 0.8286 | 0.8288 | 0.8449 | KEEP |
| 52 | `xgboost` | 0.8164 | 0.8165 | 0.8169 | 0.8201 | KEEP |
| 53 | `xgboost` | 0.8350 | 0.8350 | 0.8353 | 0.8868 | KEEP |
| 54 | `xgboost` | 0.8374 | 0.8374 | 0.8374 | 0.9507 | KEEP |
| 55 | `xgboost` | 0.8373 | 0.8373 | 0.8376 | 0.9928 | KEEP |
| 56 | `xgboost` | 0.8293 | 0.8294 | 0.8296 | 0.8456 | KEEP |
| 57 | `xgboost` | 0.8320 | 0.8320 | 0.8321 | 0.8630 | KEEP |
| 58 | `xgboost` | 0.8306 | 0.8306 | 0.8308 | 0.8499 | KEEP |
| 59 | `xgboost` | 0.8283 | 0.8283 | 0.8285 | 0.8444 | KEEP |
| 60 | `xgboost` | 0.8280 | 0.8280 | 0.8283 | 0.8442 | KEEP |
| 61 | `xgboost` | 0.8266 | 0.8266 | 0.8270 | 0.8419 | KEEP |
| 62 | `xgboost` | 0.8292 | 0.8292 | 0.8295 | 0.8461 | KEEP |
| 63 | `xgboost` | 0.8286 | 0.8286 | 0.8288 | 0.8449 | KEEP |
| 64 | `xgboost` | 0.8286 | 0.8286 | 0.8288 | 0.8449 | KEEP |
| 65 | `xgboost` | 0.8286 | 0.8286 | 0.8288 | 0.8449 | KEEP |
| 66 | `xgboost` | 0.8287 | 0.8287 | 0.8289 | 0.8439 | KEEP |
| 67 | `xgboost` | 0.8287 | 0.8287 | 0.8290 | 0.8449 | KEEP |
| 68 | `xgboost` | 0.8368 | 0.8369 | 0.8370 | 0.8971 | KEEP |

## Next experiment command
```
"C:/Users/evija/anaconda3/python.exe" -m core.runner --config configs/higgs.yaml --backbone xgboost --description "exp69: <DESCRIBE>"
```