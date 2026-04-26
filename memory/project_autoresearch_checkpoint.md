# AUTORESEARCHTABULAR — autoresearch checkpoint

_Last updated: 2026-04-26T08:03:50.519408Z_

## Session start instructions

1. Read this file (you are here).
2. Read `CLAUDE.md`.
3. Read `autoresearch_results/experiment_log.jsonl` (last 3 entries).
4. Run the audit if it's stale (> 24 h).
5. Resume the loop with the command below.

```
"C:/Users/evija/anaconda3/python.exe" -m core.runner --config configs/higgs.yaml --backbone lightgbm --description "exp41: <DESCRIBE>"
```

## Current champion
- Backbone: `lightgbm`
- Experiment: #30
- Composite: **0.8370**
- test_auc: 0.8371
- val_auc: 0.8373
- Description: exp30 [lightgbm#5] leaves 511

## Last experiment
- #40 backbone=`lightgbm` composite=0.8299 status=KEEP
- Description: exp40 [lightgbm#15] reg_alpha 0.1

## Experiment history

| # | backbone | composite | test_auc | val_auc | train_auc | status |
|---|---|---|---|---|---|---|
| 11 | `logistic_regression` | 0.6846 | 0.6846 | 0.6849 | 0.6838 | KEEP |
| 12 | `logistic_regression` | 0.6846 | 0.6846 | 0.6849 | 0.6838 | KEEP |
| 13 | `logistic_regression` | 0.6846 | 0.6846 | 0.6849 | 0.6838 | KEEP |
| 14 | `logistic_regression` | 0.6846 | 0.6846 | 0.6849 | 0.6838 | KEEP |
| 15 | `logistic_regression` | 0.6846 | 0.6846 | 0.6849 | 0.6838 | KEEP |
| 16 | `logistic_regression` | 0.6846 | 0.6846 | 0.6849 | 0.6838 | KEEP |
| 17 | `logistic_regression` | 0.6846 | 0.6846 | 0.6849 | 0.6838 | KEEP |
| 18 | `logistic_regression` | 0.6846 | 0.6846 | 0.6849 | 0.6838 | KEEP |
| 19 | `logistic_regression` | 0.6846 | 0.6846 | 0.6849 | 0.6838 | KEEP |
| 20 | `logistic_regression` | 0.6846 | 0.6846 | 0.6849 | 0.6838 | KEEP |
| 21 | `logistic_regression` | 0.6846 | 0.6846 | 0.6849 | 0.6838 | KEEP |
| 22 | `logistic_regression` | 0.6846 | 0.6846 | 0.6849 | 0.6838 | KEEP |
| 23 | `logistic_regression` | 0.6846 | 0.6846 | 0.6849 | 0.6838 | KEEP |
| 24 | `logistic_regression` | 0.6846 | 0.6846 | 0.6849 | 0.6838 | KEEP |
| 25 | `logistic_regression` | 0.6846 | 0.6846 | 0.6849 | 0.6838 | KEEP |
| 26 | `logistic_regression` | 0.6846 | 0.6846 | 0.6849 | 0.6838 | KEEP |
| 27 | `lightgbm` | 0.8256 | 0.8256 | 0.8259 | 0.8351 | KEEP |
| 28 | `lightgbm` | 0.8336 | 0.8337 | 0.8341 | 0.8731 | KEEP/CHAMPION |
| 29 | `lightgbm` | 0.8358 | 0.8358 | 0.8360 | 0.9070 | KEEP/CHAMPION |
| 30 | `lightgbm` | 0.8370 | 0.8371 | 0.8373 | 0.9529 | KEEP/CHAMPION |
| 31 | `lightgbm` | 0.8359 | 0.8359 | 0.8361 | 0.9024 | KEEP |
| 32 | `lightgbm` | 0.8301 | 0.8302 | 0.8305 | 0.8504 | KEEP |
| 33 | `lightgbm` | 0.8353 | 0.8353 | 0.8355 | 0.8902 | KEEP |
| 34 | `lightgbm` | 0.8301 | 0.8302 | 0.8305 | 0.8504 | KEEP |
| 35 | `lightgbm` | 0.8301 | 0.8302 | 0.8305 | 0.8504 | KEEP |
| 36 | `lightgbm` | 0.8292 | 0.8292 | 0.8296 | 0.8486 | KEEP |
| 37 | `lightgbm` | 0.8306 | 0.8306 | 0.8309 | 0.8513 | KEEP |
| 38 | `lightgbm` | 0.8301 | 0.8302 | 0.8304 | 0.8504 | KEEP |
| 39 | `lightgbm` | 0.8296 | 0.8296 | 0.8298 | 0.8495 | KEEP |
| 40 | `lightgbm` | 0.8299 | 0.8300 | 0.8303 | 0.8502 | KEEP |

## Next experiment command
```
"C:/Users/evija/anaconda3/python.exe" -m core.runner --config configs/higgs.yaml --backbone lightgbm --description "exp41: <DESCRIBE>"
```