# AUTORESEARCHTABULAR — autoresearch checkpoint

_Last updated: 2026-04-26T14:40:05.248355Z_

## Session start instructions

1. Read this file (you are here).
2. Read `CLAUDE.md`.
3. Read `autoresearch_results/experiment_log.jsonl` (last 3 entries).
4. Run the audit if it's stale (> 24 h).
5. Resume the loop with the command below.

```
"C:/Users/evija/anaconda3/python.exe" -m core.runner --config configs/higgs.yaml --backbone catboost --description "exp92: <DESCRIBE>"
```

## Current champion
- Backbone: `xgboost`
- Experiment: #69
- Composite: **0.8403**
- test_auc: 0.8403
- val_auc: 0.8403
- Description: exp69 [xgboost#19] depth 10 lr 0.01 iters 5000

## Last experiment
- #91 backbone=`catboost` composite=0.8223 status=KEEP
- Description: exp91 [catboost#16] border_count 254

## Experiment history

| # | backbone | composite | test_auc | val_auc | train_auc | status |
|---|---|---|---|---|---|---|
| 62 | `xgboost` | 0.8292 | 0.8292 | 0.8295 | 0.8461 | KEEP |
| 63 | `xgboost` | 0.8286 | 0.8286 | 0.8288 | 0.8449 | KEEP |
| 64 | `xgboost` | 0.8286 | 0.8286 | 0.8288 | 0.8449 | KEEP |
| 65 | `xgboost` | 0.8286 | 0.8286 | 0.8288 | 0.8449 | KEEP |
| 66 | `xgboost` | 0.8287 | 0.8287 | 0.8289 | 0.8439 | KEEP |
| 67 | `xgboost` | 0.8287 | 0.8287 | 0.8290 | 0.8449 | KEEP |
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

## Next experiment command
```
"C:/Users/evija/anaconda3/python.exe" -m core.runner --config configs/higgs.yaml --backbone catboost --description "exp92: <DESCRIBE>"
```