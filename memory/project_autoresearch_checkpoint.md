# AUTORESEARCHTABULAR — autoresearch checkpoint

_Last updated: 2026-04-26T06:46:45.118571Z_

## Session start instructions

1. Read this file (you are here).
2. Read `CLAUDE.md`.
3. Read `autoresearch_results/experiment_log.jsonl` (last 3 entries).
4. Run the audit if it's stale (> 24 h).
5. Resume the loop with the command below.

```
"C:/Users/evija/anaconda3/python.exe" -m core.runner --config configs/higgs.yaml --backbone lightgbm --description "exp2: <DESCRIBE>"
```

## Current champion
- Backbone: `lightgbm`
- Experiment: #1
- Composite: **0.8301**
- test_auc: 0.8302
- val_auc: 0.8305
- Description: exp1 [lightgbm#1] default 1000 leaves 63 lr 0.05

## Last experiment
- #1 backbone=`lightgbm` composite=0.8301 status=KEEP/CHAMPION
- Description: exp1 [lightgbm#1] default 1000 leaves 63 lr 0.05

## Experiment history

| # | backbone | composite | test_auc | val_auc | train_auc | status |
|---|---|---|---|---|---|---|
| 1 | `lightgbm` | 0.8301 | 0.8302 | 0.8305 | 0.8504 | KEEP/CHAMPION |

## Next experiment command
```
"C:/Users/evija/anaconda3/python.exe" -m core.runner --config configs/higgs.yaml --backbone lightgbm --description "exp2: <DESCRIBE>"
```