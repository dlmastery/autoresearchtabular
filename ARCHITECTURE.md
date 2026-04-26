# ARCHITECTURE — autoresearchtabular

## High-level

```
┌──────────────────────────────────────────────────────────────────┐
│                    User / autonomous loop                        │
│         (writes reasoning_blob.json, picks recipe id)            │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                       core/runner.py                              │
│  - hardware contract: P-core pinning, deterministic seeds        │
│  - audit_or_die() gate (Gate 1)                                  │
│  - composite formula fingerprint check                           │
│  - validate_reasoning_blob() pre-run (Gates 2 + 3)               │
│  - dispatch to backbone via core/backbones/registry.py           │
│  - post-run: metrics, composite, champion, dashboard refresh     │
│  - validate_reasoning_blob() post-run (Gate 3 with all 7)        │
│  - append-only row to all_runs.csv                               │
└──────────────────────────────────────────────────────────────────┘
        │              │                │              │
        ▼              ▼                ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ core/data/  │ │core/eval/   │ │core/back-   │ │core/        │
│ loader.py   │ │ audit.py    │ │bones/*      │ │reasoning.py │
│             │ │ composite.py│ │             │ │ checkpoint  │
│ Higgs CSV → │ │ metrics.py  │ │ LR, RF,     │ │             │
│ NPZ + Baldi │ │             │ │ LightGBM,   │ │ Citation    │
│ split       │ │ 7 auditors  │ │ XGBoost,    │ │ Rigor +     │
│             │ │ AUROC/AUPRC │ │ CatBoost,   │ │ Reasoning   │
│             │ │ ECE / sig-  │ │ MLP,        │ │ Blob gates  │
│             │ │ rejection   │ │ MLP-PLR,    │ │             │
│             │ │             │ │ FT-Transf.  │ │             │
│             │ │             │ │ SAINT,      │ │             │
│             │ │             │ │ NODE,       │ │             │
│             │ │             │ │ TabNet,     │ │             │
│             │ │             │ │ TabM,       │ │             │
│             │ │             │ │ TabPFN-v2,  │ │             │
│             │ │             │ │ Trompt,     │ │             │
│             │ │             │ │ ExcelFormer │ │             │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
        │              │                │              │
        └──────────────┴────────┬───────┴──────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                   autoresearch_results/                           │
│  ┌──────────────┐  ┌──────────────────────────┐                  │
│  │ all_runs.csv │  │ exp_{N:03d}_{backbone}/  │                  │
│  │ leaderboard  │  │   recipe.yaml            │                  │
│  │ JSON summary │  │   metrics.json           │                  │
│  │ data_split_  │  │   predictions.csv        │                  │
│  │ audit.md     │  │   reasoning_blob.json    │                  │
│  │ composite.   │  │   model_checkpoint.{pt,  │                  │
│  │ fingerprint  │  │   joblib} (gitignored)   │                  │
│  │ winners/     │  │                          │                  │
│  └──────────────┘  └──────────────────────────┘                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                       dashboard/dashboard.html                    │
│   live filter / sort / search / export, KPI cards,               │
│   per-backbone tabs, reasoning detail panel                      │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                            docs/                                  │
│       (mirror of dashboard/, plus index.html for                  │
│        https://dlmastery.github.io/autoresearchtabular/)         │
└──────────────────────────────────────────────────────────────────┘
```

## Data flow per experiment

```
Higgs CSV.gz
   ↓ (one-time download via scripts/download_higgs.py)
.data_cache/higgs/HIGGS.csv.gz
   ↓ (one-time materialisation in loader.py)
.data_cache/higgs/higgs.npz       # X (11M, 28) f32, y (11M,) i8
   ↓
load_higgs_split(...)
   ↓ Baldi 2014 frozen indices
{train, val, test}                 # each: (X, y, row_idx)
   ↓
audit_or_die(splits)               # Gate 1
   ↓ (writes data_split_audit.{json,md} and fingerprint.txt)
backbone.fit(X_train, y_train)
backbone.predict_proba(X_val/X_test)
   ↓
metrics: AUROC, AUPRC, ECE, bg_rej@{0.5, 0.7, 0.9}
   ↓
composite = min(test_auc, val_auc) - 0.1 * |test_auc - val_auc|
   ↓ (post-run reasoning blob written)
validate_reasoning_blob(post_run=True)
   ↓
append row to all_runs.csv
write per-experiment artifacts
update champion / winners/
refresh dashboard data
```

## Key invariants

1. **`data_split_fingerprint.txt`** is the global anchor. It is set by
   the first audit run and never changed. Every experiment row records
   this fingerprint. If it doesn't match, the run aborts.

2. **`composite.fingerprint`** is the metric anchor. SHA-256 of the
   composite formula string. Set on first runner boot; verified every
   subsequent run.

3. **`all_runs.csv` is append-only.** No row is ever deleted or
   modified. If an experiment needs to be redone, it gets a new row;
   the original failed/superseded row stays in the log.

4. **Per-experiment artifacts are immutable post-write.** Once
   `recipe.yaml` and `reasoning_blob.json` for experiment N are written,
   they are not overwritten — they are the audit trail.

5. **The runner is one-experiment-per-call.** Concurrency is
   intentionally not supported. The campaign script wraps the runner in
   a sequential loop.

## Module boundaries

- `core/data/` knows about Higgs and its split. It does *not* know
  about backbones, metrics, or reasoning.
- `core/evaluation/` knows about metrics and audits. It does *not* know
  about backbones or training.
- `core/backbones/` knows about training. It does *not* know about
  audit, composite, or reasoning.
- `core/reasoning.py` knows about the gate semantics. It does *not*
  know about training.
- `core/runner.py` is the *only* file that crosses module boundaries.

This boundary discipline is what makes the audit gate work — each
module can be replayed by a third-party verifier in isolation.

## Hardware contract surface

```python
# core/runner.py, _pin_to_safe_cores()
P_CORES = list(range(0, 16))   # 0-15
E_CORES_BANNED = {16, 17, 24, 25}
psutil.Process().cpu_affinity(P_CORES)
```

```python
# core/runner.py, _seed_all()
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(seed)
```
