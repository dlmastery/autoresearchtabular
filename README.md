# AUTORESEARCHTABULAR — Higgs UCI tabular benchmark

[![GitHub Pages](https://img.shields.io/badge/dashboard-live-brightgreen)](https://dlmastery.github.io/autoresearchtabular/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

Autonomous ML research loop on the **Higgs boson UCI tabular benchmark**
(Baldi, Sadowski & Whiteson 2014, *Nature Communications*,
arXiv:1402.4735). Runs **25 experiments per backbone × 14 backbones**
under a hardened audit protocol — every experiment ships with a frozen
data-split fingerprint, paper-cited hyperparameter changes (Citation
Rigor format), and a reasoning blob that passes a programmatic gate
before the model is even allowed to start training.

This is the third project in the **autoresearch** series:

1. **[autoresearch](https://github.com/dlmastery/autoresearch)** — FX
   forecasting (15-min EUR/USD) — composite 0.6181, 79 experiments
2. **[autoresearchimage](https://github.com/dlmastery/autoresearchimage)** —
   Computer vision (PathMNIST + WILDS-Camelyon17) — 21 experiments,
   composite 0.9966, [GitHub Pages](https://dlmastery.github.io/autoresearchimage/)
3. **autoresearchtabular** *(this repo)* — Tabular ML (Higgs UCI) — 350
   experiments planned, [live dashboard](https://dlmastery.github.io/autoresearchtabular/)

The same 7-step research protocol (`diagnose → cite → hypothesize →
predict → run-ONE → analyze → checkpoint`) and three programmatic
gates (data-split audit, Citation Rigor, Reasoning Blob Completeness)
that were validated on FX and CV are reused here, retargeted at
tabular SOTA.

---

## Why Higgs?

The dataset is the canonical large-scale tabular ML benchmark in
high-energy physics. It is **deliberately not saturated**:

| Method                                         | Test AUROC | Year | Citation                    |
|------------------------------------------------|-----------:|-----:|-----------------------------|
| Boosted Decision Trees (XGBoost-style)         | 0.810      | 2014 | Baldi 2014 §3               |
| Shallow neural network (1 hidden layer)        | 0.816      | 2014 | Baldi 2014 §3               |
| Deep neural network (5 hidden layers, 300/lyr) | 0.885      | 2014 | Baldi 2014 §3 (high-level)  |
| XGBoost (modern, 1000 trees)                   | 0.864      | 2017 | Chen 2016 reproduction      |
| FT-Transformer                                 | 0.880      | 2021 | Gorishniy 2021 §5           |
| SAINT                                          | 0.881      | 2022 | Somepalli 2022 Tab.5        |
| TabM                                           | ~0.886     | 2024 | Gorishniy 2024              |

The headroom from XGBoost (~0.864) to deep tabular SOTA (~0.886) is
~2.2 AUROC points and is **still climbing year-over-year**. There is
real signal to chase, the dataset is large enough to discriminate
methods cleanly, and the Baldi 2014 split has been frozen for over a
decade so cross-paper comparison is unambiguous.

**Industry relevance.** ATLAS and CMS at the LHC use this exact
problem class (signal-vs-background classification of jet kinematic
features) in their software triggers. Improvements at the percent
level translate directly into more signal events surviving the trigger
budget. A 1% AUC gain at L0 trigger is a *paper-grade* result in HEP.

---

## What this repo contains

```
configs/higgs.yaml              # task config: paths, split, composite formula
sota_catalog.yaml               # 14 backbones with full Citation Rigor citations
core/
  data/loader.py                # Higgs CSV.gz → NPZ + frozen Baldi split
  evaluation/
    audit.py                    # 7 auditors, audit_or_die() gate
    composite.py                # SHA-256 fingerprinted composite formula
    metrics.py                  # AUROC, AUPRC, ECE, bg-rej-at-sig-eff
  reasoning.py                  # Citation Rigor + Reasoning Blob gates
  checkpoint.py                 # per-experiment append-only summaries
  runner.py                     # one-experiment-per-call orchestrator
  backbones/
    registry.py                 # @register_backbone factory
    sklearn_baselines.py        # LR, RF
    gbm.py                      # LightGBM, XGBoost, CatBoost
    mlp.py                      # MLP with BF16 + cosine LR
    ...                         # FT-Transformer, SAINT, NODE, TabNet, TabM, ...
scripts/
  download_higgs.py             # one-shot UCI download with progress
  third_party_audit.py          # 9-section audit replayer
  sync_dashboard_to_docs.py     # mirror dashboard → docs/ for Pages
  run_campaign.py               # per-backbone 25-experiment loop
dashboard/dashboard.html        # live filter+sort+search+export dashboard
docs/                           # GitHub Pages site (mirrors dashboard)
autoresearch_results/           # per-experiment artifacts (committed)
memory/                         # checkpoint logs (committed)
tests/test_smoke.py             # smoke tests for loader/audit/runner
CLAUDE.md                       # 52-section operational protocol
```

---

## The three gates

Every experiment must pass three gates before its row is allowed into
the leaderboard:

### Gate 1 — Data-split audit (`core/evaluation/audit.py`)

Seven auditors, all of which must pass:

1. **`audit_split_disjoint`** — pairwise intersection of train/val/test
   row indices is exactly `set()`
2. **`audit_split_protocol`** — split matches Baldi 2014 prescription
   (rows `[0, 10M) / [10M, 10.5M) / [10.5M, 11M)`)
3. **`audit_class_balance`** — y∈{0,1} on every split, both classes
   present, ratio within plausible range (Higgs is ~53% signal)
4. **`audit_size_floors`** — every split has ≥ minimum size
5. **`audit_no_leakage_via_metadata`** — there is no row-identifier or
   "event-id" feature that would allow memorisation
6. **`audit_reproducibility`** — re-loading produces byte-identical
   `X.tobytes() / y.tobytes()` SHA-256
7. **`audit_feature_consistency`** — all splits have the same 28
   columns in the same order

Output: `autoresearch_results/data_split_audit.json` +
`autoresearch_results/data_split_audit.md` +
`autoresearch_results/data_split_fingerprint.txt`. The fingerprint
file is a hash of the union of train/val/test row indices. **Every
experiment row records this fingerprint** — if it ever changes, every
prior leaderboard row is invalidated by definition.

### Gate 2 — Citation Rigor

Every reasoning entry must cite at least one paper or
authoritative source for *every* hyperparameter that differs from the
backbone's registry default. Format:

> `lightgbm.num_leaves = 256` per Ke 2017 §3.2 ("LightGBM: A Highly
> Efficient Gradient Boosting Decision Tree", NeurIPS) — recommended
> for datasets with > 1M rows where the default 31 underfits.

A citation must include: (a) author + year, (b) title or venue, (c)
section/figure/table reference, (d) the *reason* the change is
expected to help on Higgs. Bare arxiv URLs are rejected.

### Gate 3 — Reasoning Blob Completeness

Every reasoning entry must have all 7 sections (`diagnose`, `cite`,
`hypothesize`, `predict`, `run`, `analyze`, `checkpoint`) and meet
per-section word floors:

| Section      | Min words | What goes there                                 |
|--------------|----------:|-------------------------------------------------|
| `diagnose`   | 25        | What is wrong / unknown about prior experiments |
| `cite`       | 30        | Citation Rigor block (Gate 2)                   |
| `hypothesize`| 30        | What HP change will do, why, on which feature   |
| `predict`    | 25        | Quantitative prediction (delta AUROC, sign)     |
| `run`        | 15        | Recipe id + diff vs predecessor                 |
| `analyze`    | 50        | Post-run: what happened vs predicted            |
| `checkpoint` | 15        | What this exp adds to the campaign              |

Pre-run: only the first 5 sections are required. Post-run: all 7. The
runner refuses to start training without a passing pre-run blob, and
refuses to write the result row without a passing post-run blob.

---

## Composite metric (fingerprinted)

```python
composite = min(test_auc, val_auc) - 0.1 * abs(test_auc - val_auc)
```

This penalises both raw quality (low `min`) and val/test drift (large
`abs` gap). The exact formula string is **SHA-256 hashed** at runner
boot; the hash is stored in `autoresearch_results/composite.fingerprint`
and embedded in every result row. If anyone tries to silently swap the
formula mid-campaign, the runner refuses to start (Goodhart-fingerprinting).

---

## Running the campaign

```bash
# 1. install
pip install -e .

# 2. download (~2.8 GB; one-time)
python scripts/download_higgs.py

# 3. audit the split (must pass before any experiment)
python -m core.evaluation.audit --config configs/higgs.yaml --triple-check

# 4. run one experiment (recipe id from sota_catalog.yaml)
python -m core.runner --config configs/higgs.yaml --recipe lightgbm_default

# 5. or: run the full 350-experiment campaign
python scripts/run_campaign.py --config configs/higgs.yaml
```

The runner is **one-experiment-per-call**: it picks up the next
experiment number from `autoresearch_results/`, runs it, writes
artifacts, and exits. The campaign script wraps this in a loop with
crash-restart.

---

## Hardware contract

- **CPU**: P-cores 0-15 only (E-cores 16/17/24/25 banned via
  `psutil.Process.cpu_affinity`)
- **VRAM**: 16 GB cap (RTX 4060 Ti or equivalent)
- **Mixed precision**: BF16 default; FP32 fallback for backbones that
  don't support BF16
- **Determinism**: `torch.manual_seed`, `np.random.seed`, `random.seed`
  fixed per experiment; `cudnn.deterministic=True`

---

## License

MIT — see [LICENSE](LICENSE).
