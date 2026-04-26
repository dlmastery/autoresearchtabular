# Beating XGBoost on Higgs Without Cheating: A 350-Experiment Audit-Gated Tabular ML Campaign

**Authors:** dlmastery / autoresearch project
**Status:** *Live document — populated by the autonomous research loop as experiments complete.*
**Repo:** https://github.com/dlmastery/autoresearchtabular
**Dashboard:** https://dlmastery.github.io/autoresearchtabular/

---

## Abstract

We run an autonomous, audit-gated research campaign on the Higgs UCI
tabular benchmark (Baldi, Sadowski & Whiteson 2014, *Nature
Communications*, arXiv:1402.4735) — 11 M rows, 28 features, binary
signal/background classification, with the frozen 10 M / 500 k / 500 k
Baldi 2014 split. The campaign runs 25 experiments per backbone across
14 backbones (logistic regression and random forest baselines; the
gradient boosting trio LightGBM / XGBoost / CatBoost; the MLP family
including PLR-encoded variants; deep tabular transformers FT-Transformer,
SAINT, NODE, TabNet; and recent SOTA TabM, TabPFN-v2, Trompt,
ExcelFormer) under three programmatic gates: a frozen data-split audit
with seven auditors, a Citation Rigor gate that rejects undocumented
hyperparameter changes, and a Reasoning Blob Completeness gate that
forbids ungrounded experiments. The composite leaderboard metric
`min(test_auc, val_auc) - 0.1 · |test_auc − val_auc|` is SHA-256
fingerprinted at runner boot to prevent silent reformulation.

*Quantitative results table populated as experiments complete.*

---

## 1. Introduction

### 1.1 Why Higgs UCI?

The Higgs UCI benchmark is the canonical *large-scale* tabular ML
benchmark with **publishable headroom in 2026**. The 2014 Baldi paper
established baselines of XGBoost-style boosted trees at 0.810 AUROC and
deep neural networks at 0.885 AUROC on the seven engineered "high-level"
features. A decade of subsequent work has slowly closed the gap: modern
XGBoost reaches ~0.864, FT-Transformer ~0.880, SAINT ~0.881, and the
2024 TabM architecture reaches ~0.886 on the *raw* 21 low-level features
plus 7 high-level features — the configuration we use throughout. The
delta from a vanilla XGBoost run to recent SOTA is ~2.2 AUROC points,
spread over 10+ years and dozens of papers. There is real, slow, hard
signal here.

The dataset is industrially relevant: the LHC ATLAS and CMS
collaborations use this exact problem class — discriminating signal jet
events from QCD background using kinematic features — in their software
trigger systems. Every published improvement at the AUROC level
translates directly into more signal events surviving the trigger
budget, with downstream impact on physics measurements at the percent
level.

### 1.2 Why an autonomous audit-gated campaign?

ML research has a reproducibility crisis. The standard pattern in
academic tabular ML is: pick a method, tune until you beat the
last-published number on whatever metric you can find, ignore the rest,
publish. We instead run the campaign under three gates that *enforce*
reproducibility before a result row is allowed in:

1. **Data-split audit gate** — seven auditors verify train/val/test
   disjointness, Baldi 2014 protocol compliance, class balance, size
   floors, no-leakage-via-metadata, byte-identical reproducibility, and
   feature-column consistency. Every experiment ships a SHA-256
   fingerprint of the row-index union; if the fingerprint changes, every
   prior leaderboard row is invalidated by construction.

2. **Citation Rigor gate** — every hyperparameter that differs from the
   backbone's registry default must cite a primary source (author, year,
   venue, section/table/figure) with a stated reason for why the change
   should help on Higgs. Bare URLs and unsupported folklore ("LightGBM
   likes 256 leaves") are rejected.

3. **Reasoning Blob Completeness gate** — every experiment carries a
   7-section reasoning blob (`diagnose / cite / hypothesize / predict /
   run / analyze / checkpoint`) with per-section word floors. The runner
   refuses to start training without a passing pre-run blob and refuses
   to write the result row without a passing post-run blob.

The campaign is the third in the **autoresearch** series; the same
gates were validated on FX forecasting (autoresearch, 79 experiments)
and computer vision (autoresearchimage, 21 experiments).

### 1.3 Research questions

- **RQ1.** With 25 experiments per backbone and Citation Rigor enforced,
  can we reach published-SOTA composite (≥ 0.880) on Higgs without
  cheating on the Baldi split?
- **RQ2.** What is the marginal value of recent (2022 +) deep tabular
  architectures (SAINT, TabM, FT-Transformer, ExcelFormer) over
  hyperparameter-tuned gradient boosting (LightGBM/XGBoost/CatBoost) on
  Higgs?
- **RQ3.** Does the val/test gap penalty in the composite formula
  meaningfully change which architecture wins, vs. raw test-AUROC
  ranking?

---

## 2. Dataset and split

### 2.1 Source

`https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz`

11,000,000 rows. Each row is one simulated proton-proton collision
event. Column 0 is the binary label (1 = signal: `gg → H → WW → ℓνℓν`,
0 = background: `tt̄`); columns 1-21 are 21 low-level kinematic
features (lepton pT/η/φ, missing energy magnitude/φ, four-jet pT/η/φ
+ b-tag); columns 22-28 are 7 high-level engineered features (m_jj,
m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb).

### 2.2 Frozen split (Baldi 2014)

| Split | Row range          | Size      |
|------:|--------------------|----------:|
| train | `[0, 10,000,000)`  | 10,000,000|
| val   | `[10,000,000, 10,500,000)` | 500,000  |
| test  | `[10,500,000, 11,000,000)` | 500,000  |

This split has been used by every paper since 2014. It is *contiguous*
in the original CSV, deterministic, and the row-index union has SHA-256
fingerprint **(populated by audit on first run)**.

### 2.3 Audit results

(Populated by `core/evaluation/audit.py` on first run — see
`autoresearch_results/data_split_audit.md` for the live audit report.)

---

## 3. Method

### 3.1 Backbone catalog

We evaluate 14 backbones organized in three tiers:

**Tier 1 — Classical baselines + gradient boosting:**

- **`logistic_regression`** — sklearn `LogisticRegression(max_iter=1000)`
  on standardized features. Per Hastie 2009 §4.4 ("Elements of Statistical
  Learning"). Floor / sanity check.
- **`random_forest`** — sklearn `RandomForestClassifier(n_estimators=500)`.
  Per Breiman 2001 ("Random Forests", Machine Learning 45:5-32).
- **`lightgbm`** — Ke 2017 NeurIPS ("LightGBM: A Highly Efficient
  Gradient Boosting Decision Tree"). Default registry params; 25
  experiments vary `num_leaves`, `min_data_in_leaf`, `learning_rate`,
  `feature_fraction`, `bagging_fraction`, `lambda_l1/l2`, `max_bin`,
  `objective`/`is_unbalance`.
- **`xgboost`** — Chen & Guestrin 2016 KDD ("XGBoost: A Scalable Tree
  Boosting System"). 25 experiments vary `max_depth`, `eta`, `subsample`,
  `colsample_bytree`, `gamma`, `lambda`, `alpha`, `tree_method`.
- **`catboost`** — Prokhorenkova 2018 NeurIPS ("CatBoost: unbiased
  boosting with categorical features"). 25 experiments vary `depth`,
  `l2_leaf_reg`, `learning_rate`, `bagging_temperature`,
  `random_strength`, `border_count`.

**Tier 2 — Deep tabular:**

- **`mlp`** — vanilla 3-layer MLP with BN + GELU + dropout. Per
  Gorishniy 2021 ("Revisiting Deep Learning Models for Tabular Data",
  NeurIPS) §4.1 baseline.
- **`mlp_plr`** — MLP with PLR (periodic + linear + ReLU) numerical
  embeddings. Per Gorishniy 2022 ("On Embeddings for Numerical Features
  in Tabular Deep Learning", NeurIPS).
- **`ft_transformer`** — Gorishniy 2021 NeurIPS §4.4. Feature
  Tokenizer + Transformer.
- **`saint`** — Somepalli 2022 ("SAINT: Improved Neural Networks for
  Tabular Data via Row Attention and Contrastive Pre-Training", arXiv
  2106.01342). Row + column attention.
- **`node`** — Popov 2020 ICLR ("Neural Oblivious Decision Ensembles
  for Deep Learning on Tabular Data").
- **`tabnet`** — Arik & Pfister 2021 AAAI ("TabNet: Attentive
  Interpretable Tabular Learning").

**Tier 3 — 2023 + frontier:**

- **`tabm`** — Gorishniy 2024 ("TabM: Advancing Tabular Deep Learning
  with Parameter-Efficient Ensembling"). Multi-head MLP ensembling.
- **`tabpfn_v2`** — Hollmann 2025 *Nature* ("Accurate predictions on
  small data with a tabular foundation model"). In-context tabular
  predictor.
- **`trompt`** — Chen 2023 ("Trompt: Towards a Better Deep Neural
  Network for Tabular Data", ICML).
- **`excelformer`** — Chen 2024 ("ExcelFormer: A Neural Network Surpassing
  GBDTs on Tabular Data", KDD).

Full Citation Rigor blocks for each backbone are in
[`sota_catalog.yaml`](sota_catalog.yaml).

### 3.2 Composite metric

```python
composite = min(test_auc, val_auc) - 0.1 * abs(test_auc - val_auc)
```

The minimum penalises raw quality on the worse of the two splits; the
absolute-gap term penalises val/test drift. This formulation is the
same one validated on FX (autoresearch) and CV (autoresearchimage). The
formula string is SHA-256 hashed at runner boot and the hash embedded
in every result row.

### 3.3 Per-experiment recipe

Each of the 25 experiments per backbone changes **exactly one
hyperparameter family at a time**, with a paper-cited reason. The
sequence within each backbone is:

1-5: registry default + 4 paper-cited "default best practice" tweaks.
6-15: capacity sweep (depth/width for NNs; trees/leaves for GBMs).
16-20: regularisation sweep (dropout, weight decay, L1/L2, bagging).
21-25: feature engineering / advanced tricks (focal loss, class
weights, label smoothing, target encoding for cat features, MixUp).

Per-experiment artefacts: `recipe.yaml`, `metrics.json`,
`reasoning_blob.json` (pre + post-run), `predictions.csv` (per-row
`val` and `test` prediction probabilities), `model_checkpoint.{pt,joblib}`
*(gitignored if > 100 MB)*, `data_split_fingerprint.txt`.

---

## 4. Results

_Live snapshot — last updated 2026-04-26T22:31:57Z._

- **Total experiments completed:** 97
- **Backbones with results:** 8 (`catboost, ft_transformer, lightgbm, logistic_regression, mlp_plr, resnet_tabular, tabm, xgboost`)
- **Global champion:** experiment #95 on `ft_transformer` — composite **0.8723** (val_auc 0.8723, test_auc 0.8726, val/test gap 0.0003)
- **Data-split fingerprint (Baldi 2014, subset_train_n=1M):** `eee12999eeae3950c0c27295…`
- **Composite formula fingerprint:** SHA-256 of `min(test_auc, val_auc) - 0.1 * abs(test_auc - val_auc)`

### 4.1 Per-backbone leaderboard

| backbone | n_exps | best composite | best test_auc | best val_auc | within-backbone gap (max−min composite) |
|---|---:|---:|---:|---:|---:|
| `catboost` | 16 | **0.8355** | 0.8355 | 0.8357 | 0.0250 |
| `ft_transformer` | 1 | **0.8723** | 0.8726 | 0.8723 | 0.0000 |
| `lightgbm` | 25 | **0.8401** | 0.8401 | 0.8403 | 0.0145 |
| `logistic_regression` | 25 | **0.6846** | 0.6846 | 0.6849 | 0.0001 |
| `mlp_plr` | 1 | **0.8623** | 0.8623 | 0.8623 | 0.0000 |
| `resnet_tabular` | 1 | **0.8678** | 0.8685 | 0.8679 | 0.0000 |
| `tabm` | 3 | **0.8675** | 0.8679 | 0.8676 | 0.0320 |
| `xgboost` | 25 | **0.8403** | 0.8403 | 0.8403 | 0.0238 |

### 4.2 Global top 10 by composite

| rank | exp | backbone | composite | val_auc | test_auc | val/test gap | description |
|---|---:|---|---:|---:|---:|---:|---|
| 1 | 95 | `ft_transformer` | **0.8723** | 0.8723 | 0.8726 | 0.0003 | PAPER_DEFAULT_FULL_10M ft_transformer recipe#1: paper Higgs  |
| 2 | 97 | `resnet_tabular` | **0.8678** | 0.8679 | 0.8685 | 0.0006 | PAPER_DEFAULT_FULL_10M resnet_tabular recipe#1: paper Higgs  |
| 3 | 92 | `tabm` | **0.8675** | 0.8676 | 0.8679 | 0.0003 | exp92 [tabm#1] paper Higgs default k=32 h=512x3 lr=2e-3 |
| 4 | 96 | `mlp_plr` | **0.8623** | 0.8623 | 0.8623 | 0.0000 | PAPER_DEFAULT_FULL_10M mlp_plr recipe#1: paper Higgs default |
| 5 | 69 | `xgboost` | **0.8403** | 0.8403 | 0.8403 | 0.0001 | exp69 [xgboost#19] depth 10 lr 0.01 iters 5000 |
| 6 | 47 | `lightgbm` | **0.8401** | 0.8403 | 0.8401 | 0.0002 | exp47 [lightgbm#22] leaves 511 lr 0.01 iters 5000 |
| 7 | 54 | `xgboost` | **0.8374** | 0.8374 | 0.8374 | 0.0000 | exp54 [xgboost#4] depth 10 |
| 8 | 94 | `tabm` | **0.8374** | 0.8374 | 0.8374 | 0.0000 | exp94 [tabm#3] @ train=1M k=16 ensemble |
| 9 | 55 | `xgboost` | **0.8373** | 0.8376 | 0.8373 | 0.0003 | exp55 [xgboost#5] depth 12 |
| 10 | 30 | `lightgbm` | **0.8370** | 0.8373 | 0.8371 | 0.0002 | exp30 [lightgbm#5] leaves 511 |

### 4.3 Notes

All test-AUROC values are computed on the Baldi 2014 frozen test split (rows `[10,500,000, 11,000,000)`); val on rows `[10,000,000, 10,500,000)`; train on the first 10,000,000 rows of `[0, 10,000,000)`. Every experiment row carries the data-split fingerprint and the composite-formula fingerprint recorded in `autoresearch_results/data_split_audit.json` and `autoresearch_results/.composite_fingerprint.json`.

## 5. Discussion

*Pending.*

---

## 6. Reproducibility

Every result in this paper is reproducible from this repository at
fingerprinted commit hashes. To replicate:

```bash
git clone https://github.com/dlmastery/autoresearchtabular.git
cd autoresearchtabular
pip install -e .
python scripts/download_higgs.py
python -m core.evaluation.audit --config configs/higgs.yaml --triple-check
python scripts/run_campaign.py --config configs/higgs.yaml --backbone lightgbm
# ...etc per backbone
```

Each experiment row in
[`autoresearch_results/all_runs.csv`](autoresearch_results/) carries the
data-split fingerprint, the composite-formula fingerprint, the recipe
hash, and the git commit hash at the time of run.

---

## 7. References

(See [`sota_catalog.yaml`](sota_catalog.yaml) for full citation blocks.)

- Baldi, P., Sadowski, P., & Whiteson, D. (2014). Searching for exotic
  particles in high-energy physics with deep learning. *Nature
  Communications*, 5(1), 4308. arXiv:1402.4735.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting
  system. *KDD*.
- Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A highly
  efficient gradient boosting decision tree. *NeurIPS*.
- Prokhorenkova, L., Gusev, G., Vorobev, A., et al. (2018). CatBoost:
  unbiased boosting with categorical features. *NeurIPS*.
- Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021).
  Revisiting deep learning models for tabular data. *NeurIPS*.
- Somepalli, G., Goldblum, M., Schwarzschild, A., et al. (2022). SAINT:
  Improved neural networks for tabular data via row attention and
  contrastive pre-training. arXiv:2106.01342.
- Hollmann, N., Müller, S., Purucker, L., et al. (2025). Accurate
  predictions on small data with a tabular foundation model. *Nature*,
  637, 319-326.
- Gorishniy, Y., et al. (2024). TabM: Advancing tabular deep learning
  with parameter-efficient ensembling. arXiv:2410.24210.

---

## 8. Acknowledgements

This campaign is part of the **autoresearch** series. The
audit-gated research protocol was developed and validated on FX
forecasting and computer vision (WILDS-Camelyon17 + PathMNIST) before
being retargeted at tabular ML. The 7-step `diagnose → cite →
hypothesize → predict → run → analyze → checkpoint` protocol is the
operating discipline that makes the autonomous loop safe to leave
running overnight.
