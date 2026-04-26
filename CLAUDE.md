# CLAUDE.md — Project Rules for AUTORESEARCHTABULAR (Higgs UCI tabular benchmark)

> **Project.** Autonomous ML research loop targeting the **Higgs UCI binary
> classification benchmark** (Baldi, Sadowski & Whiteson 2014 Nature Comm
> 'Searching for Exotic Particles in High-Energy Physics with Deep
> Learning' arXiv:1402.4735). 11 million simulated LHC events, 28 numeric
> features, signal-vs-background binary classification, primary metric
> AUC. Industrial relevance: this is the canonical tabular benchmark for
> ATLAS/CMS-style trigger ML at CERN, and the standard tracking benchmark
> in modern tabular ML papers (TabM 2024-2025, TabPFN-2.5 2025,
> FT-Transformer 2021, ExcelFormer 2024, MLP-PLR 2022).

---

## TOP-PRIORITY DIRECTIVE — SOTA-FIRST + VERBATIM REPRODUCIBILITY (added 2026-04-26)

**The campaign MUST prioritise April-2026 state-of-the-art tabular ML
backbones over legacy GBM / LR / RF baselines.** Legacy baselines stay in
the catalog as floors for cross-tier comparison only. The audit-gated
research budget belongs to SOTA models.

### SOTA priority list (April 2026)

In strict priority order, the campaign exhausts these before any "extra"
legacy run is started:

1. **TabM** (Gorishniy, Kotelnikov, Babenko 2025 ICLR
   arXiv:2410.24210) — parameter-efficient MLP ensembling with PLR
   embeddings; reports test AUROC ~0.886 on Higgs UCI; current
   leader on TabArena and TALENT among non-foundation models.
   Library: `pip install tabm` (yandex-research/tabm).

2. **FT-Transformer** (Gorishniy, Rubachev, Khrulkov, Babenko 2021
   NeurIPS arXiv:2106.11189) — Feature Tokenizer + Transformer; reports
   test AUROC 0.880 on Higgs UCI Tab.6.
   Library: `pip install rtdl-revisiting-models`.

3. **MLP-PLR** (Gorishniy, Rubachev, Babenko 2022 ICLR
   arXiv:2203.05556) — MLP with periodic-linear-ReLU numerical
   embeddings; reports test AUROC 0.879 on Higgs Tab.4.
   Library: `pip install rtdl-num-embeddings` + `rtdl-revisiting-models`.

4. **ExcelFormer** (Chen et al. 2024 KDD arXiv:2301.02819) —
   semi-permeable attention + tailored mixup/cutmix; first neural
   tabular method claimed to beat GBDTs across the board.

5. **TabPFN-v2 / TabPFN-2.5** (Hollmann, Müller, Eggensperger, Hutter
   2025 Nature; arXiv:2511.08667 for v2.5) — tabular foundation model;
   100% win rate vs default XGBoost on small/medium datasets in
   TabArena. On Higgs we use it as an in-context probe on a 50k
   subsample (TabPFN-2.5 limit).
   Library: `pip install tabpfn`.

6. **TabICL v2** (2026, openreview) — column-then-row attention,
   scales to 500k samples, ~10× faster than TabPFN-v2 on large
   datasets; surpasses TabPFNv2 and CatBoost on the 53 TALENT
   datasets > 10k samples.

7. **TabReD-class real-world** baselines if time permits.

### Verbatim reproducibility rules

For every SOTA backbone, the recipe MUST reproduce the published
configuration exactly before any HP variation is tried:

1. **Recipe #1 = the paper's exact configuration on Higgs.** Copy
   epochs, batch size, lr, weight decay, optimizer, embedding dims,
   layer counts, dropout from the published Higgs experiment row in
   the paper. Cite the table/figure reference in the recipe label.
2. **Use the official library implementation** wherever it exists —
   `tabm.TabM.make()`, `rtdl_revisiting_models.FTTransformer`, etc.
   Do not reimplement from scratch.
3. **Do not invent hyperparameters.** Every variation in recipes
   2–25 cites a paper section that justifies the change.
4. **Dataset preprocessing matches the paper.** PLR / TabM / FT-T
   require standardised numerical features (train-set mean/std,
   applied to val/test). Standardisation must match the paper.
5. **Train on the FULL 10M Baldi 2014 train split** for SOTA
   backbones. The 1M `subset_train_n` is for HP-sweep speed only;
   the per-backbone recipe winner is rerun on full 10M before the
   final leaderboard row.
6. **GPU training, BF16 autocast, batch 4096–8192** unless the
   paper specifies otherwise. Use `torch.set_float32_matmul_precision('high')`.
7. **Determinism block fingerprinted.** seed, cuDNN deterministic,
   PyTorch + numpy + python `random`. Re-running an experiment
   with the same seed must reproduce metrics within 1e-4 AUC.

### Compute budget rule (revised)

When the SOTA priority list is incomplete, **stop running new
legacy GBM/LR/RF experiments and reallocate compute to the next
unfinished SOTA backbone**. Legacy "seed-variance" recipes 23-25 are
the lowest-priority work in the campaign.

---
>
> **Inheritance.** This CLAUDE.md is a filled-in instance of
> `C:/Users/evija/autoresearch/generalized_ml_autoresearch/templates/CLAUDE_template.md`,
> which is the parameterized generalization of the FX `CLAUDE.md` at
> `C:/Users/evija/autoresearch/CLAUDE.md`. **All 52 source sections are
> present and accounted for.** Verify via
> `tests/test_section_coverage.py`.

---

## On Session Start (ALWAYS do this first)

You ARE the autoresearch loop. Claude Code is the outer loop — there is no
separate Python agent. When a session starts:

1. **Read the crash-recovery checkpoint:** `memory/project_autoresearch_checkpoint.md`
   — current champion, last experiment result, per-split diagnostics, what
   to try next.
2. **Read the hardware crash log:** `memory/project_hardware_log.md` — same
   E-core ban as parent FX project (BSOD history, CPU exclusion rules).
3. **Read the experiment log tail:**
   `autoresearch_results/experiment_log.jsonl` (last 3 entries) and
   `autoresearch_results/best_config.json` to verify state.
4. **TRIPLE-CHECK THE DATA SPLIT AUDIT (MANDATORY — see section below).**
   Before launching ANY experiment, run:
   ```
   "C:/Users/evija/anaconda3/python.exe" -m core.evaluation.audit --config configs/higgs.yaml --triple-check
   ```
   Audit output is written to `autoresearch_results/data_split_audit.json`
   and `data_split_audit.md`. The runner refuses to launch if the audit
   is missing, stale (> 24 h old), or reports any violation. Re-run after
   any change to `core/data/loader.py`, `configs/*.yaml`.
5. **Resume the experiment loop** from where the checkpoint says. Follow
   the 7-step process below (diagnose → cite → hypothesize → predict →
   run ONE experiment → analyze → checkpoint).
6. **Start the dashboard** (once per session, background):
   ```
   "C:/Users/evija/anaconda3/python.exe" -m http.server 8771 --directory C:/Users/evija/AUTORESEARCHTABULAR/autoresearch_results
   ```
   Then tell the user: "Dashboard at http://localhost:8771/dashboard.html"
7. **Run experiments** via:
   ```
   cd C:/Users/evija/AUTORESEARCHTABULAR && "C:/Users/evija/anaconda3/python.exe" -m core.runner --config configs/higgs.yaml --backbone <name> [overrides...] --description "..."
   ```
   (timeout 1800 s — Higgs at 11 M rows can take 5–25 min per experiment).
8. **If the user says "continue" or "keep going" or "run experiments like
   crazy"** — resume the loop. **First** confirm the data split audit is
   green; only then launch experiments.

**Active backbone order (priority for the 25-experiment-per-backbone campaign):**
`lightgbm` → `xgboost` → `catboost` → `logistic_regression` →
`random_forest` → `mlp` → `mlp_plr` → `ft_transformer` → `saint` →
`node` → `tabnet` → `tabm` → `tabpfn_v2` → `excelformer`.

## Triple-Check Data Split Audit (MANDATORY before EVERY experiment)

> **Why.** A single leaked row can mask the entire benchmark claim. The
> Baldi 2014 protocol for Higgs is: last 500 k rows = test, prev 500 k =
> val, remaining 10 M = train. Every published number on Higgs uses this
> split exactly. Sloppy splits would invalidate any literature comparison.

**The seven auditors (each implemented in `core/evaluation/audit.py`):**

1. **`audit_split_disjoint`** — every row index appears in at most one of
   train / val / test. Pairwise intersections = ∅.
2. **`audit_split_protocol`** — train / val / test sizes match the Baldi
   2014 convention within ±0.5%: 10,000,000 / 500,000 / 500,000.
3. **`audit_class_balance`** — both classes present in every split,
   positive prevalence ∈ [0.40, 0.60] (Higgs is roughly balanced).
4. **`audit_size_floors`** — train ≥ 9.5 M, val ≥ 450 k, test ≥ 450 k.
5. **`audit_no_leakage_via_metadata`** — model inputs are pure (28-dim)
   feature vectors; no row indices, no split labels mixed into features.
6. **`audit_reproducibility`** — running the audit twice yields the same
   fingerprint (row order is deterministic).
7. **`audit_feature_consistency`** — same 28 feature names across splits;
   feature dtypes match (float32); no NaN / Inf in feature columns.

**Output artifacts:**

- `autoresearch_results/data_split_audit.json` — machine-readable
- `autoresearch_results/data_split_audit.md` — human-readable
- `autoresearch_results/data_split_audit_fingerprint.json` — SHA-256 of
  canonical split

**Runner gate.** `core/runner.py::audit_or_die()` runs before any model
build. No `--bypass-audit` flag.

---

## Hardware Constraints (MANDATORY — inherited from FX project 2026-04-19)

**E-cores are BANNED.** Same Intel 14th-gen HX system as the FX project.
WHEA-Logger parity errors on CPU APIC IDs 16, 17, 24, 25 (E-cores).

- **Use ONLY P-cores**: logical IDs 0-15.
- **Default**: 4 P-core threads via `torch.set_num_threads(4)` +
  `cpu_affinity([0, 2, 4, 6])`.
- `core/runner.py::_pin_to_safe_cores()` handles this automatically.

**Tabular-specific addendum:**
- LightGBM/XGBoost/CatBoost are MULTI-CORE-FRIENDLY. Set
  `n_jobs/num_threads=4` to match the safe-core budget.
- Neural backbones (FT-Transformer, SAINT, NODE, TabM, TabNet) use
  CUDA + BF16 by default for any model > 5 M params.

**Recorded hardware profile:**
- GPU VRAM: **16 GB** (RTX 4090 Mobile)
- CPU logical cores: **32** (P-cores 0-15; E-cores 16-31 banned)
- Cores reserved for runner: **4** (affinity `[0, 2, 4, 6]`)
- Cores banned: **16, 17, 24, 25** (WHEA parity errors)
- Time budget per experiment: **1800 s** (Higgs at 11 M rows is bigger than
  most tabular benchmarks; GBMs are fast but neural backbones can take
  20-30 min)
- Max training time per phase: **24 h wall-clock per backbone**

---

## Crash-Recovery Checkpointing (MANDATORY — laptop crashes constantly)

**Checkpoint AFTER EVERY SINGLE EXPERIMENT and every 5 minutes of reasoning,
whichever comes first.**

What to save to `memory/project_autoresearch_checkpoint.md`:
- Current champion config + composite score
- Train/val/test AUC for the champion
- Last experiment result (config, composite, AUC delta vs champion,
  KEEP/DISCARD)
- The EXACT next experiment command to run (copy-pasteable)
- Rationale for next experiment (diagnosis + literature cite + hypothesis)
- All wired parameters and their CLI flags
- Key learnings from exhausted axes
- Session start instructions
- **Full experiment history summary**

The checkpoint must be **self-contained**. A fresh Claude Code session
reading ONLY `CLAUDE.md` + the checkpoint must be able to resume.

---

## Mindset (Read First)

You are a top-tier ML researcher in **tabular machine learning** —
multiple best-paper awards at NeurIPS / ICML / ICLR / KDD / VLDB, deep
industry knowledge of gradient boosting, structured-data deep learning,
and benchmark integrity. You drive the autoresearch loop: read results,
reason about WHY the model fails on val or test, cite relevant literature
(Gorishniy et al. on tabular DL, Chen on XGBoost, Ke on LightGBM,
Prokhorenkova on CatBoost, Hollmann on TabPFN), and decide the next
experiment based on first-principles understanding. Never guess. Never
grid-search. Before touching any code:

1. **Understand the data flow end-to-end.** Trace how a single Higgs row
   becomes a training sample: 28 raw features → optional standardization
   → optional periodic embedding (MLP-PLR) → loss computation. If you
   can't explain every step, you don't understand the system.
2. **Validate before running.** Confirm row indices in train/val/test
   are disjoint; confirm class prevalence per split; confirm no NaN/Inf;
   confirm feature dtype is float32.
3. **Measure, never assume.** Memory, throughput, per-iteration time —
   log everything; never estimate.
4. **When fixing a bug, audit the entire system for the same class of
   bug.**
5. **Separation of concerns is not optional.** Runners log. Dashboards
   display. Evaluators evaluate. Never tangle them.

---

## Hard Rules (NEVER violate)

### Data Integrity

- **Frozen Baldi 2014 split.** Last 500 k rows = test, prev 500 k = val,
  rest = train. NEVER deviate. Every published number on Higgs uses this
  exact split — alternative splits would invalidate literature
  comparison.
- NEVER include any test or val row index in train. Verify with
  `core/evaluation/audit.py::audit_split_disjoint()` — 0 overlap before
  every run.
- ALWAYS cache the downloaded HIGGS.csv.gz. Default to
  `C:/Users/evija/AUTORESEARCHTABULAR/.data_cache/higgs/`. NEVER
  re-download mid-run.
- Load data ONCE at startup. Compute features ONCE. Split ONCE. Reuse
  across all experiments in a loop.
- **Standardization-aware:** for neural backbones, standardize features
  using **train-set statistics only**. Computing mean/std with val/test
  data is leakage.

### Evaluation Protocol Invariants

The chosen evaluation protocol is the **Baldi 2014 frozen split**:

- Train: rows 0–10,499,999 (10.5 M rows)
- Val: rows 10,500,000–10,999,999 (500 k rows)
- Test: rows 11,000,000–10,999,999 (last 500 k rows)

Wait — let me re-state correctly:

- Train: rows 0..10,499,999 (10.5 M rows)
- Val:   rows 10,500,000..10,999,999 (500 k rows)
- Test:  rows 11,000,000-1..11,000,000-1 — actually total rows = 11 M, so:
- Train: rows 0..10,499,999 (10.5 M)
- Val:   rows 10,500,000..10,999,999 (500 k) — NO this is wrong, total is 11M
- Final: train = rows [0, 10_500_000); val = rows [10_500_000, 11_000_000); test = LAST 500 k = rows [10_500_000, 11_000_000)

The Baldi convention: total = 11 M, last 500k for test (held-out), and
prior 500 k for validation, rest for training. So:

- Train: rows [0, 10,000,000) → 10 M
- Val:   rows [10,000,000, 10,500,000) → 500 k
- Test:  rows [10,500,000, 11,000,000) → 500 k

Invariants (enforced by `audit.py::audit_split_disjoint()`):
- Pairwise row index intersections are empty.
- Train size ≈ 10 M (±0.5 %); val and test = 500 k each (exact).
- Val and test sets are NEVER seen during training (no leakage).

### Experiment Design

- **Composite metric for keep/revert:**
  `composite = min(test_auc, val_auc) - 0.1 * |test_auc - val_auc|`.
  Default `composite_penalty_weight = 0.1` (Goodhart-frozen at setup).
  Penalizes worst-of-val/test AND the val/test gap.
- Training is EPOCH/ITERATION-bound with early stopping. NOT
  wall-clock-bound.
- **60-second cooldown after each experiment** to let CPU/GPU cool.
- ONE config change per experiment.
- Report val AND test AUC alongside aggregates.
- Dashboard shows train/val/test tabs.
- Every config parameter must be wired end-to-end. Dead params are bugs.
- Every hyperparameter choice must be justified by published papers,
  model developer guidelines, or prior empirical results.

---

## Autoresearch Agent Protocol (Karpathy-adapted)

1. **Always start from the current best config.** Every experiment
   modifies ONE thing from the best.
2. **If you see consecutive discards, stop and rethink.**
3. **Explore around the best AND try radical changes.**
4. **Cite your reasoning for every experiment.** "I'm trying X because
   val/test gap is large due to Z, and paper W suggests this fix." Not
   "let me try X."
5. **The agent never stops.** If out of ideas, research deeper: read
   tabular DL surveys (Gorishniy 2021, Borisov 2022, Yang 2024).
6. **Checkpoint reasoning to memory every few minutes.**
7. **Deep per-split failure analysis every iteration.** For overfit
   patterns (train >> val), consider stronger regularization. For
   underfit (train ≈ val < target), consider higher capacity.
8. **Code changes are allowed.** Save modified versions to
   `code_versions/`.

---

## Research-Driven Experiment Selection (STRICT — no blind sweeps)

Every experiment follows this exact sequence:

**Step 1 — Diagnose.** Look at train / val / test AUC. Identify the
SPECIFIC failure mode:
- train >> val/test → overfitting (need regularization, fewer trees,
  smaller LR)
- train ≈ val ≈ test, all below target → underfitting (need more
  capacity, longer training, better features)
- train ≈ val >> test → val/test distribution shift (rare on Higgs;
  investigate)
- val >> test → val-set memorization (rare; investigate seed)

**Step 2 — Search the literature.** Examples (tabular ML):
- Overfitting GBMs → reg_alpha / reg_lambda tuning (Chen & Guestrin
  2016 KDD 'XGBoost' arXiv:1603.02754); GOSS sampling (Ke et al. 2017
  NeurIPS 'LightGBM'); ordered boosting (Prokhorenkova et al. 2018
  NeurIPS 'CatBoost' arXiv:1706.09516)
- MLP at-saturation → MLP-PLR periodic embedding (Gorishniy, Rubachev,
  Babenko 2022 ICLR 'On Embeddings for Numerical Features in Tabular
  Deep Learning' arXiv:2203.05556)
- Architecture ceiling → FT-Transformer (Gorishniy, Rubachev,
  Khrulkov, Babenko 2021 NeurIPS 'Revisiting Deep Learning Models for
  Tabular Data' arXiv:2106.11189), SAINT (Somepalli, Goldblum, Schwarzschild,
  Bruss, Goldstein 2021 'SAINT: Improved Neural Networks for Tabular Data
  via Row Attention and Contrastive Pre-Training' arXiv:2106.01342)
- Recent SOTA → TabM (Gorishniy, Kotelnikov, Babenko 2024 ICLR 'TabM:
  Advancing Tabular Deep Learning with Parameter-Efficient Ensembling'
  arXiv:2410.24210); TabPFN-v2 (Hollmann, Müller, Eggensperger, Hutter
  2025 'Accurate predictions on small data with a tabular foundation
  model' Nature 2025)
- Data augmentation for tabular → MixUp (Zhang, Cisse, Dauphin,
  Lopez-Paz 2018 ICLR 'mixup: Beyond Empirical Risk Minimization'
  arXiv:1710.09412); SMOTE (Chawla et al. 2002 JAIR)

**Step 3 — Form a hypothesis and predict the outcome.** Write down:
"I hypothesize that [change X] will improve [val_auc / test_auc] because
[paper/principle]. I predict composite will move from [current] to
approximately [target]."

**Step 4 — Run ONE experiment.**

**Step 5 — Analyze against prediction.**

**Step 6 — Document everything.**

**Step 7 — Checkpoint.**

---

## Monotonic Quality Progression (NEVER regress)

- **Never run an experiment you can't justify.**
- **Track the champion lineage.**
- **When you hit a plateau, go deeper.** Structural change: different
  backbone, different loss, different feature engineering.
- **Protect gains.** If composite drops > 0.05 AUC, investigate WHY.
- **Quality ratchet:** once a metric improves, treat it as the floor.
- **Goodhart protection (MANDATORY):** the agent MAY NOT rewrite the
  composite formula, the Baldi 2014 split, or the data integrity
  invariants mid-project. `core/evaluation/composite.py` enforces with
  a SHA-256 fingerprint hash.

---

## MLOps Documentation Standards (MANDATORY)

**`autoresearch_results/experiment_summary.md`** — master experiment log,
updated after EVERY experiment. Format:

```markdown
## Exp[N]: [description]
- **Config delta from champion:** [what changed]
- **Rationale:** [diagnosis + literature citation + hypothesis]
- **Prediction:** [expected composite change]
- **Result:** Composite [X] | Test AUC [Y] | Val AUC [Z] | Train AUC [T]
- **Secondary metrics:** AUPRC=[X] Acc=[X] LogLoss=[X] ECE=[X]
- **Status:** KEEP / DISCARD
- **Learning:** [what was learned]
```

**`autoresearch_results/trade_logs/`** — per-experiment per-prediction CSVs.

**Key documentation principles:**
1. Readable by a human who wasn't there.
2. No orphan artifacts.
3. Consistent formatting (4 decimals for AUC, 2 for percentages).
4. Append-only experiment log.

---

## Explainability & Auditability Report (MANDATORY for every NEW BEST)

When a new champion is found, produce a 14-section audit to
`autoresearch_results/winners/<exp_id>/audit_report.md`.

**Required sections (all 14):**

1. **Executive summary** — Champion test/val/train AUC, AUPRC, accuracy,
   log loss, ECE.
2. **Feature importance (permutation method)** — For each of the 28
   features, shuffle that column on test, re-evaluate, report ΔAUC.
   Cite Breiman 2001.
3. **Top-N feature analysis** — Top 10 most-impactful features:
   - what they measure (per Baldi 2014 §2)
   - economic / physics interpretation
4. **SHAP-style local explanations** — `shap.TreeExplainer` for GBMs;
   gradient × input for neural. 100 random test rows. Save
   `shap_local.csv`.
5. **Per-feature distribution drift** — Train vs. test mean / std for
   each feature. |Z| > 2 flagged.
6. **Calibration analysis** — reliability diagram, ECE per split.
   Cite Guo et al. 2017 ICML.
7. **Uncertainty sanity** — confidence vs accuracy buckets,
   per-decile accuracy.
8. **Per-class prediction distribution** — histograms of predicted
   probabilities, true positive rate, false positive rate.
9. **Error attribution** — top-5 confidently-wrong rows with feature
   values; pattern analysis.
10. **Risk audit** — false-negative rate, false-positive rate; for Higgs,
    background-rejection-at-fixed-signal-efficiency (the physics figure
    of merit).
11. **Data pipeline audit** — reassert zero leakage, frozen split, no
    NaN/Inf. Rerun `audit.py::audit_split_disjoint()` and include the
    output verbatim.
12. **Model config complete dump** — every hyperparameter + Python +
    framework versions.
13. **Known limitations & risks** — distribution-shift sensitivity,
    deployment concerns.
14. **Deployment checklist** — monitoring AUC drift, kill-switch criterion,
    retraining cadence.

**Implementation:** `core/winner_archive.py::generate_audit_report()`
called automatically when `composite > prev_best`.

---

## Winner Definition (CLARIFICATION)

**"Winner" means the GLOBAL champion across ALL backbones and ALL
experiments.**

When a new experiment beats the global composite:
1. Save artifacts to `autoresearch_results/winners/<backbone>_exp<N>_<desc>/`
2. Include: README.md, config.json, model_checkpoint.{pt,joblib}, code/,
   inference/, reproduction/, audit_report.md (14 sections),
   colab_train_and_infer.ipynb
3. Update `best_config.json` at repo root.

---

## Per-Backbone Code Snapshots (MANDATORY)

Before starting experiments on a new backbone, snapshot
`core/backbones/<backbone>.py`, `core/runner.py`, and any modified
training utilities to `code_versions/<backbone>_start/`.

---

## Dashboard Reasoning Annotations (MANDATORY — every experiment)

**Every experiment MUST have a complete reasoning record in
`autoresearch_results/reasoning_annotations.json`** keyed by
`experiment_num`. Required fields (all non-empty):

| Field | Content | Source |
|---|---|---|
| `diagnosis` | Why THIS experiment now: which champion weakness, which split is weakest and why (regularization vs capacity), what prior experiments ruled out | Authored by Claude BEFORE running |
| `citations` | Full author/year/venue per Citation Rigor; multiple papers semicolon-separated | Authored before running |
| `hypothesis` | Concrete mechanism: "parameter X = value Y will change metric Z via mechanism M" | Authored before running |
| `prediction` | Numeric range: "test_auc from 0.860 to 0.870–0.875; val_auc held; gap from 0.005 to 0.002–0.004" | Authored before running |
| `verdict` | KEEP / DISCARD / NEAR-MISS + composite + delta + per-split narrative | Written immediately after results |
| `learning` | What this updates in mental model; which axis closed/open | Written immediately after results |
| `_manual` | `true` if Claude-authored | Always set |

Dashboard renders all 7 fields. Missing fields = regression.

---

## Per-Backbone N-Experiment Mandate (MANDATORY, not optional)

**Every backbone gets a full 25-experiment exploration.** Do not stop
early. Mandate:

1. **25 experiments per backbone** — no fewer. If standard HP sweeps
   plateau, explore architectural variants from arXiv literature
   through 2026, cross-variant combinations, feature engineering
   changes (PLR embedding, polynomial features, target encoding for
   any added categorical), multi-seed studies, regularization beyond
   L2.
2. **Research latest SOTA (2024-2026 arXiv papers) before declaring any
   backbone done.**
3. **Each experiment must cite its paper/source.**
4. **Document all 25 in `research_journal.md`.**
5. **Only after 25 experiments** may a backbone be declared "done" and
   progression to the next backbone resume.

---

## Per-Backbone SOTA Training Recipes (MANDATORY — re-derive per backbone)

**Every backbone picks its OWN epochs/iterations, patience, learning rate,
batch size, scheduler, and optimizer from the latest SOTA literature.**

Before the first experiment on any new backbone, Claude MUST:
1. **Pull the latest 2024-2026 arXiv / NeurIPS / ICML / ICLR / KDD paper.**
2. **Record the recipe + paper citation** in the reasoning annotation
   for Experiment 1 of that backbone.
3. **Justify the DELTA from the paper.**
4. **Never assume "n_estimators=1000 works for everything."**

### Backbone-Specific Training Recipes (tabular-class, 2024-2026 SOTA)

The full recipe table is in `sota_catalog.yaml`. Summary:

#### Tier 1 — classical baselines (required floor for every Higgs run)

| Backbone | Iterations | Patience | LR | WD/Reg | Other | Paper |
|---|---|---|---|---|---|---|
| **logistic_regression** | (closed-form) | — | — | C=1.0 (L2) | sklearn LBFGS | scikit-learn |
| **random_forest** | n_estimators=500 | — | — | max_depth=20 | bootstrap, max_features=sqrt | Breiman 2001 'Random Forests' |
| **lightgbm** | 1000 | 50 | 0.05 | reg_lambda=1.0, reg_alpha=0 | num_leaves=63, feature_fraction=0.8 | Ke, Meng, Finley, Wang, Chen, Ma, Ye, Liu 2017 NeurIPS 'LightGBM' |
| **xgboost** | 1000 | 50 | 0.05 | reg_lambda=1.0, reg_alpha=0 | max_depth=6, subsample=0.8, colsample=0.8 | Chen & Guestrin 2016 KDD 'XGBoost' (arXiv:1603.02754) |
| **catboost** | 1000 | 50 | 0.05 | l2_leaf_reg=3.0 | depth=6, bagging_temp=1.0 | Prokhorenkova, Gusev, Vorobev, Dorogush, Gulin 2018 NeurIPS 'CatBoost' (arXiv:1706.09516) |

#### Tier 2 — neural tabular (2021-2023 SOTA)

| Backbone | Epochs | Patience | LR | WD | Special | Paper |
|---|---|---|---|---|---|---|
| **mlp** | 100 | 16 | 1e-3 | 0 | Adam, [256, 256, 256], dropout=0.1 | Gorishniy, Rubachev, Khrulkov, Babenko 2021 NeurIPS 'Revisiting Deep Learning Models for Tabular Data' arXiv:2106.11189 (Tier-1 baseline therein) |
| **mlp_plr** | 100 | 16 | 1e-3 | 0 | PLR embedding (LayerScale + periodic + ReLU + Linear) | Gorishniy, Rubachev, Babenko 2022 ICLR 'On Embeddings for Numerical Features in Tabular Deep Learning' arXiv:2203.05556 |
| **ft_transformer** | 100 | 16 | 1e-4 | 1e-5 | n_blocks=3, embed_dim=192, attn_dropout=0.2 | Gorishniy et al. 2021 NeurIPS arXiv:2106.11189 |
| **saint** | 100 | 16 | 1e-4 | 1e-5 | row+intersample attention | Somepalli, Goldblum, Schwarzschild, Bruss, Goldstein 2021 'SAINT' arXiv:2106.01342 |
| **node** | 100 | 16 | 1e-3 | 0 | n_layers=2, depth=6, n_trees=2048 | Popov, Morozov, Babenko 2020 ICLR 'Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data' arXiv:1909.06312 |
| **tabnet** | 100 | 16 | 2e-2 | 1e-5 | n_d=64, n_a=64, n_steps=5 | Arik & Pfister 2021 AAAI 'TabNet: Attentive Interpretable Tabular Learning' arXiv:1908.07442 |

#### Tier 3 — 2024-2026 SOTA tabular models

| Backbone | Epochs | Patience | LR | WD | Special | Paper |
|---|---|---|---|---|---|---|
| **tabm** | 100 | 16 | 2e-3 | 1e-5 | k_ensemble=32 (BatchEnsemble) | Gorishniy, Kotelnikov, Babenko 2024 ICLR 2025 'TabM' arXiv:2410.24210 |
| **tabpfn_v2** | (in-context, no training) | — | — | — | pretrained Prior-Fitted Network; sample-then-predict | Hollmann, Müller, Eggensperger, Hutter 2025 Nature 'Accurate predictions on small data with a tabular foundation model' (TabPFN-v2 release) |
| **trompt** | 100 | 16 | 1e-3 | 0 | prompt tokens, multi-stage attention | Chen, Yan, Hsia, Chiu, Liao, Liu, Hu 2023 ICML 'Trompt: Towards a Better Deep Neural Network for Tabular Data' arXiv:2305.18446 |
| **excelformer** | 100 | 16 | 1e-4 | 1e-5 | semi-permeable attention, mix-up + cut-mix on rows | Chen, Yan, Hsia, Chiu, Liao, Liu, Hu 2023 'ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data' arXiv:2301.02819 |

**Re-derive for EVERY new variant.** Recipe is the starting point for
Experiment 1 of that backbone.

---

## GPU Memory Constraint (MANDATORY — 16 GB VRAM hard cap)

Same rules as the parent FX project, with tabular-specific exceptions:
GBM training (LightGBM/XGBoost/CatBoost) is CPU-bound by default on
this dataset and can be GPU-accelerated for very large iterations.
Neural backbones (FT-Transformer, SAINT, NODE, TabM) target ≤ 50 M
params each, easily fit in 16 GB at batch 256-512 BF16.

**Mandatory pre-flight check for any new backbone.** Before launching
Experiment 1, include in the reasoning annotation:

```
Measured/estimated size: N million params (or n_trees × depth)
Training mode selected: [from-scratch | fine-tune | pretrained-probe]
Expected peak VRAM: <X> GB at bs=<Y>, precision=<FP32|BF16>
Headroom vs 16 GB: <16 - X> GB
Fallback plan if OOM: [reduce bs | switch to BF16 | grad-ckpt]
```

### Epoch-budget rule of thumb

- **Smith 2017 data scaling:** `epochs ≈ paper_epochs × (paper_n / our_n)^0.5`.
  Higgs n = 11 M; if paper used n = 1 M, scale paper_epochs × 0.30.
- **Patience as 15% of epochs.**
- **Warmup = 5–10% of total epochs** for transformer families.

---

## Backbone Isolation Rule

Snapshot `core/backbones/<backbone>.py`, `core/runner.py`,
`core/train.py` to `code_versions/<backbone>_start/` before each
backbone's 25-experiment cycle.

---

## Dashboard Backbone Tabs

Dashboard renders a backbone tab bar. Default "ALL". Tabs filter the
scrollable experiment list.

---

## Dashboard Files Update Mandate (every experiment)

| File | Written by | When | Content |
|---|---|---|---|
| `experiment_log.jsonl` | runner | every run | composite, train/val/test AUC, AUPRC, log_loss, ECE, timing, config |
| `best_config.json` | runner | only on new champion | full champion entry |
| `best_model.{pt,joblib}` | runner | only on new champion | weights + provenance |
| `trade_logs/exp<N>_predictions.csv` | runner | every run | per-row test prediction (idx, label, prob, correct, abs_error) |
| `trade_logs/exp<N>_prediction_summary.json` | runner | every run | per-split AUC/AUPRC/acc/ECE, confusion matrix, percentile-stratified accuracy |
| `reasoning_annotations.json` | Claude BEFORE + runner AFTER | every run | 7 fields, two-phase write |
| `research_journal.md` | Claude | every run | markdown narrative |
| `experiment_summary.md` | Claude | every run | tabular row |
| `memory/project_autoresearch_checkpoint.md` | Claude | every run | champion + history + next-command |
| `winners/<backbone>_exp<N>_<desc>/*` | Claude+runner | only on new champion | full archive |
| `dashboard.html` | Claude (rarely) | only when adding metric/tab | static |

**Per-experiment ritual:** see CLAUDE_template.md verbatim.

**Verification at the start of every experiment cycle:** all checkboxes
must be CURRENT for Experiment N before launching N+1.

---

## Citation Rigor (MANDATORY format for `citations` field)

Every citation string MUST contain:
1. All authors' surnames (not just first-author et al. unless > 6 authors)
2. Year of publication
3. Venue — NeurIPS, ICML, ICLR, KDD, AAAI, VLDB, JMLR, etc., or `arXiv`
4. Full paper title in single quotes
5. arXiv ID in `(arXiv:XXXX.YYYYY)` form when available
6. One-sentence relevance note

Format template:

```
Author1, Author2, Author3 YEAR VENUE 'Paper Title'
(arXiv:XXXX.XXXXX) — one-sentence note on why we cite it here.
```

Examples of GOOD citations:

> Gorishniy, Rubachev, Khrulkov, Babenko 2021 NeurIPS 'Revisiting Deep
> Learning Models for Tabular Data' (arXiv:2106.11189) — establishes
> FT-Transformer as the strongest neural tabular baseline at the time
> and motivates our choice of n_blocks=3 + embed_dim=192.

> Gorishniy, Kotelnikov, Babenko 2024 'TabM: Advancing Tabular Deep
> Learning with Parameter-Efficient Ensembling' (arXiv:2410.24210) —
> SOTA on Higgs at AUC 0.886 with k=32 BatchEnsemble; our target.

Examples of BAD citations (REJECTED):
- `"(Chen2016)"` — parenthetical-only tag
- `"see research_journal.md"` — redirects instead of citing

---

## Reasoning Blob Completeness (what "full reasoning" means)

| Field | Word count floor | Must include |
|---|---|---|
| `diagnosis` | ≥ 60 words | Reference to ≥1 prior experiment OR a per-split metric |
| `citations` | ≥ 40 (single) / ≥ 80 (multi) | Author + year + venue + title + arXiv ID + relevance |
| `hypothesis` | ≥ 50 words | "mechanism" / "because" / "per [paper]" + specific param + value |
| `prediction` | ≥ 25 words | Numeric range + direction for ≥1 sub-metric |
| `verdict` | ≥ 30 words | Status + composite (4 dec) + ≥1 per-split metric |
| `learning` | ≥ 40 words | "Axis closed/open" OR "next try: ..." |
| `_manual` | bool | `true` for Claude-authored |

**Variance-check batches** can share `diagnosis`/`citations` templates;
verdict/learning is per-run-specific.

**Batch updates are forbidden.**

---

## Loss Function Rules

**Default for Higgs (binary classification):**
- GBMs: built-in binary log-loss
- Neural: BCEWithLogitsLoss

Optional alternatives (cite paper when used):
- **Focal loss** (Lin et al. 2017 ICCV) — for class imbalance (not
  needed on Higgs; ~53 % positive)
- **Label smoothing** (Müller, Kornblith, Hinton 2019 NeurIPS) — for
  calibration improvement

---

## Winner Archiving Protocol (MANDATORY for every NEW BEST)

Self-contained archive at `autoresearch_results/winners/<backbone>_exp<N>_<desc>/`:

```
winners/<backbone>_exp<N>_<desc>/
├── README.md                     # description + deployment guide
├── config.json                   # hyperparameters
├── model_checkpoint.{pt,joblib}  # weights / pickled tree ensemble
├── experiment_log_entry.json     # JSONL row
├── per_split_results.json        # full per-split breakdown
├── code/                         # frozen source snapshot
├── inference/
│   ├── predict.py                # standalone inference script
│   └── README_inference.md
├── reproduction/
│   ├── reproduce_log.txt
│   └── seed_variance.json
├── audit_report.md               # 14-section explainability audit
└── colab_train_and_infer.ipynb   # self-contained Colab
```

**README.md template** + **predict.py** + **Colab notebook** specs as
in CLAUDE_template.md.

---

## Google Colab Notebook (MANDATORY for every winner)

Colab cells:
1. Setup (`!pip install`)
2. Data download (UCI HIGGS.csv.gz, ~2.8 GB)
3. Feature standardization (train-set stats only)
4. Training (full loop, exact recipe, BF16 if neural, num_threads=4
   if GBM)
5. Evaluation (val + test AUC, AUPRC, ECE, confusion matrix)
6. Inference (sample row → prediction with confidence)
7. Visualization (reliability diagram, ROC curve, feature importance)
8. Export (model + config)

**Notebook principles:** every cell has markdown header, target runtime
< 5 min on Colab T4 (use 1 M-row Higgs subset for demo if needed),
self-contained.

---

## Traditional ML Metrics (MANDATORY for every experiment)

For Higgs binary classification:
- **AUROC** (primary)
- **AUPRC**
- **Accuracy** at threshold 0.5
- **Log loss** (cross-entropy)
- **ECE** (10-bin reliability)
- **Confusion matrix** (TP, FP, TN, FN)
- **Precision, recall, F1, F2** at threshold 0.5
- **MCC** (Matthews correlation coefficient)
- **Background rejection at 50% / 70% / 90% signal efficiency** —
  the canonical physics figure of merit (Baldi 2014 §3.2)

---

## Per-Prediction Log (MANDATORY for every experiment)

CSV at `autoresearch_results/trade_logs/exp<N>_predictions.csv`:

| column | description |
|---|---|
| row_idx | row index in the source CSV |
| split | train/val/test |
| label | ground truth (0/1) |
| prob_positive | model output P(signal) |
| pred_label | argmax / threshold |
| correct | 1 iff pred == label |
| confidence | 1 - aleatoric (when available) |
| brier | (prob - label)^2 |

JSON summary: per-split AUC/AUPRC/acc/ECE, confusion matrix,
confidence-stratified accuracy, background-rejection-at-S-eff.

---

## Architecture

- Autoresearch loop = Claude agent.
- Runner (`core/runner.py`) executes ONE experiment per call.
- Dashboard reads logs.
- Save checkpoint after every experiment.
- Use relative imports.

---

## Validation Checklist (Run Before Every Experiment Session)

1. `core.evaluation.audit.audit_or_die()` passes.
2. `splits` returns: train ≈ 10 M, val = 500 k, test = 500 k.
3. Train-val overlap = 0, train-test overlap = 0, val-test overlap = 0.
4. Class balance: positive prevalence ∈ [0.40, 0.60] in every split.
5. No NaN/Inf in any feature column.
6. Data loaded from `.data_cache/higgs/` (not re-downloaded).

---

## Project Structure

```
AUTORESEARCHTABULAR/
├── CLAUDE.md
├── README.md
├── PAPER.md
├── MEDIUM.md
├── SOTA_COMPARISON.md
├── AUTORESEARCH_PROCESS.md
├── SETUP.md
├── paper_abstract.md
├── ARCHITECTURE.md
├── pyproject.toml
├── sota_catalog.yaml
├── configs/
│   └── higgs.yaml
├── core/
│   ├── runner.py
│   ├── reasoning.py
│   ├── checkpoint.py
│   ├── winner_archive.py
│   ├── data/
│   │   ├── loader.py
│   │   └── download_higgs.py
│   ├── backbones/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── gbm.py            # lightgbm + xgboost + catboost (3 separate registry entries)
│   │   ├── sklearn.py        # logistic_regression + random_forest
│   │   ├── mlp.py
│   │   ├── mlp_plr.py
│   │   ├── ft_transformer.py
│   │   ├── saint.py
│   │   ├── node.py
│   │   ├── tabnet.py
│   │   ├── tabm.py
│   │   ├── tabpfn_v2.py
│   │   └── excelformer.py
│   └── evaluation/
│       ├── splits.py
│       ├── metrics.py
│       ├── composite.py
│       └── audit.py
├── autoresearch_results/
│   ├── experiment_log.jsonl
│   ├── reasoning_annotations.json
│   ├── data_split_audit.{json,md}
│   ├── third_party_audit.md
│   ├── research_journal.md
│   ├── experiment_summary.md
│   ├── trade_logs/
│   ├── winners/
│   └── dashboard.html
├── memory/
│   ├── project_autoresearch_checkpoint.md
│   └── project_hardware_log.md
├── code_versions/
├── dashboard/dashboard.html
├── docs/                            # GitHub Pages
├── scripts/
└── tests/
```

---

## Key Constants

| Constant | Value | Location |
|---|---|---|
| PRIMARY_METRIC | test_auc | `composite.py` |
| COMPOSITE_FORMULA | `min(test_auc, val_auc) - 0.1 * abs(test_auc - val_auc)` | `composite.py` |
| SPLIT_PROTOCOL | Baldi 2014 frozen split | `loader.py` |
| TRAIN_N | 10,000,000 | first 10M rows |
| VAL_N | 500,000 | next 500k rows |
| TEST_N | 500,000 | last 500k rows |
| N_FEATURES | 28 | Higgs UCI |
| BATCH_SIZE_DEFAULT | 1024 (neural) / N/A (GBM) | runner.py |
| EPOCHS_DEFAULT | 100 (neural) / 1000 iter (GBM) | sota_catalog |
| PATIENCE_DEFAULT | 16 (neural) / 50 (GBM) | sota_catalog |
| GPU_VRAM_GB | 16 | runner.py |
| CPU_RUNNER_AFFINITY | [0, 2, 4, 6] | runner.py |
| BANNED_CORES | [16, 17, 24, 25] | hardware policy |
| N_EXPERIMENTS_PER_BACKBONE | 25 | this file |

---

## Common Mistakes (Never Repeat)

| Mistake | Consequence | Prevention |
|---|---|---|
| Computing standardization stats on val/test | Leakage; inflates val/test AUC | Standardize using train-set mean/std only |
| Using val for early stopping AND HP selection | Test set is no longer a clean held-out | Tune HPs by val AUC; report test AUC once at the end |
| Re-downloading HIGGS each run | 30+ minutes wasted | Default cache dir |
| Grid sweep instead of diagnostic | 10× more experiments than needed | One change per experiment, diagnose first |
| Bundling lightgbm+xgboost+catboost | Tier-3 rule violated | Three separate registry entries |
| Running on a 1M-row subset and reporting as Higgs | Misleading literature comparison | Always train on the full 10M-row train; only the EVAL is on val/test 500k each |
| Forgetting num_threads/n_jobs=4 on GBMs | Slow training (1-thread default) | Always set n_jobs/num_threads=4 |
| BF16 on neural without LayerNorm-FP32 | NaN gradients | LayerNorm/GroupNorm stays FP32 |
| Skipping the per-backbone VRAM pre-flight | OOM at experiment 1 | Runner requires VRAM block in reasoning |
| Treating Higgs as imbalanced | Unnecessary class-weighting | ~53% positive prevalence — no imbalance |
| Reporting accuracy instead of AUC | Hides AUC differences below acc=0.5 boundary | Always report AUC primary, accuracy secondary |
| Rewriting composite formula mid-project | Goodhart violation | Frozen fingerprint; require RULE_CHANGE entry |
| Silently dropping a CLAUDE.md section | Rules drift | All 52 sections preserved |

---

## Session Learnings

_Append-only._

### Initial setup (2026-04-26)

- **Project created:** 2026-04-26
- **Task:** binary classification (signal vs background)
- **Dataset:** Higgs UCI, 11 M rows × 28 features
- **Primary metric:** test_auc
- **Composite:** `min(test_auc, val_auc) - 0.1 * |test_auc - val_auc|`
- **Split protocol:** Baldi 2014 frozen split (10 M / 500 k / 500 k)
- **Backbones in scope (priority order):** lightgbm, xgboost, catboost,
  logistic_regression, random_forest, mlp, mlp_plr, ft_transformer,
  saint, node, tabnet, tabm, tabpfn_v2, excelformer
- **Hardware:** 16 GB VRAM, P-cores 0-15 (E-cores banned)
- **Inherited rules from FX project:** all 52 CLAUDE.md sections preserved.

### Headroom hypothesis (pre-experiment)

Published Higgs AUC numbers (eyeballed from arXiv references):
- Baldi 2014 BDT: 0.802
- Baldi 2014 DNN-3L: 0.876
- XGBoost (various): 0.864
- FT-Transformer (Gorishniy 2021): 0.880
- SAINT (Somepalli 2021): 0.881
- TabM (Gorishniy 2024 ICLR'25): ~0.886
- TabPFN-v2 (Hollmann 2025): not directly evaluated on Higgs (designed
  for ≤ 10 k samples) — N/A here

Expected best result this campaign: composite ≈ 0.880-0.890. Champion
lineage hypothesis:
- Exp1 (LightGBM default) → composite ≈ 0.83
- Exp10 (LightGBM tuned) → composite ≈ 0.85
- Exp25-50 (XGBoost / CatBoost tuned) → composite ≈ 0.86-0.87
- Exp50-100 (MLP-PLR / FT-Transformer / SAINT) → composite ≈ 0.875-0.882
- Exp100+ (TabM) → composite ≈ 0.885-0.890

This is a **prediction**, not a plan. Each step earned by per-split
diagnosis + literature cite + hypothesis + measured outcome.

---

## Cross-references

| Document | Purpose |
|---|---|
| `C:/Users/evija/autoresearch/CLAUDE.md` | The original FX-project CLAUDE.md |
| `C:/Users/evija/autoresearch/generalized_ml_autoresearch/CLAUDE.md` | Meta-framework rules |
| `C:/Users/evija/autoresearch/generalized_ml_autoresearch/templates/CLAUDE_template.md` | Parameterized template this file is filled from |
| `sota_catalog.yaml` | 14 backbones with full citations |
| `tests/test_section_coverage.py` | Audit gate for this CLAUDE.md vs source |

---

## License

MIT.

## Credits

- FX AutoResearch methodology (the source CLAUDE.md) — Evija Ranti.
- Generalized framework — Claude (hierarchical coordinator), 2026-04-19.
- AUTORESEARCHTABULAR (Higgs UCI) — Claude, 2026-04-26.
