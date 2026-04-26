# AUTORESEARCH_PROCESS — How the audit-gated loop runs

This is the operational manual. It explains *exactly* how a single
experiment runs from "I have an idea" to "the row is on the
leaderboard," and what every gate does in code.

## The 7-step protocol

Every experiment follows the same 7 steps in order:

```
diagnose → cite → hypothesize → predict → run-ONE → analyze → checkpoint
```

The first four steps happen **before** training. Steps 5 and 6 are the
training run itself + post-run analysis. Step 7 is the persistence step
where the experiment row is appended to the leaderboard. The runner
refuses transitions in any other order.

### 1. `diagnose`

State, in writing, what is wrong or unknown about all prior
experiments that this experiment will address. **Word floor: 25.**
This forces the researcher (or LLM) to look at the actual prior
results table before proposing a change.

Example:
> Experiments 1-4 (lightgbm with default num_leaves=31) all converge
> to val AUROC ≈ 0.78 with negligible val/test gap. The val curve
> flattens within 50 iterations, suggesting capacity is the
> bottleneck — the model has converged on what it can learn with this
> tree shape, not on what's learnable. Need to increase
> capacity-per-tree before increasing tree count.

### 2. `cite` (Citation Rigor gate)

Cite a primary source for *every* hyperparameter change vs. the
backbone's registry default. Format:

```
{hp_name} = {value} per {Author Year} §{section} ("{title}", {venue}) —
{reason this should help on Higgs}
```

The parser requires:
- author + year, format-matched against `re.match(r"[A-Z][a-z]+ \d{4}")`
- title or venue, at least one
- section/figure/table reference
- a *reason* sentence containing "Higgs" or naming a feature/property

Bare URLs are rejected. Folklore ("LightGBM likes 256 leaves on big
data") is rejected. **Word floor: 30.**

### 3. `hypothesize`

State the causal mechanism: what the change should do, *why*, and on
which feature of the data. **Word floor: 30.**

> Increasing `num_leaves` from 31 to 256 (paper-cited Ke 2017 §3.2)
> allows leaf-wise growth to capture deeper feature interactions.
> Higgs has 7 high-level engineered features (m_jj, m_bb, m_wbb,
> ...) that are explicit nonlinear combinations of the 21 low-level
> kinematics. The default 31-leaf tree cannot route on more than
> ~4 features per path; 256 leaves enables routing on ~8 features
> per path, which is the rough dimensionality of the
> high-level-feature manifold.

### 4. `predict`

Make a quantitative prediction with sign + magnitude. **Word floor: 25.**

> Expected delta vs Exp 4 baseline (val AUROC 0.781): +0.012 to +0.020
> with std ≈ 0.003 across seeds. test AUROC delta should mirror val
> within 0.003. If delta < 0.005 the hypothesis is wrong (capacity
> wasn't the bottleneck); if delta > 0.025 something else has changed
> (likely overfit, check val/test gap).

The `predict` block is the falsifier. After the run, if the actual
result is far from this prediction, the `analyze` block has to explain
why — that is the audit trail for "what did we learn."

### 5. `run-ONE`

Run *exactly one* experiment. Not three, not a sweep — one. This
forces sequential thinking. The recipe id and diff vs. predecessor are
recorded.

### 6. `analyze`

Post-run. **Word floor: 50.** The full analysis: actual vs predicted,
which prediction was right, which was wrong, what the residual means,
what to do next.

### 7. `checkpoint`

State what this experiment *adds* to the campaign — i.e., the next
question that's now in scope. **Word floor: 15.** This becomes the
seed for the next experiment's `diagnose` block.

## How the gates are encoded

```python
# core/reasoning.py
WORD_FLOORS = {
    "diagnose": 25,
    "cite": 30,
    "hypothesize": 30,
    "predict": 25,
    "run": 15,
    "analyze": 50,
    "checkpoint": 15,
}

def validate_reasoning_blob(blob: dict, *, post_run: bool) -> None:
    sections = ["diagnose", "cite", "hypothesize", "predict", "run"]
    if post_run:
        sections += ["analyze", "checkpoint"]
    for s in sections:
        if s not in blob:
            raise ReasoningGateError(f"missing section '{s}'")
        wc = len(blob[s].split())
        if wc < WORD_FLOORS[s]:
            raise ReasoningGateError(
                f"section '{s}' has {wc} words; floor is "
                f"{WORD_FLOORS[s]}"
            )
    if "cite" in sections:
        validate_citation_rigor(blob["cite"])
```

```python
def validate_citation_rigor(text: str) -> None:
    if not re.search(r"[A-Z][a-z]+\s+\d{4}", text):
        raise ReasoningGateError("citation missing author + year")
    if not re.search(r"§\s*\d", text) and \
       not re.search(r"Tab\.?\s*\d", text) and \
       not re.search(r"Fig\.?\s*\d", text):
        raise ReasoningGateError(
            "citation missing section/table/figure reference"
        )
    if "Higgs" not in text and not re.search(
        r"\b(low-level|high-level|jet|lepton|signal|background|"
        r"missing energy|m_jj|m_bb|m_wbb)\b", text
    ):
        raise ReasoningGateError(
            "citation missing reason naming Higgs or a Higgs feature"
        )
```

## Per-experiment workflow

```
researcher (LLM or human) authors reasoning_blob.json (sections 1-5)
         │
         ▼
  runner.commit_pre_run()
         │
         ├─► validate_reasoning_blob(blob, post_run=False)
         │     - all 5 pre-run sections present
         │     - word floors met
         │     - validate_citation_rigor()
         │
         ├─► audit_or_die(config)  ← Gate 1
         │     - 7 auditors run; first failure raises
         │     - data_split_fingerprint.txt verified to match prior
         │
         ├─► composite formula fingerprint verified
         │
         ▼
  backbone.fit(X_train, y_train)
         │
         ▼
  predictions on val + test
         │
         ▼
  metrics computed: AUROC, AUPRC, ECE, bg-rej-at-sig-eff
         │
         ▼
  composite = min(test, val) - 0.1 * |test - val|
         │
         ▼
  researcher writes analyze + checkpoint sections
         │
         ▼
  runner.commit_post_run()
         │
         ├─► validate_reasoning_blob(blob, post_run=True)
         │     - all 7 sections present
         │     - all word floors met
         │
         ├─► append row to all_runs.csv
         ├─► write per-experiment artefacts
         ├─► refresh dashboard data
         └─► save reasoning_blob.json (final)
```

## The data-split audit (Gate 1)

`core/evaluation/audit.py` runs seven auditors:

1. **`audit_split_disjoint`** — `set(train_idx) & set(val_idx) ==
   set()`, similar for val/test, train/test. Failures here are
   immediately fatal — there is no recovery path.
2. **`audit_split_protocol`** — verifies the contiguous Baldi 2014
   protocol: `train = [0, 10M)`, `val = [10M, 10.5M)`,
   `test = [10.5M, 11M)`.
3. **`audit_class_balance`** — y∈{0,1}, both classes present, signal
   fraction ∈ [0.45, 0.60] (Higgs is ~0.529 signal-fraction by
   design).
4. **`audit_size_floors`** — train ≥ 1M, val ≥ 100k, test ≥ 100k.
5. **`audit_no_leakage_via_metadata`** — verifies there is no
   `event_id` column or row-position-derived feature.
6. **`audit_reproducibility`** — re-loads the NPZ, checks
   `sha256(X.tobytes()) + sha256(y.tobytes())` matches the previous
   run's recorded digest.
7. **`audit_feature_consistency`** — all splits have 28 columns in
   the documented `LOW_LEVEL_FEATURES + HIGH_LEVEL_FEATURES` order.

Output:
- `data_split_audit.json` — machine-readable audit report
- `data_split_audit.md` — human-readable audit report
- `data_split_fingerprint.txt` — single-line SHA-256 of the row-index
  union; embedded in every result row

## What "fail-fast" actually means

The runner has a single `audit_or_die(config)` call near the top of
`main()`. Any auditor that fails raises `AuditFailure`, which is *not*
caught. The process exits with non-zero. The `all_runs.csv` is
untouched. There is no "warning, continuing anyway" mode. This is
the design.

## Recovering from a failed run

When an experiment fails any gate:

1. The runner exits non-zero
2. `all_runs.csv` is unchanged
3. The half-written `recipe.yaml` and `reasoning_blob.json` (pre-run
   only) remain on disk, prefixed with the experiment number
4. The researcher fixes the gate failure (typically: improve the
   reasoning blob to meet word floor or fix a citation)
5. The experiment is re-run

There is no "force-pass" mode. There is no `--skip-audit` flag.
This is also by design.

## Champion + winner archive

When an experiment's composite > current best:

1. The runner writes `winners/exp_{N}_{backbone}/` with the full
   recipe + reasoning + checkpoint
2. `latest_best.json` is updated atomically
3. The dashboard's "champion" card shows the new winner

When the campaign completes, the per-backbone winners are reranked,
the cross-backbone champion is identified, and a 3-seed rerun is
performed for the cross-backbone champion to record mean ± std on test.

## Hardware contract

- P-cores 0-15 only (E-cores 16/17/24/25 banned via
  `psutil.Process.cpu_affinity`)
- 16 GB VRAM cap; mixed precision BF16 default
- `torch.manual_seed`, `np.random.seed`, `random.seed` fixed per
  experiment
- `cudnn.deterministic=True`, `cudnn.benchmark=False`

The hardware contract is recorded in
`memory/project_hardware_log.md` and re-checked at runner boot.
