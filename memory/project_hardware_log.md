# Hardware log — AUTORESEARCHTABULAR

## Workstation contract

| Component       | Value                                            |
|-----------------|--------------------------------------------------|
| OS              | Windows 11 Home, version 10.0.26200              |
| CPU             | Intel hybrid (P-core + E-core architecture)      |
| Pinned cores    | P-cores 0-15 (E-cores 16/17/24/25 banned)        |
| GPU             | RTX 4060 Ti class, 16 GB VRAM cap                |
| Python (training) | `C:\Users\evija\anaconda3\python.exe` (3.12.3) |
| Python (system) | Windows Store Python 3.13 (NOT used for training) |
| LightGBM/XGBoost/CatBoost | conda-forge (CPU)                       |
| PyTorch         | CUDA-enabled, BF16 autocast on GPU               |

The runner pins to P-cores via `psutil.Process.cpu_affinity([0, 2, 4, 6])`
and forces `torch.set_num_threads(4)`. The `cpu_runner_affinity` and
`banned_cores` lists in `configs/higgs.yaml` are the source of truth.

## Training-time observations

- LightGBM at `subset_train_n=1_000_000` with default 1000 trees:
  ~95-100 s training time + ~80 s for predict + per-row CSV write.
- Per-experiment total wall time: ~3 min for GBM family on 1M training.
- 16 GB VRAM is comfortable for FT-Transformer at batch 256 on 1M rows;
  full 10M training would require gradient accumulation.
- BF16 autocast on RTX 4060 Ti gives ~1.4× speedup vs FP32 on the MLP.

## Determinism

- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `random.seed(seed)`, `np.random.seed(seed)`, `torch.manual_seed(seed)`
- LightGBM/XGBoost/CatBoost respect their `seed` / `random_state` /
  `random_seed` parameters; results reproduce exactly across runs given
  the same overrides.

## Storage

- `.data_cache/higgs/HIGGS.csv.gz` — 2.62 GB (UCI source)
- `.data_cache/higgs/higgs.npz` — 0.95 GB (materialised, float32 X +
  int8 y)
- Per-experiment predictions CSV — ~70 MB per run (gitignored)
- `autoresearch_results/experiment_log.jsonl` — append-only metrics log
- `autoresearch_results/reasoning_annotations.json` — per-experiment
  reasoning blobs (committed)

## Known issues

- Windows Store Python 3.13 does not have lightgbm/xgboost/catboost
  installed — the campaign script prefers `C:\Users\evija\anaconda3\
  python.exe` automatically.
- Predictions CSV at 70 MB per run x 150 runs = 10.5 GB — gitignored
  via `autoresearch_results/trade_logs/*_predictions.csv`. The
  prediction summary JSON is committed.
- `runpy` warning when importing `core.evaluation.audit` via `-m`
  flag; cosmetic only.
