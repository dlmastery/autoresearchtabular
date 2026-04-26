"""Higgs UCI data loader with frozen Baldi 2014 split.

The dataset:
  - 11,000,000 rows × 28 features (21 low-level + 7 high-level)
  - First column is the binary label (1 = signal, 0 = background)
  - Source: https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
  - Reference: Baldi, Sadowski & Whiteson 2014 Nature Communications
    'Searching for Exotic Particles in High-Energy Physics with Deep
    Learning' arXiv:1402.4735

The Baldi 2014 frozen split (every paper since uses this):
  - Train: rows [0, 10,000,000)        → 10.0 M rows
  - Val:   rows [10,000,000, 10,500,000)  → 500 k rows
  - Test:  rows [10,500,000, 11,000,000)  → 500 k rows
"""
from __future__ import annotations
import gzip
import os
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
N_TOTAL = 11_000_000
N_TRAIN = 10_000_000
N_VAL = 500_000
N_TEST = 500_000  # last 500k

LOW_LEVEL_FEATURES = [
    "lepton_pT", "lepton_eta", "lepton_phi",
    "missing_energy_magnitude", "missing_energy_phi",
    "jet_1_pt", "jet_1_eta", "jet_1_phi", "jet_1_b-tag",
    "jet_2_pt", "jet_2_eta", "jet_2_phi", "jet_2_b-tag",
    "jet_3_pt", "jet_3_eta", "jet_3_phi", "jet_3_b-tag",
    "jet_4_pt", "jet_4_eta", "jet_4_phi", "jet_4_b-tag",
]
HIGH_LEVEL_FEATURES = [
    "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb",
]
FEATURE_NAMES = LOW_LEVEL_FEATURES + HIGH_LEVEL_FEATURES
SPLIT_NAMES = ("train", "val", "test")


def _resolve_cache(data_cache: str) -> Path:
    p = Path(data_cache) / "higgs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _download(cache_dir: Path) -> Path:
    csv_gz = cache_dir / "HIGGS.csv.gz"
    if csv_gz.exists() and csv_gz.stat().st_size > 2_500_000_000:
        return csv_gz
    print(f"[higgs] downloading {UCI_URL} -> {csv_gz} (this is ~2.8 GB)...")
    urllib.request.urlretrieve(UCI_URL, csv_gz)
    print(f"[higgs] download OK ({csv_gz.stat().st_size / 1e9:.2f} GB)")
    return csv_gz


def _materialize_npz(csv_gz: Path, npz_path: Path) -> None:
    print(f"[higgs] materializing CSV -> NPZ at {npz_path} (one-time, ~3-5 min)...")
    X = np.empty((N_TOTAL, 28), dtype=np.float32)
    y = np.empty((N_TOTAL,), dtype=np.int8)
    with gzip.open(csv_gz, "rt") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split(",")
            if len(parts) != 29:
                raise ValueError(f"row {i}: got {len(parts)} fields, expected 29")
            y[i] = int(float(parts[0]))
            X[i] = [float(p) for p in parts[1:]]
            if i % 1_000_000 == 0:
                print(f"  parsed {i:>10,d} / {N_TOTAL:,}", flush=True)
    print(f"  parsed {N_TOTAL:,} rows total")
    np.savez_compressed(npz_path, X=X, y=y)
    print(f"[higgs] wrote {npz_path} ({npz_path.stat().st_size / 1e9:.2f} GB)")


def _load_arrays(data_cache: str) -> Tuple[np.ndarray, np.ndarray]:
    cache_dir = _resolve_cache(data_cache)
    npz = cache_dir / "higgs.npz"
    if not npz.exists():
        csv_gz = _download(cache_dir)
        _materialize_npz(csv_gz, npz)
    z = np.load(npz)
    X = z["X"]
    y = z["y"].astype(np.int64)
    if X.shape != (N_TOTAL, 28) or y.shape != (N_TOTAL,):
        raise RuntimeError(f"unexpected shapes: X={X.shape} y={y.shape}")
    return X, y


def load_higgs_split(data_cache: str = "./.data_cache",
                     subset_train_n: Optional[int] = None,
                     standardize: bool = False
                     ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load the Baldi 2014 frozen Higgs split.

    Returns dict keyed by SPLIT_NAMES. Each value is (X, y, row_index).
    Row indices are 0-based positions in the original CSV (for the audit
    to verify pairwise disjointness).

    If `subset_train_n` is set, the train fold is restricted to the first
    `subset_train_n` rows of the train range (deterministic, contiguous).
    Val/test are ALWAYS the full 500 k each.

    If `standardize` is True, mean/std are computed on the full train
    set and applied to all three splits. (Stats from train only — no
    leakage.)
    """
    X, y = _load_arrays(data_cache)
    train_end = N_TRAIN
    val_end = N_TRAIN + N_VAL  # 10_500_000
    test_end = val_end + N_TEST  # 11_000_000

    train_idx = np.arange(0, train_end, dtype=np.int64)
    val_idx = np.arange(train_end, val_end, dtype=np.int64)
    test_idx = np.arange(val_end, test_end, dtype=np.int64)

    if subset_train_n is not None and subset_train_n < N_TRAIN:
        train_idx = train_idx[:subset_train_n]

    out = {
        "train": (X[train_idx], y[train_idx], train_idx),
        "val":   (X[val_idx],   y[val_idx],   val_idx),
        "test":  (X[test_idx],  y[test_idx],  test_idx),
    }

    if standardize:
        mean = out["train"][0].mean(axis=0).astype(np.float32)
        std = (out["train"][0].std(axis=0) + 1e-7).astype(np.float32)
        for k in SPLIT_NAMES:
            Xk, yk, ik = out[k]
            out[k] = ((Xk - mean) / std, yk, ik)
        out["_standardize"] = {"mean": mean, "std": std}  # type: ignore[assignment]
    return out
