"""Smoke tests: data loader, audit, composite fingerprint, reasoning gate.

Run with `pytest tests/test_smoke.py -v`. The data-loader tests require
the Higgs NPZ to already be materialised in `.data_cache/`.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.evaluation.composite import (  # noqa: E402
    composite_fingerprint, compute_composite, CompositeFingerprintError,
)
from core.evaluation.metrics import (  # noqa: E402
    safe_auc, expected_calibration_error, background_rejection_at_signal_eff,
    compute_split_metrics,
)
from core.reasoning import (  # noqa: E402
    ReasoningEntry, validate_reasoning_blob, validate_citation_rigor,
)


def test_composite_formula_default_fingerprint():
    fp = composite_fingerprint()
    assert isinstance(fp, str) and len(fp) == 16
    fp2 = composite_fingerprint("min(test_auc, val_auc) - 0.1 * abs(test_auc - val_auc)")
    assert fp == fp2


def test_compute_composite_simple():
    out = compute_composite(test_auc=0.85, val_auc=0.82)
    assert pytest.approx(out["composite"], rel=1e-6) == 0.82 - 0.1 * 0.03
    assert isinstance(out["composite_fingerprint"], str)


def test_composite_fingerprint_mismatch_raises(tmp_path):
    fp_path = tmp_path / "fp.json"
    out1 = compute_composite(0.9, 0.85, fingerprint_path=str(fp_path))
    assert out1["composite"] > 0
    with pytest.raises(CompositeFingerprintError):
        compute_composite(0.9, 0.85,
                          formula="(test_auc + val_auc) / 2",  # changed!
                          fingerprint_path=str(fp_path))


def test_safe_auc_handles_single_class():
    y = np.array([1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.isnan(safe_auc(y, p))


def test_compute_split_metrics_shape():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=1000)
    p = rng.uniform(0, 1, size=1000)
    m = compute_split_metrics(y, p)
    for k in ("auc", "auprc", "ece", "accuracy", "log_loss", "f1",
              "br_at_50sig", "br_at_70sig", "cm_tp", "cm_fp"):
        assert k in m


def test_background_rejection_finite():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=10000)
    p = rng.uniform(0, 1, size=10000)
    br = background_rejection_at_signal_eff(y, p, signal_eff=0.7)
    assert br > 0


def test_reasoning_blob_rejects_short_diagnosis():
    e = ReasoningEntry(
        diagnosis="too short",
        citations=("Ke, Meng, Finley, Wang 2017 NeurIPS 'LightGBM' "
                   "(arXiv:1711.08251) — gradient boosting; reference for "
                   "the lightgbm baseline used on Higgs UCI."),
        hypothesis=("This is a long enough hypothesis that should pass the "
                    "word floor of fifty words. We hypothesise that the "
                    "default LightGBM configuration provides a competitive "
                    "starting baseline on the Higgs UCI Baldi 2014 frozen "
                    "split, with expected validation AUROC near 0.83 and "
                    "negligible val/test gap."),
        prediction=("Predict val AUROC 0.83 ± 0.01 and test AUROC 0.83 ± "
                    "0.01 with val/test gap below 0.005 since Higgs is "
                    "i.i.d."),
    )
    issues = validate_reasoning_blob(e, post_run=False)
    assert any("diagnosis" in i for i in issues)


def test_reasoning_blob_passes_when_complete():
    e = ReasoningEntry(
        diagnosis=" ".join(["This"] * 70),
        citations=(
            "Ke, Meng, Finley, Wang 2017 NeurIPS 'LightGBM: A Highly "
            "Efficient Gradient Boosting Decision Tree' (arXiv:1711.08251) "
            "— canonical leaf-wise GBM reference: provides the published "
            "baseline configuration used on the Higgs UCI benchmark and "
            "establishes the histogram + GOSS algorithm with default "
            "hyperparameters num_leaves=31 and learning_rate=0.1."
        ),
        hypothesis=" ".join(["Hypothesis"] * 60),
        prediction=" ".join(["Predict"] * 30),
    )
    issues = validate_reasoning_blob(e, post_run=False)
    assert not issues, f"unexpected: {issues}"


def test_citation_rigor_rejects_bare_url():
    issues = validate_citation_rigor("see https://example.com")
    assert issues  # multiple problems


def test_data_loader_materialised_or_skip():
    """If the NPZ exists, sanity-check shapes + class balance.
    Otherwise skip cleanly."""
    cache = ROOT / ".data_cache" / "higgs" / "higgs.npz"
    if not cache.exists():
        pytest.skip("higgs.npz not materialised; skipping loader smoke")
    from core.data import load_higgs_split
    splits = load_higgs_split(data_cache=str(ROOT / ".data_cache"),
                               subset_train_n=10_000,
                               standardize=False)
    Xtr, ytr, _ = splits["train"]
    Xv, yv, _ = splits["val"]
    Xt, yt, _ = splits["test"]
    assert Xtr.shape[1] == 28
    assert Xv.shape == (500_000, 28)
    assert Xt.shape == (500_000, 28)
    for y in (ytr, yv, yt):
        assert set(np.unique(y).tolist()) == {0, 1}


def test_audit_artifacts_present_or_skip():
    """If results_dir/data_split_audit.json exists, validate structure."""
    p = ROOT / "autoresearch_results" / "data_split_audit.json"
    if not p.exists():
        pytest.skip("audit not yet run")
    with open(p) as f:
        payload = json.load(f)
    assert payload["overall_status"] in ("PASS", "FAIL")
    assert payload["data_mode"] == "higgs"
    assert "split_fingerprint" in payload
    assert len(payload["auditors"]) == 7
