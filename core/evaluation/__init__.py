"""Evaluation module."""
from .metrics import compute_split_metrics, safe_auc, safe_auprc, expected_calibration_error
from .composite import compute_composite, composite_fingerprint, CompositeFingerprintError
from .audit import audit_or_die, run_full_audit
