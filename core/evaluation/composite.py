"""Composite metric with Goodhart-protection fingerprint."""
from __future__ import annotations
import hashlib
import json
import math
import os
from typing import Dict


_DEFAULT_FORMULA = "min(test_auc, val_auc) - 0.1 * abs(test_auc - val_auc)"


def composite_fingerprint(formula: str = _DEFAULT_FORMULA) -> str:
    return hashlib.sha256(formula.strip().encode("utf-8")).hexdigest()[:16]


class CompositeFingerprintError(RuntimeError):
    pass


def _safe_eval(formula: str, env: Dict[str, float]) -> float:
    allowed = {"min": min, "max": max, "abs": abs, "math": math}
    allowed.update(env)
    code = compile(formula, "<composite>", "eval")
    for name in code.co_names:
        if name not in allowed:
            raise CompositeFingerprintError(f"composite formula uses disallowed name: {name!r}")
    return float(eval(code, {"__builtins__": {}}, allowed))  # noqa: S307


def compute_composite(test_auc: float, val_auc: float,
                      formula: str = _DEFAULT_FORMULA,
                      fingerprint_path: str | None = None) -> Dict[str, float | str]:
    fp = composite_fingerprint(formula)
    if fingerprint_path is not None:
        if os.path.exists(fingerprint_path):
            try:
                with open(fingerprint_path) as f:
                    saved = json.load(f)
            except Exception:
                saved = {}
            if saved.get("fingerprint") and saved.get("fingerprint") != fp:
                raise CompositeFingerprintError(
                    f"Composite formula fingerprint mismatch — old={saved['fingerprint']!r} "
                    f"new={fp!r}. Mid-project rewrites of the composite are forbidden.")
        else:
            os.makedirs(os.path.dirname(fingerprint_path) or ".", exist_ok=True)
            with open(fingerprint_path, "w") as f:
                json.dump({"fingerprint": fp, "formula": formula}, f, indent=2)
    env = {"test_auc": float(test_auc), "val_auc": float(val_auc)}
    score = _safe_eval(formula, env)
    return {"composite": score, "composite_formula": formula, "composite_fingerprint": fp}
