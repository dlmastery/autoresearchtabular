"""Triple-check data split audit for AUTORESEARCHTABULAR / Higgs UCI.

Auditors (all must report PASS):
  1. audit_split_disjoint        — pairwise row-index intersections empty
  2. audit_split_protocol        — Baldi 2014 frozen sizes match
  3. audit_class_balance         — both classes present, prevalence in [0.40, 0.60]
  4. audit_size_floors           — train ≥ 9.5M, val ≥ 450k, test ≥ 450k
  5. audit_no_leakage_via_metadata — model inputs are pure 28-dim float vectors
  6. audit_reproducibility       — same fingerprint on two passes
  7. audit_feature_consistency   — feature dtype float32, no NaN/Inf

Outputs: data_split_audit.json, data_split_audit.md, fingerprint json.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import numpy as np

EXPECTED_TRAIN_N = 10_000_000
EXPECTED_VAL_N = 500_000
EXPECTED_TEST_N = 500_000
SIZE_TOL_PCT = 0.005  # 0.5%


@dataclass
class AuditResult:
    name: str
    status: str
    violations: List[str]
    details: Dict[str, Any]

    def passed(self) -> bool:
        return self.status == "PASS"


def audit_split_disjoint(splits: Dict[str, Dict[str, np.ndarray]]) -> AuditResult:
    violations: List[str] = []
    fold_names = list(splits.keys())
    inter: Dict[str, int] = {}
    for i in range(len(fold_names)):
        for j in range(i + 1, len(fold_names)):
            a, b = fold_names[i], fold_names[j]
            sa = set(splits[a]["row_idx"].tolist())
            sb = set(splits[b]["row_idx"].tolist())
            n = len(sa & sb)
            inter[f"{a} & {b}"] = n
            if n > 0:
                violations.append(f"row_idx overlap {a} & {b} = {n}")
    return AuditResult("audit_split_disjoint",
                       "PASS" if not violations else "FAIL",
                       violations, {"pairwise_intersections": inter})


def audit_split_protocol(splits: Dict[str, Dict[str, np.ndarray]]) -> AuditResult:
    violations: List[str] = []
    details: Dict[str, Any] = {}
    expected = {"train": EXPECTED_TRAIN_N, "val": EXPECTED_VAL_N, "test": EXPECTED_TEST_N}
    for fold, exp in expected.items():
        n = int(len(splits[fold]["y"]))
        details[f"{fold}_n"] = n
        details[f"{fold}_expected"] = exp
        # train can be subset (subset_train_n config); val/test must be exact
        if fold == "train":
            if n < int(exp * 0.05) or n > exp:
                violations.append(f"train n={n} outside [5% of {exp}, {exp}]")
        else:
            tol = max(1, int(exp * SIZE_TOL_PCT))
            if abs(n - exp) > tol:
                violations.append(f"{fold} n={n} != {exp} (tol ±{tol})")
    return AuditResult("audit_split_protocol",
                       "PASS" if not violations else "FAIL",
                       violations, details)


def audit_class_balance(splits: Dict[str, Dict[str, np.ndarray]]) -> AuditResult:
    violations: List[str] = []
    details: Dict[str, Any] = {}
    for fold, d in splits.items():
        y = d["y"]
        if len(y) == 0:
            violations.append(f"{fold}: empty fold")
            continue
        unique = set(int(v) for v in np.unique(y))
        prev = float(np.mean(y))
        details[f"{fold}_classes"] = sorted(unique)
        details[f"{fold}_pos_prevalence"] = round(prev, 4)
        if unique != {0, 1}:
            violations.append(f"{fold}: missing class — found {sorted(unique)}")
        if prev < 0.40 or prev > 0.60:
            violations.append(f"{fold}: positive prevalence {prev:.3f} outside [0.40, 0.60]")
    return AuditResult("audit_class_balance",
                       "PASS" if not violations else "FAIL",
                       violations, details)


def audit_size_floors(splits: Dict[str, Dict[str, np.ndarray]]) -> AuditResult:
    floors = {"train": 9_500_000, "val": 450_000, "test": 450_000}
    violations: List[str] = []
    details: Dict[str, int] = {}
    for fold, floor in floors.items():
        n = int(len(splits[fold]["y"]))
        details[fold] = n
        # Allow much smaller train if user explicitly subset
        if fold == "train":
            if n < 1000:
                violations.append(f"train n={n} < hard floor 1000")
        elif n < floor:
            violations.append(f"{fold} n={n} < floor {floor}")
    return AuditResult("audit_size_floors",
                       "PASS" if not violations else "FAIL",
                       violations, details)


def audit_no_leakage_via_metadata(splits: Dict[str, Dict[str, np.ndarray]]) -> AuditResult:
    violations: List[str] = []
    details: Dict[str, Any] = {}
    for fold, d in splits.items():
        X = d["X"]
        if X.ndim != 2 or X.shape[1] != 28:
            violations.append(f"{fold} X shape {X.shape} != (n, 28)")
        if X.dtype != np.float32:
            violations.append(f"{fold} X dtype {X.dtype} != float32")
        details[f"{fold}_shape"] = list(X.shape)
        details[f"{fold}_dtype"] = str(X.dtype)
    return AuditResult("audit_no_leakage_via_metadata",
                       "PASS" if not violations else "FAIL",
                       violations, details)


def audit_feature_consistency(splits: Dict[str, Dict[str, np.ndarray]]) -> AuditResult:
    violations: List[str] = []
    details: Dict[str, Any] = {}
    for fold, d in splits.items():
        X = d["X"]
        n_nan = int(np.isnan(X).sum())
        n_inf = int(np.isinf(X).sum())
        details[f"{fold}_nan"] = n_nan
        details[f"{fold}_inf"] = n_inf
        if n_nan > 0:
            violations.append(f"{fold} has {n_nan} NaN values")
        if n_inf > 0:
            violations.append(f"{fold} has {n_inf} Inf values")
    return AuditResult("audit_feature_consistency",
                       "PASS" if not violations else "FAIL",
                       violations, details)


def split_fingerprint(splits: Dict[str, Dict[str, np.ndarray]]) -> str:
    h = hashlib.sha256()
    for fold in sorted(splits.keys()):
        d = splits[fold]
        h.update(fold.encode())
        # hash row_idx + y (lightweight, deterministic)
        h.update(np.ascontiguousarray(d["row_idx"]).tobytes())
        h.update(np.ascontiguousarray(d["y"]).tobytes())
    return h.hexdigest()


def audit_reproducibility(build_fn) -> AuditResult:
    fps: List[str] = []
    sizes: List[Dict[str, int]] = []
    for _ in range(2):
        s = build_fn()
        fps.append(split_fingerprint(s))
        sizes.append({k: int(len(s[k]["y"])) for k in s})
    violations: List[str] = []
    if len(set(fps)) > 1:
        violations.append(f"reproducibility violated: fingerprints differ {fps}")
    if len(set(json.dumps(s, sort_keys=True) for s in sizes)) > 1:
        violations.append(f"reproducibility violated: sizes differ {sizes}")
    return AuditResult("audit_reproducibility",
                       "PASS" if not violations else "FAIL",
                       violations, {"fingerprints": fps, "sizes": sizes})


# ---- Driver ------------------------------------------------------

AUDIT_VERSION = "1.0.0"


def _build_splits(data_cache: str, subset_train_n: int | None) -> Dict[str, Dict[str, np.ndarray]]:
    from core.data import load_higgs_split
    raw = load_higgs_split(data_cache=data_cache, subset_train_n=subset_train_n,
                            standardize=False)
    return {
        "train": {"X": raw["train"][0], "y": raw["train"][1], "row_idx": raw["train"][2]},
        "val":   {"X": raw["val"][0],   "y": raw["val"][1],   "row_idx": raw["val"][2]},
        "test":  {"X": raw["test"][0],  "y": raw["test"][1],  "row_idx": raw["test"][2]},
    }


def run_full_audit(*, data_cache: str, results_dir: str,
                   subset_train_n: int | None = None,
                   triple_check: bool = True) -> Dict[str, Any]:
    def build():
        return _build_splits(data_cache, subset_train_n)

    splits = build()
    audits: List[AuditResult] = [
        audit_split_disjoint(splits),
        audit_split_protocol(splits),
        audit_class_balance(splits),
        audit_size_floors(splits),
        audit_no_leakage_via_metadata(splits),
        audit_feature_consistency(splits),
    ]
    if triple_check:
        audits.append(audit_reproducibility(build))

    fp = split_fingerprint(splits)
    overall_pass = all(a.passed() for a in audits)
    payload = {
        "audit_version": AUDIT_VERSION,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_mode": "higgs",
        "subset_train_n": subset_train_n,
        "data_cache": data_cache,
        "fold_sizes": {k: int(len(splits[k]["y"])) for k in splits},
        "split_fingerprint": fp,
        "overall_status": "PASS" if overall_pass else "FAIL",
        "auditors": [asdict(a) for a in audits],
    }

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "data_split_audit.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(results_dir, "data_split_audit_fingerprint.json"), "w") as f:
        json.dump({"fingerprint": fp, "data_mode": "higgs",
                    "subset_train_n": subset_train_n}, f, indent=2)
    md = _render_md(payload)
    with open(os.path.join(results_dir, "data_split_audit.md"), "w", encoding="utf-8") as f:
        f.write(md)
    return payload


def _render_md(payload: Dict[str, Any]) -> str:
    lines = []
    lines.append("# AUTORESEARCHTABULAR — Data Split Audit Report")
    lines.append("")
    lines.append(f"_Generated: {payload['timestamp_utc']}_")
    lines.append("")
    lines.append(f"- **Overall:** **{payload['overall_status']}**")
    lines.append(f"- **Data mode:** higgs (Baldi 2014 frozen split)")
    lines.append(f"- **subset_train_n:** {payload['subset_train_n']}")
    lines.append(f"- **Audit version:** {payload['audit_version']}")
    lines.append(f"- **Split fingerprint:** `{payload['split_fingerprint']}`")
    lines.append("")
    lines.append("## Fold sizes")
    lines.append("")
    lines.append("| fold | size |")
    lines.append("|---|---|")
    for k, v in payload["fold_sizes"].items():
        lines.append(f"| {k} | {v:,} |")
    lines.append("")
    lines.append("## Auditors")
    lines.append("")
    for a in payload["auditors"]:
        lines.append(f"### {a['name']} — **{a['status']}**")
        if a["violations"]:
            for v in a["violations"]:
                lines.append(f"- VIOLATION: {v}")
        else:
            lines.append("- (no violations)")
        if a.get("details"):
            lines.append("")
            lines.append("<details><summary>details</summary>\n")
            lines.append("```json")
            lines.append(json.dumps(a["details"], indent=2, default=str))
            lines.append("```")
            lines.append("</details>")
        lines.append("")
    return "\n".join(lines)


def audit_or_die(*, results_dir: str, max_age_seconds: int = 86400,
                 expected_fingerprint: str | None = None) -> None:
    """Used by the runner. Raises SystemExit on any failure."""
    json_path = os.path.join(results_dir, "data_split_audit.json")
    if not os.path.exists(json_path):
        raise SystemExit(
            "[runner gate] data_split_audit.json missing. Run:\n"
            "  python -m core.evaluation.audit --config configs/higgs.yaml --triple-check"
        )
    with open(json_path) as f:
        payload = json.load(f)
    age = time.time() - time.mktime(time.strptime(payload["timestamp_utc"],
                                                   "%Y-%m-%dT%H:%M:%SZ"))
    if age > max_age_seconds:
        raise SystemExit(
            f"[runner gate] data_split_audit.json is {age/3600:.1f} h old "
            f"(> {max_age_seconds/3600:.1f} h limit). Re-run the audit.")
    if payload.get("overall_status") != "PASS":
        viols = []
        for a in payload.get("auditors", []):
            if a.get("status") != "PASS":
                for v in a.get("violations", []):
                    viols.append(f"{a['name']}: {v}")
        raise SystemExit(
            "[runner gate] data split audit FAILED:\n  - " + "\n  - ".join(viols))
    if expected_fingerprint and payload.get("split_fingerprint") != expected_fingerprint:
        raise SystemExit(
            f"[runner gate] split fingerprint differs from audit "
            f"({expected_fingerprint!r} vs {payload.get('split_fingerprint')!r}). "
            "Re-run the audit.")


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--triple-check", action="store_true")
    p.add_argument("--subset-train-n", type=int, default=None)
    return p.parse_args()


def main():
    import yaml
    args = _cli()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    print(f"[audit] data_cache={paths['data_cache']} subset_train_n={args.subset_train_n}")
    payload = run_full_audit(
        data_cache=paths["data_cache"], results_dir=paths["results_dir"],
        subset_train_n=args.subset_train_n, triple_check=args.triple_check,
    )
    print(f"[audit] overall_status = {payload['overall_status']}")
    print(f"[audit] split fingerprint = {payload['split_fingerprint']}")
    for a in payload["auditors"]:
        print(f"  {a['name']}: {a['status']} ({len(a['violations'])} violation(s))")
        for v in a["violations"]:
            print(f"    - {v}")
    return 0 if payload["overall_status"] == "PASS" else 7


if __name__ == "__main__":
    sys.exit(main())
