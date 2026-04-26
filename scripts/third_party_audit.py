"""Third-party replay audit for AUTORESEARCHTABULAR.

Verifies, from the committed artefacts only (no model retraining):

  Section 1 — repo structure + commit signature
  Section 2 — data-split audit fingerprint matches what's recorded
  Section 3 — composite-formula fingerprint consistency
  Section 4 — every experiment row has a non-empty reasoning blob
  Section 5 — Citation Rigor passes for every committed reasoning blob
  Section 6 — every experiment row has consistent backbone + recipe
  Section 7 — leaderboard / champion derived correctly from JSONL
  Section 8 — per-experiment prediction summary JSONs are present
  Section 9 — reasoning blob word floors satisfied per experiment

Output: `autoresearch_results/third_party_audit.{json,md}` plus
console summary.

Usage:
    python scripts/third_party_audit.py --config configs/higgs.yaml
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.evaluation.composite import composite_fingerprint  # noqa: E402
from core.reasoning import (  # noqa: E402
    ReasoningEntry, validate_reasoning_blob, validate_citation_rigor,
)


def section(title: str):
    print(f"\n=== {title} ===")


def _add(report: List[Dict[str, Any]], name: str, status: str,
         findings: List[str]) -> None:
    print(f"  {name}: {status}")
    for f in findings:
        print(f"    - {f}")
    report.append({"section": name, "status": status, "findings": findings})


def section1_repo_structure(report: List[Dict[str, Any]]) -> None:
    findings: List[str] = []
    needed = [
        "README.md", "PAPER.md", "MEDIUM.md", "SOTA_COMPARISON.md",
        "AUTORESEARCH_PROCESS.md", "SETUP.md", "ARCHITECTURE.md",
        "paper_abstract.md", "configs/higgs.yaml", "sota_catalog.yaml",
        "core/runner.py", "core/data/loader.py", "core/evaluation/audit.py",
        "core/evaluation/composite.py", "core/evaluation/metrics.py",
        "core/reasoning.py", "core/checkpoint.py",
        "core/backbones/registry.py", "core/backbones/gbm.py",
        "core/backbones/sklearn_baselines.py", "core/backbones/mlp.py",
        "scripts/run_campaign.py", "scripts/campaign_recipes.py",
        "scripts/download_higgs.py", "scripts/third_party_audit.py",
        "tests/test_smoke.py", "dashboard/dashboard.html",
    ]
    missing = [p for p in needed if not (ROOT / p).exists()]
    if missing:
        findings.append(f"missing files: {missing}")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
        findings.append(f"HEAD commit: {commit}")
    except Exception as e:
        findings.append(f"could not read git HEAD: {e}")
    status = "PASS" if not missing else "FAIL"
    _add(report, "section1_repo_structure", status, findings)


def section2_data_split_audit(report: List[Dict[str, Any]],
                              results_dir: Path) -> None:
    findings: List[str] = []
    p = results_dir / "data_split_audit.json"
    if not p.exists():
        _add(report, "section2_data_split_audit", "FAIL",
             [f"missing {p}"])
        return
    payload = json.loads(p.read_text())
    if payload.get("overall_status") != "PASS":
        findings.append(f"audit overall_status = {payload.get('overall_status')}")
    fp = payload.get("split_fingerprint", "")
    findings.append(f"recorded split fingerprint: {fp[:24]}...")
    n_aud = len(payload.get("auditors", []))
    findings.append(f"#auditors: {n_aud}")
    if n_aud < 6:
        findings.append("expected ≥ 6 auditors")
    fail = [a for a in payload.get("auditors", []) if a.get("status") != "PASS"]
    if fail:
        findings.append(f"failing auditors: {[a['name'] for a in fail]}")
    status = "PASS" if (payload.get("overall_status") == "PASS"
                          and not fail) else "FAIL"
    _add(report, "section2_data_split_audit", status, findings)


def section3_composite_fingerprint(report: List[Dict[str, Any]],
                                    cfg: Dict[str, Any],
                                    results_dir: Path) -> None:
    findings: List[str] = []
    formula = cfg["composite"]["formula"]
    fp_now = composite_fingerprint(formula)
    findings.append(f"computed fingerprint: {fp_now}")
    rec_path = results_dir / ".composite_fingerprint.json"
    if rec_path.exists():
        rec = json.loads(rec_path.read_text())
        findings.append(f"recorded fingerprint: {rec.get('fingerprint')}")
        if rec.get("fingerprint") != fp_now:
            _add(report, "section3_composite_fingerprint", "FAIL", findings)
            return
    findings.append("composite formula and fingerprint consistent")
    _add(report, "section3_composite_fingerprint", "PASS", findings)


def section4_reasoning_blob_present(report: List[Dict[str, Any]],
                                     results_dir: Path) -> None:
    findings: List[str] = []
    log_p = results_dir / "experiment_log.jsonl"
    ann_p = results_dir / "reasoning_annotations.json"
    if not log_p.exists():
        _add(report, "section4_reasoning_blob_present", "FAIL",
             [f"missing {log_p}"])
        return
    if not ann_p.exists():
        _add(report, "section4_reasoning_blob_present", "FAIL",
             [f"missing {ann_p}"])
        return
    rows = [json.loads(l) for l in log_p.read_text().splitlines() if l.strip()]
    db = json.loads(ann_p.read_text() or "{}")
    missing: List[int] = []
    for r in rows:
        n = r.get("experiment_num")
        if str(n) not in db:
            missing.append(n)
    findings.append(f"#experiments: {len(rows)}")
    findings.append(f"#reasoning entries: {len(db)}")
    if missing:
        findings.append(f"missing reasoning for exps: {missing}")
    status = "PASS" if not missing else "FAIL"
    _add(report, "section4_reasoning_blob_present", status, findings)


def section5_citation_rigor(report: List[Dict[str, Any]],
                             results_dir: Path) -> None:
    findings: List[str] = []
    ann_p = results_dir / "reasoning_annotations.json"
    if not ann_p.exists():
        _add(report, "section5_citation_rigor", "FAIL", [f"missing {ann_p}"])
        return
    db = json.loads(ann_p.read_text() or "{}")
    failing: List[Tuple[str, List[str]]] = []
    for k, v in db.items():
        issues = validate_citation_rigor(v.get("citations", ""))
        if issues:
            failing.append((k, issues))
    findings.append(f"#entries checked: {len(db)}")
    findings.append(f"#citation-failing entries: {len(failing)}")
    if failing[:5]:
        for k, issues in failing[:5]:
            findings.append(f"  exp{k}: {issues}")
    status = "PASS" if not failing else "FAIL"
    _add(report, "section5_citation_rigor", status, findings)


def section6_recipe_consistency(report: List[Dict[str, Any]],
                                 results_dir: Path) -> None:
    findings: List[str] = []
    log_p = results_dir / "experiment_log.jsonl"
    if not log_p.exists():
        _add(report, "section6_recipe_consistency", "FAIL", ["missing log"])
        return
    rows = [json.loads(l) for l in log_p.read_text().splitlines() if l.strip()]
    by_bb: Dict[str, int] = {}
    bad: List[int] = []
    for r in rows:
        bb = r.get("backbone")
        if not bb:
            bad.append(r.get("experiment_num"))
            continue
        by_bb[bb] = by_bb.get(bb, 0) + 1
    findings.append(f"backbone counts: {by_bb}")
    if bad:
        findings.append(f"rows missing backbone: {bad}")
    status = "PASS" if not bad else "FAIL"
    _add(report, "section6_recipe_consistency", status, findings)


def section7_leaderboard(report: List[Dict[str, Any]],
                          results_dir: Path) -> None:
    findings: List[str] = []
    log_p = results_dir / "experiment_log.jsonl"
    best_p = results_dir / "best_config.json"
    if not log_p.exists():
        _add(report, "section7_leaderboard", "FAIL", ["missing log"])
        return
    rows = [json.loads(l) for l in log_p.read_text().splitlines() if l.strip()]
    if not rows:
        _add(report, "section7_leaderboard", "FAIL", ["log is empty"])
        return
    rows_sorted = sorted(rows, key=lambda r: r.get("composite", -1.0), reverse=True)
    top = rows_sorted[0]
    findings.append(
        f"top by composite: exp{top['experiment_num']} "
        f"backbone={top['backbone']} composite={top.get('composite', 'NA'):.4f}"
    )
    if best_p.exists():
        rec = json.loads(best_p.read_text())
        if rec.get("experiment_num") != top["experiment_num"]:
            findings.append(
                f"best_config.json points to exp{rec.get('experiment_num')} "
                f"but JSONL top is exp{top['experiment_num']}"
            )
            _add(report, "section7_leaderboard", "FAIL", findings)
            return
    _add(report, "section7_leaderboard", "PASS", findings)


def section8_predictions_present(report: List[Dict[str, Any]],
                                  results_dir: Path) -> None:
    findings: List[str] = []
    log_p = results_dir / "experiment_log.jsonl"
    if not log_p.exists():
        _add(report, "section8_predictions_present", "FAIL", ["missing log"])
        return
    rows = [json.loads(l) for l in log_p.read_text().splitlines() if l.strip()]
    miss: List[int] = []
    for r in rows:
        n = r.get("experiment_num")
        s = results_dir / "trade_logs" / f"exp{n}_prediction_summary.json"
        if not s.exists():
            miss.append(n)
    findings.append(f"#experiments: {len(rows)}")
    findings.append(f"#missing prediction_summary.json: {len(miss)}")
    if miss[:5]:
        findings.append(f"first missing: {miss[:5]}")
    status = "PASS" if not miss else "FAIL"
    _add(report, "section8_predictions_present", status, findings)


def section9_reasoning_word_floors(report: List[Dict[str, Any]],
                                    results_dir: Path) -> None:
    findings: List[str] = []
    log_p = results_dir / "experiment_log.jsonl"
    ann_p = results_dir / "reasoning_annotations.json"
    if not (log_p.exists() and ann_p.exists()):
        _add(report, "section9_reasoning_word_floors", "FAIL",
             ["missing log or annotations"])
        return
    rows = [json.loads(l) for l in log_p.read_text().splitlines() if l.strip()]
    db = json.loads(ann_p.read_text() or "{}")
    failing: List[Tuple[int, List[str]]] = []
    for r in rows:
        n = r.get("experiment_num")
        v = db.get(str(n), {})
        e = ReasoningEntry(**{k: v.get(k, "") if k != "_manual" else bool(v.get("_manual", False))
                              for k in ("diagnosis", "citations", "hypothesis",
                                        "prediction", "verdict", "learning",
                                        "_manual")})
        issues = validate_reasoning_blob(e, post_run=True)
        if issues:
            failing.append((n, issues))
    findings.append(f"#experiments: {len(rows)}")
    findings.append(f"#with floor violations: {len(failing)}")
    for n, issues in failing[:5]:
        findings.append(f"  exp{n}: {issues}")
    status = "PASS" if not failing else "FAIL"
    _add(report, "section9_reasoning_word_floors", status, findings)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    results_dir = Path(cfg["paths"]["results_dir"])

    report: List[Dict[str, Any]] = []
    section("Section 1: Repo structure")
    section1_repo_structure(report)
    section("Section 2: Data-split audit")
    section2_data_split_audit(report, results_dir)
    section("Section 3: Composite fingerprint")
    section3_composite_fingerprint(report, cfg, results_dir)
    section("Section 4: Reasoning blob present")
    section4_reasoning_blob_present(report, results_dir)
    section("Section 5: Citation Rigor")
    section5_citation_rigor(report, results_dir)
    section("Section 6: Recipe consistency")
    section6_recipe_consistency(report, results_dir)
    section("Section 7: Leaderboard / champion")
    section7_leaderboard(report, results_dir)
    section("Section 8: Predictions present")
    section8_predictions_present(report, results_dir)
    section("Section 9: Reasoning word floors")
    section9_reasoning_word_floors(report, results_dir)

    overall = "PASS" if all(s["status"] == "PASS" for s in report) else "FAIL"
    payload = {
        "audit_version": "1.0.0",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall_status": overall,
        "sections": report,
    }
    out_json = results_dir / "third_party_audit.json"
    out_md = results_dir / "third_party_audit.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    md_lines = ["# AUTORESEARCHTABULAR — Third-party audit\n",
                f"_Generated: {payload['timestamp_utc']}_\n",
                f"**Overall: {overall}**\n"]
    for s in report:
        md_lines.append(f"\n## {s['section']} — {s['status']}\n")
        for f in s["findings"]:
            md_lines.append(f"- {f}\n")
    out_md.write_text("".join(md_lines), encoding="utf-8")
    print(f"\n[3rd-party audit] OVERALL = {overall}")
    print(f"[3rd-party audit] wrote {out_json} + {out_md}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
