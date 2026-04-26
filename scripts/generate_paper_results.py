"""Populate PAPER.md, MEDIUM.md, SOTA_COMPARISON.md, paper_abstract.md
with the live leaderboard from `autoresearch_results/experiment_log.jsonl`.

Usage:
    python scripts/generate_paper_results.py --config configs/higgs.yaml
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load_log(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _build_per_backbone_table(rows: List[Dict[str, Any]]) -> str:
    by_bb: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_bb[r["backbone"]].append(r)

    out = ["| backbone | n_exps | best composite | best test_auc | best val_auc | within-backbone gap (max−min composite) |",
           "|---|---:|---:|---:|---:|---:|"]
    for bb in sorted(by_bb.keys()):
        rs = by_bb[bb]
        comps = [r.get("composite", float("-inf")) for r in rs]
        tests = [r.get("test_auc", 0.0) for r in rs]
        vals = [r.get("val_auc", 0.0) for r in rs]
        if not comps:
            continue
        i = max(range(len(rs)), key=lambda j: comps[j])
        out.append(
            f"| `{bb}` | {len(rs)} | **{comps[i]:.4f}** | {tests[i]:.4f} | "
            f"{vals[i]:.4f} | {max(comps)-min(comps):.4f} |"
        )
    return "\n".join(out)


def _build_top10_table(rows: List[Dict[str, Any]]) -> str:
    rs = sorted(rows, key=lambda r: r.get("composite", float("-inf")),
                reverse=True)[:10]
    out = ["| rank | exp | backbone | composite | val_auc | test_auc | val/test gap | description |",
           "|---|---:|---|---:|---:|---:|---:|---|"]
    for i, r in enumerate(rs):
        gap = abs(r.get("val_auc", 0) - r.get("test_auc", 0))
        out.append(
            f"| {i+1} | {r.get('experiment_num','?')} | `{r.get('backbone','?')}` | "
            f"**{r.get('composite',0):.4f}** | {r.get('val_auc',0):.4f} | "
            f"{r.get('test_auc',0):.4f} | {gap:.4f} | "
            f"{r.get('description','?')[:60]} |"
        )
    return "\n".join(out)


def _build_summary_block(rows: List[Dict[str, Any]],
                         audit: Dict[str, Any]) -> str:
    n = len(rows)
    if n == 0:
        return ("_Campaign in progress — no completed experiments yet._")
    by_bb = defaultdict(list)
    for r in rows:
        by_bb[r["backbone"]].append(r)
    backbones_done = sorted(by_bb.keys())
    best = max(rows, key=lambda r: r.get("composite", float("-inf")))
    best_comp = best.get("composite", 0)
    best_bb = best.get("backbone", "?")
    best_exp = best.get("experiment_num", "?")
    fp = audit.get("split_fingerprint", "?")[:24]
    return (f"- **Total experiments completed:** {n}\n"
            f"- **Backbones with results:** {len(backbones_done)} "
            f"(`{', '.join(backbones_done)}`)\n"
            f"- **Global champion:** experiment #{best_exp} on `{best_bb}` "
            f"— composite **{best_comp:.4f}** "
            f"(val_auc {best.get('val_auc',0):.4f}, "
            f"test_auc {best.get('test_auc',0):.4f}, "
            f"val/test gap {abs(best.get('val_auc',0)-best.get('test_auc',0)):.4f})\n"
            f"- **Data-split fingerprint (Baldi 2014, subset_train_n=1M):** "
            f"`{fp}…`\n"
            f"- **Composite formula fingerprint:** SHA-256 of "
            f"`min(test_auc, val_auc) - 0.1 * abs(test_auc - val_auc)`")


def _patch_section(text: str, header: str, body: str) -> str:
    """Replace the section starting at `header` up to the next `## `.

    `body` may or may not include the header — we always strip a leading
    duplicate header before substituting, so duplicates from earlier runs
    are healed.
    """
    pattern = re.compile(rf"{re.escape(header)}\s*\n[\s\S]*?(?=\n## |\Z)")
    body_clean = body.lstrip()
    if body_clean.startswith(header):
        body_clean = body_clean[len(header):].lstrip("\n")
    rep = f"{header}\n\n{body_clean.rstrip()}\n"
    if pattern.search(text):
        # Greedy de-dup: collapse any consecutive duplicate headers.
        out = pattern.sub(rep, text, count=1)
        # Heal any pre-existing duplicate headers immediately preceding rep.
        dup = re.compile(rf"({re.escape(header)}\s*\n\s*){{2,}}")
        out = dup.sub(f"{header}\n\n", out)
        return out
    return text.rstrip() + "\n\n" + rep


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    results_dir = Path(cfg["paths"]["results_dir"])
    rows = _load_log(results_dir / "experiment_log.jsonl")
    audit_p = results_dir / "data_split_audit.json"
    audit = json.loads(audit_p.read_text(encoding="utf-8")) if audit_p.exists() else {}
    summary = _build_summary_block(rows, audit)
    per_bb = _build_per_backbone_table(rows)
    top10 = _build_top10_table(rows)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Update paper_abstract.md
    abs_p = ROOT / "paper_abstract.md"
    if abs_p.exists():
        text = abs_p.read_text(encoding="utf-8")
        block = (
            f"## Headline result\n\n"
            f"_Live snapshot — last updated {ts}._\n\n"
            f"{summary}\n\n"
            f"### Per-backbone leaderboard\n\n{per_bb}\n\n"
            f"### Global top 10\n\n{top10}\n"
        )
        text = _patch_section(text, "## Headline result", block)
        abs_p.write_text(text, encoding="utf-8")
        print(f"[generate] updated {abs_p}")

    # Update PAPER.md §4
    paper_p = ROOT / "PAPER.md"
    if paper_p.exists():
        text = paper_p.read_text(encoding="utf-8")
        block = (
            f"## 4. Results\n\n"
            f"_Live snapshot — last updated {ts}._\n\n"
            f"{summary}\n\n"
            f"### 4.1 Per-backbone leaderboard\n\n{per_bb}\n\n"
            f"### 4.2 Global top 10 by composite\n\n{top10}\n\n"
            f"### 4.3 Notes\n\n"
            f"All test-AUROC values are computed on the Baldi 2014 frozen "
            f"test split (rows `[10,500,000, 11,000,000)`); val on rows "
            f"`[10,000,000, 10,500,000)`; train on the first "
            f"{cfg['training'].get('subset_train_n','?'):,} rows of "
            f"`[0, 10,000,000)`. Every experiment row carries the "
            f"data-split fingerprint and the composite-formula fingerprint "
            f"recorded in `autoresearch_results/data_split_audit.json` and "
            f"`autoresearch_results/.composite_fingerprint.json`.\n"
        )
        text = _patch_section(text, "## 4. Results", block)
        paper_p.write_text(text, encoding="utf-8")
        print(f"[generate] updated {paper_p}")

    # Update SOTA_COMPARISON.md "Where we expect to land"
    sota_p = ROOT / "SOTA_COMPARISON.md"
    if sota_p.exists():
        text = sota_p.read_text(encoding="utf-8")
        block = (
            f"## Where we expect to land\n\n"
            f"_Live snapshot — last updated {ts}._\n\n"
            f"{summary}\n\n"
            f"### Per-backbone leaderboard\n\n{per_bb}\n\n"
            f"### Global top 10\n\n{top10}\n"
        )
        text = _patch_section(text, "## Where we expect to land", block)
        sota_p.write_text(text, encoding="utf-8")
        print(f"[generate] updated {sota_p}")

    # Update MEDIUM.md "What the loop produced overnight"
    med_p = ROOT / "MEDIUM.md"
    if med_p.exists():
        text = med_p.read_text(encoding="utf-8")
        block = (
            f"## What the loop produced overnight\n\n"
            f"_Live snapshot — last updated {ts}._\n\n"
            f"{summary}\n\n"
            f"### Per-backbone leaderboard\n\n{per_bb}\n\n"
            f"### Global top 10\n\n{top10}\n"
        )
        text = _patch_section(text, "## What the loop produced overnight", block)
        med_p.write_text(text, encoding="utf-8")
        print(f"[generate] updated {med_p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
