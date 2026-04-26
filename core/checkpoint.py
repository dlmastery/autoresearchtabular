"""Crash-recovery checkpoint manager (tabular)."""
from __future__ import annotations
import os
from datetime import datetime
from typing import Dict, List, Optional


class CheckpointManager:
    def __init__(self, memory_dir: str, results_dir: str):
        self.memory_dir = memory_dir
        self.results_dir = results_dir
        os.makedirs(memory_dir, exist_ok=True)
        self.path = os.path.join(memory_dir, "project_autoresearch_checkpoint.md")
        self.summary_path = os.path.join(results_dir, "experiment_summary.md")

    def refresh(self, *, experiment_record: Dict, next_command: str,
                champion: Optional[Dict] = None,
                history_rows: Optional[List[Dict]] = None) -> None:
        ts = datetime.utcnow().isoformat() + "Z"
        lines: List[str] = []
        lines.append("# AUTORESEARCHTABULAR — autoresearch checkpoint")
        lines.append("")
        lines.append(f"_Last updated: {ts}_")
        lines.append("")
        lines.append("## Session start instructions")
        lines.append("")
        lines.append("1. Read this file (you are here).")
        lines.append("2. Read `CLAUDE.md`.")
        lines.append("3. Read `autoresearch_results/experiment_log.jsonl` (last 3 entries).")
        lines.append("4. Run the audit if it's stale (> 24 h).")
        lines.append("5. Resume the loop with the command below.")
        lines.append("")
        lines.append("```")
        lines.append(next_command)
        lines.append("```")
        lines.append("")
        lines.append("## Current champion")
        if champion:
            lines.append(f"- Backbone: `{champion.get('backbone', '?')}`")
            lines.append(f"- Experiment: #{champion.get('experiment_num', '?')}")
            lines.append(f"- Composite: **{champion.get('composite', float('nan')):.4f}**")
            lines.append(f"- test_auc: {champion.get('test_auc', float('nan')):.4f}")
            lines.append(f"- val_auc: {champion.get('val_auc', float('nan')):.4f}")
            lines.append(f"- Description: {champion.get('description', '')}")
        else:
            lines.append("_no champion yet — run experiment 1_")
        lines.append("")
        e = experiment_record
        lines.append("## Last experiment")
        lines.append(f"- #{e.get('experiment_num', '?')} backbone=`{e.get('backbone', '?')}` "
                     f"composite={e.get('composite', float('nan')):.4f} "
                     f"status={e.get('status', '?')}")
        lines.append(f"- Description: {e.get('description', '')}")
        lines.append("")
        lines.append("## Experiment history")
        lines.append("")
        lines.append("| # | backbone | composite | test_auc | val_auc | train_auc | status |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in (history_rows or []):
            lines.append(
                f"| {r.get('experiment_num', '?')} | `{r.get('backbone', '?')}` | "
                f"{r.get('composite', float('nan')):.4f} | "
                f"{r.get('test_auc', float('nan')):.4f} | "
                f"{r.get('val_auc', float('nan')):.4f} | "
                f"{r.get('train_auc', float('nan')):.4f} | "
                f"{r.get('status', '?')} |"
            )
        lines.append("")
        lines.append("## Next experiment command")
        lines.append("```")
        lines.append(next_command)
        lines.append("```")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def append_to_summary(self, e: Dict) -> None:
        os.makedirs(os.path.dirname(self.summary_path) or ".", exist_ok=True)
        header_needed = not os.path.exists(self.summary_path)
        with open(self.summary_path, "a", encoding="utf-8") as f:
            if header_needed:
                f.write("# Experiment Summary — AUTORESEARCHTABULAR\n\n")
            f.write(f"\n## Exp{e.get('experiment_num', '?')}: {e.get('description', '')}\n")
            f.write(f"- **Backbone:** `{e.get('backbone', '?')}`\n")
            f.write(f"- **Composite:** {e.get('composite', float('nan')):.4f}\n")
            f.write(f"- **test_auc:** {e.get('test_auc', float('nan')):.4f}\n")
            f.write(f"- **val_auc:** {e.get('val_auc', float('nan')):.4f}\n")
            f.write(f"- **train_auc:** {e.get('train_auc', float('nan')):.4f}\n")
            f.write(f"- **Status:** {e.get('status', '?')}\n")
