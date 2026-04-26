"""Write a single reasoning entry into autoresearch_results/reasoning_annotations.json.

Used by both the smoke test and the campaign loop.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict


def write_entry(path: str, exp_num: int, entry: Dict[str, str]) -> None:
    db: Dict[str, Dict] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                db = json.load(f)
            except json.JSONDecodeError:
                db = {}
    db[str(exp_num)] = entry
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", required=True)
    p.add_argument("--exp", type=int, required=True)
    p.add_argument("--diagnosis", required=True)
    p.add_argument("--citations", required=True)
    p.add_argument("--hypothesis", required=True)
    p.add_argument("--prediction", required=True)
    args = p.parse_args()
    write_entry(args.annotations, args.exp, {
        "diagnosis": args.diagnosis,
        "citations": args.citations,
        "hypothesis": args.hypothesis,
        "prediction": args.prediction,
        "verdict": "",
        "learning": "",
        "_manual": True,
    })
    print(f"[reasoning] wrote exp {args.exp} to {args.annotations}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
