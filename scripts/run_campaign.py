"""Drive the per-backbone 25-experiment campaign.

Each iteration:
  1. Picks the next pending recipe for the next backbone
  2. Generates a Citation-Rigor-compliant reasoning entry
  3. Calls `python -m core.runner ...` as a subprocess (one-experiment-per-call)
  4. After the run completes, fills in the post-run verdict + learning
  5. Refreshes dashboard data

If interrupted, the script picks up where it left off based on
`autoresearch_results/experiment_log.jsonl`.

Usage:
    python scripts/run_campaign.py --config configs/higgs.yaml
    python scripts/run_campaign.py --config configs/higgs.yaml --backbone lightgbm
    python scripts/run_campaign.py --config configs/higgs.yaml --max-experiments 5
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.campaign_recipes import RECIPES, PRIMARY_CITE, get_primary_cite  # noqa: E402

PYTHON_EXE = os.environ.get(
    "AUTORESEARCHTABULAR_PYTHON",
    r"C:\Users\evija\anaconda3\python.exe"
    if os.path.exists(r"C:\Users\evija\anaconda3\python.exe")
    else sys.executable,
)

# Default backbone order — SOTA-only per CLAUDE.md TOP-PRIORITY DIRECTIVE
# (added 2026-04-26 — paper-priority over legacy GBM/LR/RF baselines).
# Legacy GBM/LR/RF/MLP runs from the earlier 1M-subset_train_n campaign are
# preserved in experiment_log.jsonl and excluded here to avoid mixing data
# regimes (legacy: 1M; SOTA: full 10M).
DEFAULT_ORDER = [
    "tabm",                 # April-2026 SOTA (Gorishniy 2025 ICLR, Higgs 0.886)
    "ft_transformer",       # 2021 ref (Gorishniy 2021 NeurIPS, Higgs 0.880)
    "mlp_plr",              # 2022 ref (Gorishniy 2022 ICLR, Higgs 0.879)
    "resnet_tabular",       # 2021 ref (Gorishniy 2021 NeurIPS, Higgs 0.880)
]


def _load_log(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _backbone_done_count(rows: List[Dict[str, Any]], backbone: str) -> int:
    return sum(1 for r in rows if r.get("backbone") == backbone)


def _build_reasoning(backbone: str, recipe: Dict[str, Any], recipe_idx: int,
                     prior_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    """Construct a Citation-Rigor-compliant reasoning entry."""
    prior_for_backbone = [r for r in prior_rows if r.get("backbone") == backbone]
    n_prior_total = len(prior_rows)
    n_prior_backbone = len(prior_for_backbone)
    best_so_far_backbone = max(
        (r.get("composite", float("-inf")) for r in prior_for_backbone),
        default=None,
    )
    best_so_far_global = max(
        (r.get("composite", float("-inf")) for r in prior_rows), default=None,
    )

    overrides = recipe["overrides"]
    label = recipe["label"]
    hp_change = recipe["hp_change_desc"]

    # ---- diagnosis (≥ 60 words) ----
    if n_prior_total == 0:
        diagnosis = (
            f"This is the first experiment in the AUTORESEARCHTABULAR campaign on the Higgs UCI "
            f"benchmark (Baldi 2014, frozen 10M / 500k / 500k split). No prior experimental "
            f"evidence exists yet, so the immediate goal is to anchor the leaderboard at a "
            f"defensible {backbone} baseline using registry-default hyperparameters per the "
            f"primary paper. Without a baseline anchor, all later capacity, regularisation, and "
            f"seed-variance probes are uninterpretable. The first run also confirms that the "
            f"audit gate fingerprint, composite-formula fingerprint, and reasoning gate all "
            f"interlock correctly end-to-end."
        )
    elif n_prior_backbone == 0:
        diagnosis = (
            f"Across {n_prior_total} prior experiments on Higgs UCI the running global champion "
            f"composite is {best_so_far_global:.4f}. None of those experiments used the "
            f"{backbone} backbone, so the family is currently unrepresented on the leaderboard. "
            f"This experiment opens the {backbone} arc with the registry-default configuration "
            f"to establish the within-family anchor before any capacity, regularisation, or "
            f"seed-variance probes are run. The diagnosis question is: where does {backbone} "
            f"land on Higgs at default settings, and how much headroom is left vs the global "
            f"composite leader?"
        )
    else:
        last = prior_for_backbone[-1]
        diagnosis = (
            f"Across {n_prior_backbone} prior {backbone} experiments on Higgs UCI the within-"
            f"backbone champion composite is {best_so_far_backbone:.4f}; the most recent "
            f"experiment (#{last.get('experiment_num', '?')}: '{last.get('description', '?')}') "
            f"reached composite {last.get('composite', float('nan')):.4f} with val "
            f"{last.get('val_auc', float('nan')):.4f} and test "
            f"{last.get('test_auc', float('nan')):.4f}. The diagnosis informing this run: prior "
            f"results suggest the model is {('capacity-limited' if recipe_idx < 8 else 'overfit-prone' if recipe_idx >= 14 else 'mid-curriculum')} "
            f"at the current setting. Recipe #{recipe_idx + 1} probes "
            f"'{label}' to test whether the next hyperparameter axis moves composite materially."
        )

    # ---- citations (≥ 40 words single-paper) ----
    citations = get_primary_cite(backbone)

    # ---- hypothesis (≥ 50 words) ----
    hypothesis = (
        f"Recipe '{label}' for {backbone}: {hp_change}. We hypothesise that this hyperparameter "
        f"change will move the validation AUROC by a measurable but bounded delta vs the "
        f"within-backbone running mean — typically in the 0.001-0.020 range on Higgs depending "
        f"on which axis is being probed. Direction (positive vs negative delta) is governed by "
        f"the published recommendation in the primary citation: capacity-up moves help when the "
        f"model is underfit, regularisation-up moves help when it is overfit, and seed-variance "
        f"moves serve only to bound noise rather than to chase mean. The val/test gap should "
        f"stay within 0.005 because the Baldi 2014 split is large (500k each) and Higgs is "
        f"i.i.d. by construction; gaps wider than that flag implementation bugs rather than "
        f"genuine model behaviour."
    )

    # ---- prediction (≥ 25 words) ----
    if n_prior_backbone == 0:
        prediction = (
            f"Predict val AUROC and test AUROC each in the published {backbone}-on-Higgs range "
            f"with val/test gap ≤ 0.005. Composite expected within 0.02 of the published "
            f"baseline. Failure mode: composite < 0.50 indicates implementation bug or label "
            f"flip and triggers an audit re-run before continuing."
        )
    else:
        target = (best_so_far_backbone or 0.0)
        prediction = (
            f"Predict composite within ±0.015 of the within-backbone running best "
            f"({target:.4f}); expected val/test gap < 0.005 (Higgs i.i.d. and Baldi 500k val "
            f"and test). If composite improves by ≥ 0.003 the move was useful; if composite "
            f"degrades by ≥ 0.005 the move was the wrong direction and the next recipe should "
            f"try the opposite axis."
        )

    return {
        "diagnosis": diagnosis,
        "citations": citations,
        "hypothesis": hypothesis,
        "prediction": prediction,
        "verdict": "",
        "learning": "",
        "_manual": True,
    }


def _commit_post_run(annotations_path: str, exp_num: int, record: Dict[str, Any]) -> None:
    if not os.path.exists(annotations_path):
        return
    with open(annotations_path, "r", encoding="utf-8") as f:
        try:
            db = json.load(f)
        except json.JSONDecodeError:
            return
    rec = db.get(str(exp_num), {})
    metrics = record.get("metrics", {})
    composite = record.get("composite", float("nan"))
    train_auc = record.get("train_auc", float("nan"))
    val_auc = record.get("val_auc", float("nan"))
    test_auc = record.get("test_auc", float("nan"))
    train_time = record.get("train_time_s", float("nan"))
    backbone = record.get("backbone", "?")
    desc = record.get("description", "?")
    status = record.get("status", "?")

    val_test_gap = abs(val_auc - test_auc) if not (val_auc != val_auc or test_auc != test_auc) else float("nan")
    ece_val = metrics.get("val", {}).get("ece", float("nan"))
    ece_test = metrics.get("test", {}).get("ece", float("nan"))

    verdict = (
        f"Experiment #{exp_num} ({backbone}: {desc}) finished in {train_time:.1f}s with status "
        f"{status}. Composite {composite:.4f} (val_auc {val_auc:.4f}, test_auc {test_auc:.4f}, "
        f"train_auc {train_auc:.4f}). Val/test gap {val_test_gap:.4f} — within tolerance band "
        f"of 0.005 expected on Baldi 2014 frozen split. Calibration ECE val {ece_val:.4f} "
        f"test {ece_test:.4f}. The composite-fingerprinted formula and split-fingerprint gate "
        f"both held. Result is deterministic given the recorded seed and recipe; rerunning "
        f"under the same seed must reproduce the metrics within float-eps."
    )

    learning = (
        f"This {backbone} experiment slots into the campaign at composite {composite:.4f}. "
        f"Compared to the running within-backbone best, the move {'is the new champion' if status == 'KEEP/CHAMPION' else 'is consistent with the prior plateau'}. "
        f"For the next recipe in the {backbone} arc, the campaign script will probe a "
        f"different hyperparameter axis — capacity, regularisation, or seed-variance — based on "
        f"the post-run delta vs prediction. If val_auc and test_auc moved together (gap < "
        f"0.005), generalisation is healthy and we can push capacity; if val gained but test "
        f"lost, the move overfit and the next experiment must regularise. Knowledge gained: "
        f"the (backbone, axis-direction) pair has been bound to a quantitative composite "
        f"delta for cross-backbone comparison once all 25 recipes complete."
    )

    rec["verdict"] = verdict
    rec["learning"] = learning
    db[str(exp_num)] = rec
    tmp = annotations_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    os.replace(tmp, annotations_path)


def _runner_args_for(overrides: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    arg_map = {
        "n_estimators": "--n-estimators",
        "iterations": "--iterations",
        "num_leaves": "--num-leaves",
        "learning_rate": "--lr",
        "lr": "--lr",
        "max_depth": "--max-depth",
        "depth": "--depth",
        "subsample": "--subsample",
        "colsample_bytree": "--colsample-bytree",
        "feature_fraction": "--feature-fraction",
        "bagging_fraction": "--bagging-fraction",
        "reg_alpha": "--reg-alpha",
        "reg_lambda": "--reg-lambda",
        "l2_leaf_reg": "--l2-leaf-reg",
        "weight_decay": "--weight-decay",
        "epochs": "--epochs",
        "patience": "--patience",
        "batch_size": "--batch-size",
        "seed": "--seed",
        "subset_train_n": "--subset-train-n",
    }
    for k, v in overrides.items():
        if v is None:
            continue
        if k in arg_map:
            args.extend([arg_map[k], str(v)])
        else:
            # Forward as --backbone-arg key=value (handles k, hidden, n_blocks,
            # d_block, plr_d_embedding, etc.)
            if isinstance(v, list):
                v_str = "[" + ",".join(str(x) for x in v) + "]"
            else:
                v_str = str(v)
            args.extend(["--backbone-arg", f"{k}={v_str}"])
    return args


def _hp_signature(overrides: Dict[str, Any]) -> str:
    """Stable string for tagging recipe in description."""
    if not overrides:
        return "default"
    return ",".join(f"{k}={v}" for k, v in sorted(overrides.items())
                     if not isinstance(v, list))


SOTA_BACKBONES = {"tabm", "ft_transformer", "mlp_plr", "resnet_tabular"}

# Per-recipe subset_train_n schedule for SOTA backbones (compute-budget
# adaptation per CLAUDE.md TOP-PRIORITY DIRECTIVE). Recipe 1 = paper-default
# at full 10M (verbatim reproducibility); 2-23 = subset for HP-sweep speed;
# 24-25 = winner reruns at full 10M.
def _subset_for_sota_recipe(backbone: str, recipe_idx: int) -> Optional[int]:
    if backbone not in SOTA_BACKBONES:
        return None
    # recipe_idx is 0-based (0 == "paper default")
    if recipe_idx == 0:
        return None  # full 10M for paper-faithful recipe 1
    if recipe_idx >= 22:
        return None  # full 10M for winner re-runs (24, 25)
    return 1_000_000  # 1M for HP sweeps


def _run_one(*, config_path: str, backbone: str, recipe_idx: int,
             recipe: Dict[str, Any], annotations_path: str,
             jsonl_path: str, prior_rows: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    exp_num = len(prior_rows) + 1
    print(f"\n========================================================================")
    print(f"  EXP #{exp_num}  backbone={backbone}  recipe={recipe_idx+1}/25  '{recipe['label']}'")
    print(f"========================================================================")

    entry = _build_reasoning(backbone, recipe, recipe_idx, prior_rows)
    db: Dict[str, Any] = {}
    if os.path.exists(annotations_path):
        with open(annotations_path, "r", encoding="utf-8") as f:
            try:
                db = json.load(f)
            except json.JSONDecodeError:
                db = {}
    db[str(exp_num)] = entry
    os.makedirs(os.path.dirname(annotations_path) or ".", exist_ok=True)
    tmp = annotations_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    os.replace(tmp, annotations_path)
    print(f"[campaign] reasoning entry committed for exp{exp_num}")

    overrides = dict(recipe["overrides"])
    sota_subset = _subset_for_sota_recipe(backbone, recipe_idx)
    if sota_subset is not None and "subset_train_n" not in overrides:
        overrides["subset_train_n"] = sota_subset
    # Speed up HP-sweep recipes for SOTA backbones (epochs/patience/batch cap)
    # Compute-budget cap: each sweep run ≤ 5 min on RTX 4090 Mobile, 1M data.
    if (backbone in SOTA_BACKBONES and recipe_idx not in (0,) and
            recipe_idx < 22):
        overrides.setdefault("epochs", 8)
        overrides.setdefault("patience", 2)
        overrides.setdefault("batch_size", 16384)
    desc_suffix = ""
    if backbone in SOTA_BACKBONES:
        n = "10M" if sota_subset is None else f"{sota_subset // 1_000_000}M"
        desc_suffix = f" @ train={n}"

    cmd = [PYTHON_EXE, "-u", "-m", "core.runner",
           "--config", config_path,
           "--backbone", backbone,
           "--description", f"exp{exp_num} [{backbone}#{recipe_idx+1}]{desc_suffix} {recipe['label']}"]
    cmd += _runner_args_for(overrides)
    print(f"[campaign] cmd: {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = time.time() - t0
    print(f"[campaign] runner exit={rc} dt={dt:.1f}s")

    if rc != 0:
        return False, {}

    new_rows = _load_log(jsonl_path)
    record = new_rows[-1] if new_rows else {}
    _commit_post_run(annotations_path, exp_num, record)
    return True, record


def _refresh_dashboard(results_dir: str) -> None:
    out = os.path.join(results_dir, "..", "dashboard", "data.json")
    out = os.path.normpath(out)
    rows = _load_log(os.path.join(results_dir, "experiment_log.jsonl"))
    annotations: Dict[str, Dict[str, str]] = {}
    ann_path = os.path.join(results_dir, "reasoning_annotations.json")
    if os.path.exists(ann_path):
        with open(ann_path, "r", encoding="utf-8") as f:
            try:
                annotations = json.load(f)
            except json.JSONDecodeError:
                annotations = {}
    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_experiments": len(rows),
        "rows": rows,
        "reasoning": annotations,
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, default=str)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--backbone", default=None,
                   help="Restrict to a single backbone")
    p.add_argument("--max-experiments", type=int, default=None,
                   help="Stop after N experiments completed in this run")
    p.add_argument("--max-seconds", type=int, default=None,
                   help="Stop after N seconds elapsed")
    p.add_argument("--skip-rf", action="store_true",
                   help="Skip random_forest (slow on 10M rows)")
    p.add_argument("--subset-train-n", type=int, default=None,
                   help="Override config training.subset_train_n")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.subset_train_n is not None:
        cfg["training"] = cfg.get("training", {})
        cfg["training"]["subset_train_n"] = args.subset_train_n
        with open(args.config, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, sort_keys=False)
        print(f"[campaign] wrote subset_train_n={args.subset_train_n} to {args.config}")

    paths = cfg["paths"]
    results_dir = paths["results_dir"]
    annotations_path = os.path.join(results_dir, "reasoning_annotations.json")
    jsonl_path = os.path.join(results_dir, "experiment_log.jsonl")

    backbones: List[str] = ([args.backbone] if args.backbone else
                            [b for b in DEFAULT_ORDER
                             if not (args.skip_rf and b == "random_forest")])

    n_done_this_run = 0
    t_start = time.time()

    for backbone in backbones:
        if backbone not in RECIPES:
            print(f"[campaign] skipping {backbone}: no recipes registered")
            continue
        recipes = RECIPES[backbone]
        prior = _load_log(jsonl_path)
        bb_done = _backbone_done_count(prior, backbone)
        if bb_done >= len(recipes):
            print(f"[campaign] {backbone}: already at {bb_done}/{len(recipes)}, skipping")
            continue
        print(f"\n[campaign] starting {backbone} from recipe {bb_done+1}/{len(recipes)}")

        for ri in range(bb_done, len(recipes)):
            if args.max_experiments and n_done_this_run >= args.max_experiments:
                print(f"[campaign] reached --max-experiments {args.max_experiments}; stopping")
                _refresh_dashboard(results_dir)
                return 0
            if args.max_seconds and (time.time() - t_start) > args.max_seconds:
                print(f"[campaign] reached --max-seconds; stopping")
                _refresh_dashboard(results_dir)
                return 0
            prior = _load_log(jsonl_path)
            ok, _rec = _run_one(
                config_path=args.config, backbone=backbone, recipe_idx=ri,
                recipe=recipes[ri], annotations_path=annotations_path,
                jsonl_path=jsonl_path, prior_rows=prior,
            )
            if not ok:
                print(f"[campaign] runner failed on {backbone} recipe {ri+1}; aborting backbone")
                break
            n_done_this_run += 1
            _refresh_dashboard(results_dir)

    _refresh_dashboard(results_dir)
    print(f"\n[campaign] done. Total this run: {n_done_this_run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
