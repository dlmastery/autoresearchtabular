"""One experiment per invocation — AUTORESEARCHTABULAR runner.

Refuses to launch unless data-split audit + Citation Rigor + Reasoning
Blob Completeness gates pass.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import random
from dataclasses import asdict
from typing import Any, Dict

import numpy as np


def _pin_to_safe_cores() -> None:
    if os.environ.get("AUTORESEARCHTABULAR_USE_ALL_CORES") == "1":
        return
    n_threads = int(os.environ.get("AUTORESEARCHTABULAR_N_THREADS", "4"))
    try:
        import torch
        torch.set_num_threads(n_threads)
    except Exception:
        pass
    try:
        import psutil
        p = psutil.Process()
        if hasattr(p, "cpu_affinity"):
            p.cpu_affinity([0, 2, 4, 6])
    except Exception as exc:
        print(f"[warn] pin_to_safe_cores failed: {exc!r}", file=sys.stderr)


_pin_to_safe_cores()

import yaml  # noqa: E402

from .backbones import create_backbone, BACKBONE_REGISTRY  # noqa: E402
from .data import load_higgs_split, SPLIT_NAMES  # noqa: E402
from .evaluation import (  # noqa: E402
    compute_split_metrics, compute_composite, audit_or_die,
)
from .evaluation.composite import CompositeFingerprintError  # noqa: E402
from .reasoning import (  # noqa: E402
    ReasoningEntry, validate_reasoning_blob, load_reasoning_db, commit_post_run,
)
from .checkpoint import CheckpointManager  # noqa: E402


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_sota(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_recipe(backbone: str, sota: Dict, overrides: Dict) -> Dict[str, Any]:
    pool = []
    for tier in ("tier1_baselines", "tier2_neural", "tier3_sota_2024_2026"):
        pool.extend(sota.get(tier, []))
    recipe = next((r for r in pool if r.get("id") == backbone), {})
    out = dict(recipe)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


def _next_experiment_num(jsonl_path: str) -> int:
    if not os.path.exists(jsonl_path):
        return 1
    n = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n + 1


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--num-leaves", type=int, default=None)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--reg-alpha", type=float, default=None)
    p.add_argument("--reg-lambda", type=float, default=None)
    p.add_argument("--feature-fraction", type=float, default=None)
    p.add_argument("--bagging-fraction", type=float, default=None)
    p.add_argument("--subsample", type=float, default=None)
    p.add_argument("--colsample-bytree", type=float, default=None)
    p.add_argument("--l2-leaf-reg", type=float, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--bypass-reasoning-gate", action="store_true")
    p.add_argument("--subset-train-n", type=int, default=None,
                   help="train on first N rows (val/test always full 500k)")
    return p.parse_args()


def _per_pred_csv(path: str, pred_arrays: Dict[str, Dict[str, np.ndarray]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("row_idx,split,label,prob_positive,pred_label,correct,brier\n")
        for split in ("train", "val", "test"):
            d = pred_arrays.get(split)
            if d is None:
                continue
            y, p, ri = d["y"], d["prob"], d["row_idx"]
            preds = (p >= 0.5).astype(int)
            for i in range(len(y)):
                f.write(f"{int(ri[i])},{split},{int(y[i])},{p[i]:.6f},"
                        f"{int(preds[i])},{int(int(preds[i])==int(y[i]))},"
                        f"{(p[i]-y[i])**2:.6f}\n")


def main() -> int:
    args = _cli()
    cfg = _load_config(args.config)
    sota_path = os.path.join(os.path.dirname(args.config) or ".", "..", "sota_catalog.yaml")
    sota = _load_sota(sota_path)
    overrides = {
        "epochs": args.epochs, "patience": args.patience, "lr": args.lr,
        "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "n_estimators": args.n_estimators, "num_leaves": args.num_leaves,
        "max_depth": args.max_depth, "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda, "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction, "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree, "l2_leaf_reg": args.l2_leaf_reg,
        "depth": args.depth, "iterations": args.iterations,
    }
    recipe = _resolve_recipe(args.backbone, sota, {})
    if not recipe:
        print(f"[warn] backbone {args.backbone!r} not in sota_catalog.yaml; using bare defaults",
              file=sys.stderr)
    if args.backbone not in BACKBONE_REGISTRY:
        print(f"[error] backbone {args.backbone!r} not registered. "
              f"Available: {sorted(BACKBONE_REGISTRY)}", file=sys.stderr)
        return 2

    paths = cfg["paths"]
    results_dir = paths["results_dir"]
    memory_dir = paths["memory_dir"]
    data_cache = paths["data_cache"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "trade_logs"), exist_ok=True)
    os.makedirs(memory_dir, exist_ok=True)

    jsonl_path = os.path.join(results_dir, "experiment_log.jsonl")
    annotations_path = os.path.join(results_dir, "reasoning_annotations.json")
    fp_path = os.path.join(results_dir, ".composite_fingerprint.json")
    exp_num = _next_experiment_num(jsonl_path)

    # Audit gate
    audit_or_die(results_dir=results_dir, max_age_seconds=86400)
    print(f"[runner] data split audit OK")

    # Reasoning gate
    db = load_reasoning_db(annotations_path)
    raw = db.get(str(exp_num))
    entry = ReasoningEntry(**{
        k: raw.get(k, "") if k != "_manual" else bool(raw.get("_manual", False))
        for k in ("diagnosis", "citations", "hypothesis", "prediction",
                  "verdict", "learning", "_manual")
    }) if raw else ReasoningEntry()
    violations = validate_reasoning_blob(entry, post_run=False)
    if violations and not args.bypass_reasoning_gate:
        print(f"\n[reasoning gate FAILED for exp{exp_num}]", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 3

    # Seed
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    _seed_all(seed)

    # Data
    subset_train_n = args.subset_train_n if args.subset_train_n is not None \
        else cfg["training"].get("subset_train_n")
    standardize = bool(cfg["training"].get("standardize_features", True))
    family = recipe.get("family", "")
    needs_std = family in ("neural_tabular", "neural_tabular_ensemble", "linear")
    splits = load_higgs_split(data_cache=data_cache, subset_train_n=subset_train_n,
                               standardize=needs_std and standardize)
    X_train, y_train, ri_train = splits["train"]
    X_val,   y_val,   ri_val   = splits["val"]
    X_test,  y_test,  ri_test  = splits["test"]
    print(f"[runner] sizes: train={len(y_train):,} val={len(y_val):,} test={len(y_test):,}")

    # Build + train
    cfg_params = dict(recipe.get("params", {}) if isinstance(recipe.get("params"), dict)
                       else recipe)
    # Strip known non-init keys
    for k in ("id", "family", "paper", "precision", "arch", "params",
              "sklearn_class", "scheduler"):
        cfg_params.pop(k, None)
    # Apply overrides
    for k, v in overrides.items():
        if v is not None:
            cfg_params[k] = v
    # Pull arch-level keys for neural backbones
    arch = recipe.get("arch", {}) if isinstance(recipe.get("arch"), dict) else {}
    cfg_params.update({k: v for k, v in arch.items() if k not in cfg_params})

    print(f"[runner] backbone={args.backbone} params={cfg_params}")
    t0 = time.time()
    model = create_backbone(args.backbone, **cfg_params)
    info = model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    train_time = time.time() - t0

    # Predict + metrics
    p_train = model.predict_proba(X_train)
    p_val = model.predict_proba(X_val)
    p_test = model.predict_proba(X_test)
    metrics = {
        "train": compute_split_metrics(y_train, p_train),
        "val":   compute_split_metrics(y_val,   p_val),
        "test":  compute_split_metrics(y_test,  p_test),
    }
    print(f"  train AUC={metrics['train']['auc']:.4f} acc={metrics['train']['accuracy']:.4f}")
    print(f"  val   AUC={metrics['val']['auc']:.4f} acc={metrics['val']['accuracy']:.4f} "
          f"ECE={metrics['val']['ece']:.4f}")
    print(f"  test  AUC={metrics['test']['auc']:.4f} acc={metrics['test']['accuracy']:.4f} "
          f"ECE={metrics['test']['ece']:.4f}")

    try:
        comp = compute_composite(test_auc=metrics["test"]["auc"],
                                  val_auc=metrics["val"]["auc"],
                                  formula=cfg["composite"]["formula"],
                                  fingerprint_path=fp_path)
    except CompositeFingerprintError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 5

    # Decide champion
    best_path = os.path.join(results_dir, "best_config.json")
    prev_best = None
    if os.path.exists(best_path):
        try:
            with open(best_path) as f:
                prev_best = json.load(f).get("composite", float("-inf"))
        except Exception:
            prev_best = float("-inf")
    is_champion = (prev_best is None) or (comp["composite"] > (prev_best or float("-inf")))

    record = {
        "experiment_num": exp_num,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backbone": args.backbone,
        "description": args.description,
        "seed": seed,
        "subset_train_n": subset_train_n,
        "standardize": needs_std and standardize,
        "params": cfg_params,
        "metrics": metrics,
        "train_auc": metrics["train"]["auc"],
        "val_auc": metrics["val"]["auc"],
        "test_auc": metrics["test"]["auc"],
        "composite": comp["composite"],
        "composite_formula": comp["composite_formula"],
        "composite_fingerprint": comp["composite_fingerprint"],
        "train_time_s": train_time,
        "info": info,
        "status": "KEEP/CHAMPION" if is_champion else (
            "KEEP" if comp["composite"] >= 0.5 else "DISCARD"
        ),
    }

    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    print(f"[runner] wrote {jsonl_path} (exp{exp_num}) composite={record['composite']:.4f}")

    if is_champion:
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)
        try:
            ext = ".pt" if hasattr(model, "model") and "torch" in str(type(getattr(model, "model", "")) or "").lower() else ".joblib"
            model.save(os.path.join(results_dir, f"best_model{ext}"))
        except Exception as e:
            print(f"[warn] could not save best_model: {e}")
        print(f"[runner] NEW CHAMPION composite={record['composite']:.4f}")

    # Per-prediction CSV
    pred_csv = os.path.join(results_dir, "trade_logs", f"exp{exp_num}_predictions.csv")
    _per_pred_csv(pred_csv, {
        "train": {"y": y_train, "prob": p_train, "row_idx": ri_train[:len(y_train)]},
        "val":   {"y": y_val,   "prob": p_val,   "row_idx": ri_val},
        "test":  {"y": y_test,  "prob": p_test,  "row_idx": ri_test},
    })
    summary = os.path.join(results_dir, "trade_logs", f"exp{exp_num}_prediction_summary.json")
    with open(summary, "w") as f:
        json.dump({"per_split": metrics, "composite": comp["composite"]}, f, indent=2, default=str)

    # Post-run reasoning fallback
    raw = (load_reasoning_db(annotations_path) or {}).get(str(exp_num), {})
    if not (raw.get("verdict") or "").strip():
        commit_post_run(
            annotations_path, exp_num,
            verdict=(f"TODO-REWRITE — auto-filled. composite={record['composite']:.4f}; "
                     f"train AUC={metrics['train']['auc']:.4f}, val={metrics['val']['auc']:.4f}, "
                     f"test={metrics['test']['auc']:.4f}; status={record['status']}"),
            learning="TODO-REWRITE — Claude must write the per-split learning narrative.",
            fallback=True,
        )

    # Checkpoint
    history_rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                history_rows.append(json.loads(line))
            except Exception:
                pass
    cm = CheckpointManager(memory_dir=memory_dir, results_dir=results_dir)
    next_cmd = (f'"C:/Users/evija/anaconda3/python.exe" -m core.runner '
                f'--config configs/higgs.yaml --backbone {args.backbone} '
                f'--description "exp{exp_num+1}: <DESCRIBE>"')
    champ = json.load(open(best_path)) if os.path.exists(best_path) else None
    cm.refresh(experiment_record=record, next_command=next_cmd,
                champion=champ, history_rows=history_rows[-30:])
    cm.append_to_summary(record)

    print(f"[runner] DONE exp{exp_num} composite={record['composite']:.4f} "
          f"status={record['status']}  ({train_time:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
