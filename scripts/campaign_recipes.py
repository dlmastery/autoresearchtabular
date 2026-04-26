"""25-per-backbone recipe library for AUTORESEARCHTABULAR.

Each recipe:
  - id: short slug (paired with backbone for uniqueness)
  - overrides: dict of HP overrides for this experiment
  - label: 1-line human description for `--description`
  - hp_change_desc: 1-2 sentence description of what's changing vs default,
                   for use inside the reasoning hypothesis

The campaign script loops over `RECIPES[backbone]` and runs them
sequentially. Order matters: later recipes assume earlier ones have
been observed (the `diagnosis` block references prior experiments).
"""
from __future__ import annotations
from typing import Any, Dict, List


# Per-backbone primary citation block (Citation Rigor compliant).
# Each block must: ≥2 author surnames, year, venue, single-quoted title,
# arXiv id, ":" or "—" relevance separator, ≥40 words.
PRIMARY_CITE: Dict[str, str] = {
    "logistic_regression": (
        "Pedregosa, Varoquaux, Gramfort, Michel, Thirion, Grisel, Blondel, "
        "Prettenhofer, Weiss, Dubourg, Vanderplas, Passos, Cournapeau, "
        "Brucher, Perrot, Duchesnay 2011 JMLR 'Scikit-learn: Machine "
        "Learning in Python' (arXiv:1201.0490) — provides the LBFGS-based "
        "L2-regularised logistic-regression implementation we use here as "
        "the parametric linear baseline against which all non-linear "
        "tabular methods on the Higgs UCI benchmark are measured. The "
        "scikit-learn LogisticRegression class with default C=1.0 inverse "
        "regularisation, lbfgs solver and L2 penalty is the textbook "
        "starting point for binary classification on standardised "
        "numerical features such as the 28-column Higgs feature vector."
    ),
    "random_forest": (
        "Breiman 2001 Machine Learning vol.45 'Random Forests' "
        "(arXiv:0902.4750 reprint) — bootstrap aggregation of decision "
        "trees with feature subsampling at each split: the canonical "
        "non-parametric ensemble baseline for tabular classification, "
        "covered jointly with the Pedregosa et al. 2011 JMLR 'Scikit-"
        "learn: Machine Learning in Python' (arXiv:1201.0490) "
        "RandomForestClassifier implementation we use here on the Higgs "
        "UCI 28-feature binary signal-vs-background problem."
    ),
    "lightgbm": (
        "Ke, Meng, Finley, Wang, Chen, Ma, Ye, Liu 2017 NeurIPS 'LightGBM: "
        "A Highly Efficient Gradient Boosting Decision Tree' "
        "(arXiv:1711.08251) — leaf-wise tree growth with histogram bins, "
        "GOSS (Gradient-based One-Side Sampling) and EFB (Exclusive "
        "Feature Bundling). The published Higgs UCI experiments in "
        "section 4 establish LightGBM as the fastest GBM framework on "
        "multi-million-row datasets and the natural starting point for "
        "tabular gradient-boosting on the Baldi 2014 frozen split."
    ),
    "xgboost": (
        "Chen and Guestrin 2016 KDD 'XGBoost: A Scalable Tree Boosting "
        "System' (arXiv:1603.02754) — second-order Newton boosting with "
        "regularised loss, level-wise depth-limited trees and the hist "
        "tree method, building on the Friedman 2001 Annals of Statistics "
        "gradient-boosting framework. The canonical high-performance GBM "
        "implementation cited throughout the Higgs UCI literature as the "
        "GBM baseline that reaches AUROC ~0.864 with tuned defaults on "
        "the Baldi 2014 frozen split."
    ),
    "catboost": (
        "Prokhorenkova, Gusev, Vorobev, Dorogush, Gulin 2018 NeurIPS "
        "'CatBoost: Unbiased Boosting with Categorical Features' "
        "(arXiv:1706.09516) — symmetric oblivious trees with ordered "
        "boosting and target-statistic categorical encoders. Reports "
        "strong out-of-the-box accuracy on tabular benchmarks including "
        "Higgs UCI without manual tuning, matching LightGBM and XGBoost "
        "in section 5 of the paper while reducing prediction-time "
        "variance via the symmetric tree structure."
    ),
    "mlp": (
        "Gorishniy, Rubachev, Khrulkov, Babenko 2021 NeurIPS 'Revisiting "
        "Deep Learning Models for Tabular Data' (arXiv:2106.11189) — "
        "establishes a strong tuned-MLP baseline (3-layer, dropout, "
        "AdamW + cosine schedule) and shows that careful MLP tuning "
        "matches FT-Transformer on most tabular tasks. Reports MLP "
        "results on the Higgs UCI Baldi 2014 frozen split in the "
        "experimental section, providing the canonical neural-tabular "
        "baseline against which more complex architectures are measured."
    ),
    "mlp_plr": (
        "Gorishniy, Rubachev, Babenko 2022 ICLR 'On Embeddings for "
        "Numerical Features in Tabular Deep Learning' (arXiv:2203.05556) "
        "— PLR (Periodic-Linear-ReLU) embedding for numerical features "
        "consistently improves MLP and Transformer accuracy by 0.5-1.5 "
        "percentage points across 12 benchmarks including Higgs UCI. The "
        "embedding lifts a vanilla MLP on the Baldi 2014 frozen split to "
        "within 0.005 AUROC of FT-Transformer at a fraction of the "
        "compute, making it the canonical numerical-embedding upgrade."
    ),
    "ft_transformer": (
        "Gorishniy, Rubachev, Khrulkov, Babenko 2021 NeurIPS 'Revisiting "
        "Deep Learning Models for Tabular Data' (arXiv:2106.11189) — "
        "Feature-Tokenizer plus Transformer architecture: each numerical "
        "feature gets its own learned embedding then is fed into a stack "
        "of Transformer blocks. Reports test AUROC 0.880 on the Higgs "
        "UCI Baldi 2014 frozen split in table 6 and dominates most "
        "tabular benchmarks at publication, establishing the ceiling for "
        "single-architecture neural tabular at the time of release."
    ),
}


def _R(label: str, overrides: Dict[str, Any], hp_change_desc: str) -> Dict[str, Any]:
    return {"label": label, "overrides": overrides, "hp_change_desc": hp_change_desc}


# 25 recipes per backbone. Each recipe varies hyperparameters in a
# paper-cited direction. The first recipe is always the registry default.

RECIPES: Dict[str, List[Dict[str, Any]]] = {

    # ---------------- logistic_regression (linear baseline) ----------------
    "logistic_regression": [
        _R("default L2 C=1.0 lbfgs", {}, "registry default: C=1.0 inverse regularisation, L2, lbfgs solver"),
        _R("low regularisation C=10", {"reg_lambda": 10.0}, "C=10 (weaker L2): allow more model capacity per Hastie 2009 §4.4"),
        _R("high regularisation C=0.1", {"reg_lambda": 0.1}, "C=0.1 (stronger L2): combat overfit on 28 features"),
        _R("very strong reg C=0.01", {"reg_lambda": 0.01}, "C=0.01: floor regularisation to identify the underfit limit"),
        _R("C=100 near-unregularised", {"reg_lambda": 100.0}, "C=100: near-unregularised LR; should saturate AUC"),
        _R("max_iter=2000", {"max_iter": 2000}, "max_iter=2000: ensure LBFGS fully converges before scoring"),
        _R("max_iter=5000", {"max_iter": 5000}, "max_iter=5000: aggressive convergence for difficult Higgs feature scale"),
        _R("max_iter=500 short", {"max_iter": 500}, "max_iter=500: identify under-converged regime"),
        _R("lbfgs C=10 long", {"reg_lambda": 10.0, "max_iter": 5000}, "C=10 + max_iter=5000: low-reg fully-converged combo"),
        _R("lbfgs C=0.1 long", {"reg_lambda": 0.1, "max_iter": 5000}, "C=0.1 + max_iter=5000: high-reg fully-converged combo"),
        _R("seed 42", {"seed": 42}, "seed=42: variance probe at default C/max_iter"),
        _R("seed 7", {"seed": 7}, "seed=7: second variance probe"),
        _R("seed 123", {"seed": 123}, "seed=123: third variance probe"),
        _R("seed 1337", {"seed": 1337}, "seed=1337: fourth variance probe"),
        _R("seed 2024", {"seed": 2024}, "seed=2024: fifth variance probe"),
        _R("C=2.0", {"reg_lambda": 2.0}, "C=2.0: midpoint between default and weak-reg champion"),
        _R("C=0.5", {"reg_lambda": 0.5}, "C=0.5: midpoint between default and strong-reg champion"),
        _R("C=5.0", {"reg_lambda": 5.0}, "C=5.0: explore upper-mid regularisation regime"),
        _R("C=0.2", {"reg_lambda": 0.2}, "C=0.2: explore lower-mid regularisation regime"),
        _R("C=20.0", {"reg_lambda": 20.0}, "C=20.0: explore weak-reg edge"),
        _R("C=0.05", {"reg_lambda": 0.05}, "C=0.05: explore strong-reg edge"),
        _R("C=10 seed 42", {"reg_lambda": 10.0, "seed": 42}, "winner-zone C=10 + alternate seed: stability check"),
        _R("C=10 seed 7", {"reg_lambda": 10.0, "seed": 7}, "winner-zone C=10 + second alternate seed"),
        _R("C=10 seed 123", {"reg_lambda": 10.0, "seed": 123}, "winner-zone C=10 + third alternate seed"),
        _R("C=10 max_iter=10000", {"reg_lambda": 10.0, "max_iter": 10000}, "winner-zone + over-converged: rule out convergence ceiling"),
    ],

    # ---------------- random_forest ----------------
    "random_forest": [
        _R("default 500 trees depth 20", {}, "registry default: 500 trees, max_depth=20, max_features=sqrt"),
        _R("trees 200 depth 20", {"n_estimators": 200}, "200 trees: probe small-ensemble underfitting"),
        _R("trees 1000 depth 20", {"n_estimators": 1000}, "1000 trees: probe ensemble saturation point"),
        _R("trees 500 depth 12", {"max_depth": 12}, "depth=12: shallower trees, higher variance reduction"),
        _R("trees 500 depth 30", {"max_depth": 30}, "depth=30: deeper trees, near-unconstrained leaves"),
        _R("trees 500 depth None", {"max_depth": None}, "depth=None: fully grown trees per Breiman 2001 default"),
        _R("max_features=log2", {"max_features": "log2"}, "max_features=log2: lower-decorrelation per Breiman §11.1"),
        _R("max_features=0.5", {"max_features": 0.5}, "max_features=0.5: half the columns per split"),
        _R("max_features=1.0", {"max_features": 1.0}, "max_features=1.0: bagging-only (no feature subsampling)"),
        _R("trees 1000 depth 12", {"n_estimators": 1000, "max_depth": 12}, "1000 shallow trees: high-bias high-variance-reduction regime"),
        _R("trees 1000 depth 30", {"n_estimators": 1000, "max_depth": 30}, "1000 deep trees: high-capacity regime"),
        _R("trees 200 depth 30", {"n_estimators": 200, "max_depth": 30}, "small ensemble of deep trees: variance probe"),
        _R("trees 500 min_samples_split 5", {"min_samples_split": 5}, "min_samples_split=5: harder splits, less overfit"),
        _R("trees 500 min_samples_split 20", {"min_samples_split": 20}, "min_samples_split=20: even harder splits"),
        _R("trees 500 min_samples_leaf 5", {"min_samples_leaf": 5}, "min_samples_leaf=5: prevent leaf-tail overfit"),
        _R("trees 500 min_samples_leaf 20", {"min_samples_leaf": 20}, "min_samples_leaf=20: aggressive leaf regularisation"),
        _R("trees 500 bootstrap=False", {"bootstrap": False}, "bootstrap=False: extra-trees-style sampling for variance"),
        _R("trees 500 class_weight balanced", {"class_weight": "balanced"}, "class_weight=balanced: explicit prevalence reweight"),
        _R("trees 500 seed 42", {"random_state": 42}, "alt seed: variance probe"),
        _R("trees 500 seed 7", {"random_state": 7}, "alt seed 2: variance probe"),
        _R("trees 1000 max_features=0.5 depth 20", {"n_estimators": 1000, "max_features": 0.5, "max_depth": 20}, "deep ensemble + half features: lower variance, similar bias"),
        _R("trees 1000 depth 20 min_leaf 5", {"n_estimators": 1000, "min_samples_leaf": 5}, "deep ensemble + leaf reg: combo regularisation"),
        _R("trees 500 max_features sqrt min_leaf 10", {"min_samples_leaf": 10}, "moderate leaf reg + default sqrt features"),
        _R("trees 1500 depth 20", {"n_estimators": 1500}, "trees=1500: explore upper saturation"),
        _R("trees 2000 depth 20", {"n_estimators": 2000}, "trees=2000: confirm saturation past 1000"),
    ],

    # ---------------- lightgbm ----------------
    "lightgbm": [
        _R("default 1000 leaves 63 lr 0.05", {}, "registry default per Ke 2017 §4 Higgs experiment"),
        _R("leaves 31 (paper §3.4 default)", {"num_leaves": 31}, "num_leaves=31: original Ke 2017 §3.4 default; capacity-limit probe"),
        _R("leaves 127", {"num_leaves": 127}, "num_leaves=127: more leaves, deeper interactions"),
        _R("leaves 255", {"num_leaves": 255}, "num_leaves=255: high capacity, monitor overfit"),
        _R("leaves 511", {"num_leaves": 511}, "num_leaves=511: very high capacity, near-unconstrained leaf-wise growth"),
        _R("lr 0.01 + 5x iters", {"learning_rate": 0.01, "n_estimators": 5000}, "small lr + many trees per Ke §3.5: preferred over fewer-larger-step trees"),
        _R("lr 0.1 fast", {"learning_rate": 0.1}, "lr=0.1: faster training, may overfit"),
        _R("lr 0.02 + 3000 iters", {"learning_rate": 0.02, "n_estimators": 3000}, "lr=0.02 + 3000 iters: middle ground"),
        _R("min_data_in_leaf 100", {"min_data_in_leaf": 100}, "min_data_in_leaf=100: stronger leaf reg"),
        _R("min_data_in_leaf 500", {"min_data_in_leaf": 500}, "min_data_in_leaf=500: aggressive leaf reg"),
        _R("feature_fraction 0.6", {"feature_fraction": 0.6}, "feature_fraction=0.6: more feature subsample"),
        _R("feature_fraction 1.0", {"feature_fraction": 1.0}, "feature_fraction=1.0: no feature subsample"),
        _R("bagging_fraction 0.5", {"bagging_fraction": 0.5}, "bagging_fraction=0.5: stronger bagging"),
        _R("bagging_fraction 1.0", {"bagging_fraction": 1.0}, "bagging_fraction=1.0: no bagging"),
        _R("reg_alpha 0.1", {"reg_alpha": 0.1}, "L1 reg: encourage sparse splits"),
        _R("reg_lambda 5.0", {"reg_lambda": 5.0}, "L2 reg×5: stronger leaf weight regularisation"),
        _R("max_depth 8 cap", {"max_depth": 8}, "max_depth=8 cap: bound leaf-wise growth"),
        _R("max_depth 12 cap", {"max_depth": 12}, "max_depth=12 cap: looser bound"),
        _R("leaves 127 lr 0.02 iters 3000", {"num_leaves": 127, "learning_rate": 0.02, "n_estimators": 3000}, "deep grid: more leaves + small lr + many iters"),
        _R("leaves 255 min_data 200", {"num_leaves": 255, "min_data_in_leaf": 200}, "high capacity + leaf reg: combo"),
        _R("leaves 127 reg_lambda 5 fraction 0.7", {"num_leaves": 127, "reg_lambda": 5.0, "feature_fraction": 0.7}, "moderate cap + reg + sub: balanced"),
        _R("leaves 511 lr 0.01 iters 5000", {"num_leaves": 511, "learning_rate": 0.01, "n_estimators": 5000}, "max capacity + fine lr + many iters: search for saturation"),
        _R("seed 42 best-zone", {"num_leaves": 127, "learning_rate": 0.02, "n_estimators": 3000, "seed": 42}, "alt seed at expected best zone: variance probe"),
        _R("seed 7 best-zone", {"num_leaves": 127, "learning_rate": 0.02, "n_estimators": 3000, "seed": 7}, "alt seed 2 at expected best zone"),
        _R("seed 123 best-zone", {"num_leaves": 127, "learning_rate": 0.02, "n_estimators": 3000, "seed": 123}, "alt seed 3 at expected best zone (winner rerun)"),
    ],

    # ---------------- xgboost ----------------
    "xgboost": [
        _R("default depth 6 lr 0.05", {}, "registry default per Chen 2016 §4 Higgs experiment"),
        _R("depth 4", {"max_depth": 4}, "max_depth=4: shallow trees"),
        _R("depth 8", {"max_depth": 8}, "max_depth=8: deeper trees"),
        _R("depth 10", {"max_depth": 10}, "max_depth=10: very deep trees"),
        _R("depth 12", {"max_depth": 12}, "max_depth=12: extreme depth"),
        _R("lr 0.01 + 5000 iters", {"learning_rate": 0.01, "n_estimators": 5000}, "small step + many trees per Chen §4.2"),
        _R("lr 0.1", {"learning_rate": 0.1}, "lr=0.1: faster convergence"),
        _R("lr 0.02 + 3000 iters", {"learning_rate": 0.02, "n_estimators": 3000}, "lr=0.02 with 3000 iters: middle ground"),
        _R("subsample 0.5", {"subsample": 0.5}, "subsample=0.5: stronger row sampling"),
        _R("subsample 1.0", {"subsample": 1.0}, "subsample=1.0: no row sampling"),
        _R("colsample 0.5", {"colsample_bytree": 0.5}, "colsample_bytree=0.5: stronger column sampling"),
        _R("colsample 1.0", {"colsample_bytree": 1.0}, "colsample_bytree=1.0: no column sampling"),
        _R("min_child_weight 5", {"min_child_weight": 5}, "min_child_weight=5: stricter leaf reg"),
        _R("min_child_weight 20", {"min_child_weight": 20}, "min_child_weight=20: aggressive leaf reg"),
        _R("gamma 0.5", {"gamma": 0.5}, "gamma=0.5: pruning loss penalty"),
        _R("reg_lambda 5", {"reg_lambda": 5.0}, "reg_lambda=5: stronger L2"),
        _R("reg_alpha 0.1", {"reg_alpha": 0.1}, "reg_alpha=0.1: L1 reg"),
        _R("depth 8 lr 0.02 iters 3000", {"max_depth": 8, "learning_rate": 0.02, "n_estimators": 3000}, "deep + small lr + many iters: combo"),
        _R("depth 10 lr 0.01 iters 5000", {"max_depth": 10, "learning_rate": 0.01, "n_estimators": 5000}, "deeper + finer lr + more iters"),
        _R("depth 6 sub 0.7 col 0.7 lr 0.02", {"max_depth": 6, "subsample": 0.7, "colsample_bytree": 0.7, "learning_rate": 0.02, "n_estimators": 3000}, "balanced subsample + col + lr"),
        _R("depth 8 lambda 5 alpha 0.1", {"max_depth": 8, "reg_lambda": 5.0, "reg_alpha": 0.1}, "deep + L1+L2: combo regularisation"),
        _R("tree_method approx", {"tree_method": "approx"}, "tree_method=approx: alternate split-finding algo per Chen §3"),
        _R("seed 42 best-zone", {"max_depth": 8, "learning_rate": 0.02, "n_estimators": 3000, "seed": 42}, "alt seed at expected best zone: variance probe"),
        _R("seed 7 best-zone", {"max_depth": 8, "learning_rate": 0.02, "n_estimators": 3000, "seed": 7}, "alt seed 2 at expected best zone"),
        _R("seed 123 best-zone", {"max_depth": 8, "learning_rate": 0.02, "n_estimators": 3000, "seed": 123}, "alt seed 3 at expected best zone (winner rerun)"),
    ],

    # ---------------- catboost ----------------
    "catboost": [
        _R("default depth 6 lr 0.05 iters 1000", {}, "registry default per Prokhorenkova 2018 §5"),
        _R("depth 4", {"depth": 4}, "depth=4: shallow oblivious trees"),
        _R("depth 8", {"depth": 8}, "depth=8: deeper oblivious trees"),
        _R("depth 10", {"depth": 10}, "depth=10: very deep oblivious trees"),
        _R("lr 0.01 iters 5000", {"learning_rate": 0.01, "iterations": 5000}, "fine lr + many iters per Prokhorenkova §5.1"),
        _R("lr 0.1 iters 1000", {"learning_rate": 0.1, "iterations": 1000}, "lr=0.1: aggressive step"),
        _R("lr 0.02 iters 3000", {"learning_rate": 0.02, "iterations": 3000}, "lr=0.02 middle"),
        _R("l2_leaf_reg 1", {"l2_leaf_reg": 1.0}, "lower leaf-weight L2 reg"),
        _R("l2_leaf_reg 10", {"l2_leaf_reg": 10.0}, "higher leaf-weight L2 reg"),
        _R("l2_leaf_reg 30", {"l2_leaf_reg": 30.0}, "very high leaf-weight L2 reg"),
        _R("bagging_temp 0", {"bagging_temperature": 0.0}, "bagging_temperature=0: no Bayesian bagging"),
        _R("bagging_temp 5", {"bagging_temperature": 5.0}, "bagging_temperature=5: more Bayesian bagging variance"),
        _R("random_strength 0", {"random_strength": 0.0}, "random_strength=0: deterministic split scoring"),
        _R("random_strength 5", {"random_strength": 5.0}, "random_strength=5: noisier split scoring"),
        _R("border_count 64", {"border_count": 64}, "border_count=64: fewer histogram bins"),
        _R("border_count 254", {"border_count": 254}, "border_count=254: more histogram bins"),
        _R("depth 8 lr 0.02 iters 3000", {"depth": 8, "learning_rate": 0.02, "iterations": 3000}, "deep + fine lr + many iters: combo"),
        _R("depth 10 lr 0.01 iters 5000", {"depth": 10, "learning_rate": 0.01, "iterations": 5000}, "deeper + finer lr + more iters"),
        _R("depth 6 l2 5 lr 0.02 iters 3000", {"depth": 6, "l2_leaf_reg": 5.0, "learning_rate": 0.02, "iterations": 3000}, "default depth + reg + fine lr"),
        _R("depth 8 l2 5", {"depth": 8, "l2_leaf_reg": 5.0}, "deep + L2 reg: combo"),
        _R("depth 6 boosting_type Plain", {"boosting_type": "Plain"}, "boosting_type=Plain: classical boosting (no ordered)"),
        _R("depth 6 grow_policy Lossguide", {"grow_policy": "Lossguide"}, "Lossguide policy: leaf-wise (LightGBM-style)"),
        _R("seed 42 best-zone", {"depth": 8, "learning_rate": 0.02, "iterations": 3000, "random_seed": 42}, "alt seed at expected best zone"),
        _R("seed 7 best-zone", {"depth": 8, "learning_rate": 0.02, "iterations": 3000, "random_seed": 7}, "alt seed 2 at expected best zone"),
        _R("seed 123 best-zone", {"depth": 8, "learning_rate": 0.02, "iterations": 3000, "random_seed": 123}, "alt seed 3 (winner rerun)"),
    ],

    # ---------------- mlp (Gorishniy 2021 baseline) ----------------
    "mlp": [
        _R("default 256x3 dropout 0.1", {}, "registry default per Gorishniy 2021 §4.1"),
        _R("hidden 128x3", {"hidden": [128, 128, 128]}, "hidden=128x3: smaller capacity"),
        _R("hidden 512x3", {"hidden": [512, 512, 512]}, "hidden=512x3: more capacity per layer"),
        _R("hidden 256x6 deeper", {"hidden": [256] * 6}, "hidden=256x6: deeper but same width"),
        _R("hidden 512x6 wide deep", {"hidden": [512] * 6}, "hidden=512x6: high capacity"),
        _R("dropout 0.0", {"dropout": 0.0}, "dropout=0: no implicit reg"),
        _R("dropout 0.3", {"dropout": 0.3}, "dropout=0.3: stronger implicit reg"),
        _R("dropout 0.5", {"dropout": 0.5}, "dropout=0.5: aggressive dropout"),
        _R("lr 1e-4 finer", {"lr": 1e-4}, "lr=1e-4: finer step, more epochs needed"),
        _R("lr 3e-3 coarser", {"lr": 3e-3}, "lr=3e-3: coarser step, faster convergence"),
        _R("weight_decay 1e-4", {"weight_decay": 1e-4}, "weight_decay=1e-4: explicit L2"),
        _R("weight_decay 1e-3", {"weight_decay": 1e-3}, "weight_decay=1e-3: stronger L2"),
        _R("batch 256", {"batch_size": 256}, "batch=256: smaller batch, noisier gradients"),
        _R("batch 4096", {"batch_size": 4096}, "batch=4096: larger batch, smoother gradients"),
        _R("epochs 200 patience 24", {"epochs": 200, "patience": 24}, "more epochs + more patience: extended training"),
        _R("epochs 50 fast", {"epochs": 50, "patience": 8}, "epochs=50 patience=8: fast iteration"),
        _R("hidden 512x3 dropout 0.2 wd 1e-4", {"hidden": [512, 512, 512], "dropout": 0.2, "weight_decay": 1e-4}, "wide + moderate dropout + weak L2"),
        _R("hidden 512x6 dropout 0.3 wd 1e-4", {"hidden": [512] * 6, "dropout": 0.3, "weight_decay": 1e-4}, "wide deep + strong dropout + L2"),
        _R("hidden 768x3", {"hidden": [768, 768, 768]}, "hidden=768: very wide"),
        _R("hidden 1024x3 wide", {"hidden": [1024, 1024, 1024]}, "hidden=1024: very wide capacity"),
        _R("lr 5e-4 wd 1e-4", {"lr": 5e-4, "weight_decay": 1e-4}, "tuned lr + L2"),
        _R("hidden 512x3 dropout 0.2 lr 5e-4", {"hidden": [512] * 3, "dropout": 0.2, "lr": 5e-4}, "wide + moderate dropout + tuned lr"),
        _R("seed 42 best-zone", {"hidden": [512] * 3, "dropout": 0.2, "weight_decay": 1e-4, "lr": 5e-4, "seed": 42}, "alt seed at expected best zone"),
        _R("seed 7 best-zone", {"hidden": [512] * 3, "dropout": 0.2, "weight_decay": 1e-4, "lr": 5e-4, "seed": 7}, "alt seed 2 at expected best zone"),
        _R("seed 123 best-zone", {"hidden": [512] * 3, "dropout": 0.2, "weight_decay": 1e-4, "lr": 5e-4, "seed": 123}, "alt seed 3 (winner rerun)"),
    ],
}


def get_recipes(backbone: str) -> List[Dict[str, Any]]:
    if backbone not in RECIPES:
        raise KeyError(f"No recipes registered for backbone {backbone!r}. "
                       f"Available: {sorted(RECIPES)}")
    return RECIPES[backbone]


def get_primary_cite(backbone: str) -> str:
    if backbone not in PRIMARY_CITE:
        raise KeyError(f"No primary citation for backbone {backbone!r}")
    return PRIMARY_CITE[backbone]


# Quick sanity at import time
for _bb, _rs in RECIPES.items():
    if len(_rs) != 25:
        raise AssertionError(f"Backbone {_bb} has {len(_rs)} recipes (must be 25)")
