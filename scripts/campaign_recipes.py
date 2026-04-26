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
    "tabm": (
        "Gorishniy, Kotelnikov, Babenko 2025 ICLR 'TabM: Advancing "
        "Tabular Deep Learning With Parameter-Efficient Ensembling' "
        "(arXiv:2410.24210) — k-head BatchEnsemble of MLPs sharing a "
        "PLR-embedded backbone. Reports test AUROC ~0.886 on the Higgs "
        "UCI Baldi 2014 frozen split, currently leading TabArena and "
        "TALENT among non-foundation tabular methods. Default recipe "
        "follows the paper: k=32, hidden=[512]×3, lr=2e-3, batch 4096, "
        "AdamW, cosine schedule, weight decay 1e-5."
    ),
    "ft_transformer": (
        "Gorishniy, Rubachev, Khrulkov, Babenko 2021 NeurIPS 'Revisiting "
        "Deep Learning Models for Tabular Data' (arXiv:2106.11189) — "
        "Feature Tokenizer + Transformer architecture. Reports test "
        "AUROC 0.880 on the Higgs UCI Baldi 2014 frozen split (Tab.6). "
        "Default recipe per the paper: 3 transformer blocks, d_token "
        "192, attention heads 8, FFN multiplier 4/3, attention dropout "
        "0.2, residual dropout 0.0, AdamW lr 1e-4, weight decay 1e-5, "
        "batch 1024, cosine schedule."
    ),
    "mlp_plr": (
        "Gorishniy, Rubachev, Babenko 2022 ICLR 'On Embeddings for "
        "Numerical Features in Tabular Deep Learning' "
        "(arXiv:2203.05556) — periodic-linear-ReLU embedding for each "
        "numerical feature feeding a 3-layer MLP. Reports test AUROC "
        "0.879 on Higgs Tab.4, recovering most of the gap between "
        "vanilla MLP and FT-Transformer for ~10× lower compute. "
        "Default recipe: PLR n_frequencies=48, d_embedding=64, hidden "
        "[512]×3, dropout 0.1, AdamW lr 1e-3, batch 4096."
    ),
    "resnet_tabular": (
        "Gorishniy, Rubachev, Khrulkov, Babenko 2021 NeurIPS 'Revisiting "
        "Deep Learning Models for Tabular Data' (arXiv:2106.11189) — "
        "tabular ResNet baseline that the paper shows matches FT-"
        "Transformer on most benchmarks at a fraction of the compute. "
        "Reports test AUROC 0.880 on Higgs Tab.6. Default recipe: "
        "n_blocks=2, d_block=256, d_hidden_multiplier=2, dropout1 "
        "0.25, AdamW lr 1e-3, batch 4096."
    ),
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

    # ===================================================================
    # SOTA TIER (April 2026 priority — see CLAUDE.md SOTA-FIRST directive)
    # ===================================================================

    # ---------------- TabM (Gorishniy 2025 ICLR — arXiv:2410.24210) ----
    # Recipe #1 = paper's reported Higgs config (verbatim).
    "tabm": [
        _R("paper Higgs default k=32 h=512x3 lr=2e-3", {},
           "TabM paper-default per arXiv:2410.24210 §5: k=32 ensemble heads, hidden [512]×3, AdamW lr 2e-3, weight decay 1e-5, batch 4096"),
        _R("k=8 ensemble", {"k": 8}, "k=8 (smaller ensemble) per Tab.4 ablation"),
        _R("k=16 ensemble", {"k": 16}, "k=16 ensemble per Tab.4 ablation"),
        _R("k=64 ensemble", {"k": 64}, "k=64 ensemble per Tab.4 ablation"),
        _R("hidden=256x3", {"hidden": [256, 256, 256]}, "narrower MLP per §5.2 ablation"),
        _R("hidden=1024x3", {"hidden": [1024, 1024, 1024]}, "wider MLP per §5.2 ablation"),
        _R("hidden=512x6 deeper", {"hidden": [512] * 6}, "deeper backbone per §5.2 ablation"),
        _R("dropout=0.0", {"dropout": 0.0}, "no dropout per §5.2; tests reg saturation"),
        _R("dropout=0.3", {"dropout": 0.3}, "high dropout per Gorishniy 2021 MLP ablation"),
        _R("lr=1e-3 finer", {"lr": 1e-3}, "finer learning rate per §A.3"),
        _R("lr=5e-3 coarser", {"lr": 5e-3}, "coarser learning rate per §A.3"),
        _R("weight_decay=1e-4", {"weight_decay": 1e-4}, "stronger L2 per §5.3"),
        _R("weight_decay=0", {"weight_decay": 0.0}, "no L2 per §5.3 ablation"),
        _R("batch=2048 smaller", {"batch_size": 2048}, "smaller batch per §A.3"),
        _R("batch=8192 larger", {"batch_size": 8192}, "larger batch per §A.3 scaling"),
        _R("epochs=50 fast", {"epochs": 50, "patience": 8}, "faster schedule for early-converging configs"),
        _R("k=32 wider 1024x3", {"k": 32, "hidden": [1024] * 3}, "k=32 + wider per Tab.5 best-zone"),
        _R("k=32 deeper 512x6", {"k": 32, "hidden": [512] * 6}, "k=32 + deeper per §5.2 combo"),
        _R("k=64 wider 1024x3", {"k": 64, "hidden": [1024] * 3}, "k=64 + wider: capacity push"),
        _R("k=32 dropout=0.2 wd=1e-4", {"k": 32, "dropout": 0.2, "weight_decay": 1e-4},
           "moderate reg combo per §5.3"),
        _R("k=32 lr=1e-3 wd=1e-4", {"k": 32, "lr": 1e-3, "weight_decay": 1e-4},
           "tuned-lr + L2 per §5.3 best-zone"),
        _R("k=32 hidden=1024x3 lr=1e-3", {"k": 32, "hidden": [1024] * 3, "lr": 1e-3},
           "wider + finer lr per §A.3 sweep"),
        _R("seed 42 best-zone", {"k": 32, "hidden": [1024] * 3, "lr": 1e-3, "weight_decay": 1e-4, "seed": 42},
           "alt seed at expected best zone"),
        _R("seed 7 best-zone", {"k": 32, "hidden": [1024] * 3, "lr": 1e-3, "weight_decay": 1e-4, "seed": 7},
           "alt seed 2 at expected best zone"),
        _R("seed 123 best-zone", {"k": 32, "hidden": [1024] * 3, "lr": 1e-3, "weight_decay": 1e-4, "seed": 123},
           "alt seed 3 at expected best zone (winner rerun)"),
    ],

    # ---------------- FT-Transformer (Gorishniy 2021 NeurIPS arXiv:2106.11189) ----
    "ft_transformer": [
        _R("paper Higgs default 3blk d=192 h=8", {},
           "FT-Transformer paper-default per arXiv:2106.11189 Tab.6: 3 blocks, d_token 192, 8 heads, attn dropout 0.2, ffn dropout 0.1, AdamW lr 1e-4, batch 1024"),
        _R("d_block=128 narrower", {"d_block": 128}, "narrower token per §3.1 ablation"),
        _R("d_block=256 wider", {"d_block": 256}, "wider token per §3.1 ablation"),
        _R("d_block=384 wider", {"d_block": 384}, "even wider token per §3.1 ablation"),
        _R("n_blocks=2 shallower", {"n_blocks": 2}, "2 blocks per §A.2 ablation"),
        _R("n_blocks=4 deeper", {"n_blocks": 4}, "4 blocks per §A.2 ablation"),
        _R("n_blocks=6 deepest", {"n_blocks": 6}, "6 blocks per §A.2 ablation"),
        _R("attn_heads=4", {"attention_n_heads": 4}, "4 heads per §A.2"),
        _R("attn_heads=16", {"attention_n_heads": 16}, "16 heads per §A.2"),
        _R("attn_dropout=0.0", {"attention_dropout": 0.0}, "no attention dropout per Tab.7"),
        _R("attn_dropout=0.4", {"attention_dropout": 0.4}, "high attention dropout per Tab.7"),
        _R("ffn_dropout=0.0", {"ffn_dropout": 0.0}, "no ffn dropout per Tab.7"),
        _R("ffn_dropout=0.2", {"ffn_dropout": 0.2}, "high ffn dropout per Tab.7"),
        _R("residual_dropout=0.1", {"residual_dropout": 0.1}, "add residual dropout per Tab.7"),
        _R("lr=5e-5 finer", {"lr": 5e-5}, "finer lr per §A.3"),
        _R("lr=3e-4 coarser", {"lr": 3e-4}, "coarser lr per §A.3"),
        _R("weight_decay=1e-4", {"weight_decay": 1e-4}, "stronger L2 per §A.3"),
        _R("batch=256", {"batch_size": 256}, "smaller batch per §A.3"),
        _R("batch=2048 larger", {"batch_size": 2048}, "larger batch per §A.3"),
        _R("d=256 + 4blk", {"d_block": 256, "n_blocks": 4}, "wider+deeper combo"),
        _R("d=384 + 4blk", {"d_block": 384, "n_blocks": 4}, "max capacity combo"),
        _R("d=192 + dropouts low", {"attention_dropout": 0.1, "ffn_dropout": 0.05},
           "tuned dropout per §A.3 best-zone"),
        _R("seed 42 best-zone", {"d_block": 256, "n_blocks": 4, "lr": 1e-4, "seed": 42},
           "alt seed at best zone"),
        _R("seed 7 best-zone", {"d_block": 256, "n_blocks": 4, "lr": 1e-4, "seed": 7},
           "alt seed 2 at best zone"),
        _R("seed 123 best-zone", {"d_block": 256, "n_blocks": 4, "lr": 1e-4, "seed": 123},
           "alt seed 3 at best zone (winner rerun)"),
    ],

    # ---------------- MLP-PLR (Gorishniy 2022 ICLR arXiv:2203.05556) ----
    "mlp_plr": [
        _R("paper Higgs default plr=48x64 h=512x3", {},
           "MLP-PLR paper-default per arXiv:2203.05556 Tab.4: PLR n_frequencies=48 d_embedding=64, hidden [512]×3, dropout 0.1, AdamW lr 1e-3, batch 4096"),
        _R("plr_n_freq=24 fewer", {"plr_n_frequencies": 24},
           "fewer PLR frequencies per §3.2 ablation"),
        _R("plr_n_freq=96 more", {"plr_n_frequencies": 96},
           "more PLR frequencies per §3.2 ablation"),
        _R("plr_d_emb=32 narrower", {"plr_d_embedding": 32},
           "narrower PLR embedding per §3.2"),
        _R("plr_d_emb=128 wider", {"plr_d_embedding": 128},
           "wider PLR embedding per §3.2"),
        _R("plr_lite=True", {"plr_lite": True},
           "PLR-Lite (shared frequencies) per §3.2"),
        _R("hidden=256x3 narrower", {"hidden": [256, 256, 256]},
           "narrower MLP head per §3.3"),
        _R("hidden=1024x3 wider", {"hidden": [1024, 1024, 1024]},
           "wider MLP head per §3.3"),
        _R("hidden=512x6 deeper", {"hidden": [512] * 6},
           "deeper MLP head per §3.3"),
        _R("dropout=0.0", {"dropout": 0.0}, "no dropout per §3.3 ablation"),
        _R("dropout=0.3", {"dropout": 0.3}, "stronger dropout per §3.3"),
        _R("lr=5e-4 finer", {"lr": 5e-4}, "finer lr per §A"),
        _R("lr=3e-3 coarser", {"lr": 3e-3}, "coarser lr per §A"),
        _R("weight_decay=1e-4", {"weight_decay": 1e-4}, "L2 reg per §A"),
        _R("weight_decay=1e-3", {"weight_decay": 1e-3}, "stronger L2 per §A"),
        _R("batch=2048", {"batch_size": 2048}, "smaller batch per §A"),
        _R("batch=8192", {"batch_size": 8192}, "larger batch per §A"),
        _R("plr=96x128 + 1024x3", {"plr_n_frequencies": 96, "plr_d_embedding": 128, "hidden": [1024] * 3},
           "max capacity PLR + MLP per §3.4"),
        _R("plr=48x64 + 512x6", {"hidden": [512] * 6}, "deeper MLP, default PLR"),
        _R("plr=48x64 + dropout=0.2 wd=1e-4", {"dropout": 0.2, "weight_decay": 1e-4},
           "tuned reg combo per §A"),
        _R("plr_lite + 1024x3", {"plr_lite": True, "hidden": [1024] * 3},
           "PLR-Lite at high capacity"),
        _R("hidden=1024x3 + lr=5e-4", {"hidden": [1024] * 3, "lr": 5e-4},
           "wider + finer lr"),
        _R("seed 42 best-zone", {"hidden": [1024] * 3, "lr": 5e-4, "weight_decay": 1e-4, "seed": 42},
           "alt seed at best zone"),
        _R("seed 7 best-zone", {"hidden": [1024] * 3, "lr": 5e-4, "weight_decay": 1e-4, "seed": 7},
           "alt seed 2 at best zone"),
        _R("seed 123 best-zone", {"hidden": [1024] * 3, "lr": 5e-4, "weight_decay": 1e-4, "seed": 123},
           "alt seed 3 at best zone (winner rerun)"),
    ],

    # ---------------- ResNet-tabular (Gorishniy 2021 NeurIPS arXiv:2106.11189) ----
    "resnet_tabular": [
        _R("paper Higgs default 2blk d=256 mult=2", {},
           "ResNet paper-default per arXiv:2106.11189 Tab.6: n_blocks=2 d_block=256 d_hidden_multiplier=2 dropout1=0.25"),
        _R("n_blocks=4", {"n_blocks": 4}, "deeper per §A.2"),
        _R("n_blocks=6", {"n_blocks": 6}, "deepest per §A.2"),
        _R("d_block=128", {"d_block": 128}, "narrower per §A.2"),
        _R("d_block=384", {"d_block": 384}, "wider per §A.2"),
        _R("d_block=512", {"d_block": 512}, "widest per §A.2"),
        _R("d_hidden_mult=1", {"d_hidden_multiplier": 1.0}, "narrower hidden per §A.2"),
        _R("d_hidden_mult=4", {"d_hidden_multiplier": 4.0}, "wider hidden per §A.2"),
        _R("dropout1=0.0", {"dropout1": 0.0}, "no first dropout"),
        _R("dropout1=0.5", {"dropout1": 0.5}, "high first dropout"),
        _R("dropout2=0.1", {"dropout2": 0.1}, "add second dropout per §A.2"),
        _R("lr=5e-4", {"lr": 5e-4}, "finer lr per §A.3"),
        _R("lr=3e-3", {"lr": 3e-3}, "coarser lr per §A.3"),
        _R("weight_decay=1e-4", {"weight_decay": 1e-4}, "L2 reg per §A.3"),
        _R("weight_decay=1e-3", {"weight_decay": 1e-3}, "stronger L2 per §A.3"),
        _R("batch=2048", {"batch_size": 2048}, "smaller batch per §A.3"),
        _R("batch=8192", {"batch_size": 8192}, "larger batch per §A.3"),
        _R("4blk + d=384", {"n_blocks": 4, "d_block": 384}, "deeper + wider combo"),
        _R("6blk + d=512", {"n_blocks": 6, "d_block": 512}, "deepest + widest combo"),
        _R("4blk + d_hidden_mult=4", {"n_blocks": 4, "d_hidden_multiplier": 4.0},
           "deeper + wider FFN combo"),
        _R("4blk + dropout1=0.5 wd=1e-4", {"n_blocks": 4, "dropout1": 0.5, "weight_decay": 1e-4},
           "tuned reg combo per §A.3"),
        _R("4blk + d=384 + lr=5e-4", {"n_blocks": 4, "d_block": 384, "lr": 5e-4},
           "best-zone capacity + lr"),
        _R("seed 42 best-zone", {"n_blocks": 4, "d_block": 384, "lr": 5e-4, "weight_decay": 1e-4, "seed": 42},
           "alt seed at best zone"),
        _R("seed 7 best-zone", {"n_blocks": 4, "d_block": 384, "lr": 5e-4, "weight_decay": 1e-4, "seed": 7},
           "alt seed 2 at best zone"),
        _R("seed 123 best-zone", {"n_blocks": 4, "d_block": 384, "lr": 5e-4, "weight_decay": 1e-4, "seed": 123},
           "alt seed 3 at best zone (winner rerun)"),
    ],

    # ===================================================================
    # LEGACY / BASELINE TIER (kept for cross-tier comparison only)
    # ===================================================================

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
