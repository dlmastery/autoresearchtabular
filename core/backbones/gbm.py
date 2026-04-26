"""GBM backbones: lightgbm + xgboost + catboost (3 separate registry entries
per CLAUDE.md Tier-3 rule)."""
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from .registry import register_backbone


class _GBMBase:
    """Common interface for GBM wrappers."""
    framework: str = ""

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Any]:
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError


@register_backbone("lightgbm")
class LightGBM(_GBMBase):
    framework = "lightgbm"

    def __init__(self,
                 n_estimators: int = 1000, num_leaves: int = 63,
                 learning_rate: float = 0.05,
                 feature_fraction: float = 0.8, bagging_fraction: float = 0.8,
                 bagging_freq: int = 5, min_data_in_leaf: int = 20,
                 reg_alpha: float = 0.0, reg_lambda: float = 1.0,
                 max_depth: int = -1, n_jobs: int = 4,
                 early_stopping_rounds: int = 50, **_):
        self.params = dict(
            n_estimators=n_estimators, num_leaves=num_leaves,
            learning_rate=learning_rate, feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction, bagging_freq=bagging_freq,
            min_data_in_leaf=min_data_in_leaf, reg_alpha=reg_alpha,
            reg_lambda=reg_lambda, max_depth=max_depth, n_jobs=n_jobs,
            objective="binary", metric="auc", verbose=-1,
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import lightgbm as lgb
        params = dict(self.params)
        n_estimators = params.pop("n_estimators")
        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set) if X_val is not None else None
        callbacks = []
        if self.early_stopping_rounds and val_set is not None:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
        callbacks.append(lgb.log_evaluation(period=0))
        self.model = lgb.train(
            params, train_set, num_boost_round=n_estimators,
            valid_sets=[val_set] if val_set else None,
            callbacks=callbacks,
        )
        return {"best_iteration": int(self.model.best_iteration or n_estimators),
                "best_score": dict(self.model.best_score) if self.model.best_score else {}}

    def predict_proba(self, X):
        return np.asarray(self.model.predict(X))

    def save(self, path: str) -> None:
        self.model.save_model(path)


@register_backbone("xgboost")
class XGBoost(_GBMBase):
    framework = "xgboost"

    def __init__(self, n_estimators: int = 1000, max_depth: int = 6,
                 learning_rate: float = 0.05, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, min_child_weight: float = 1,
                 reg_alpha: float = 0.0, reg_lambda: float = 1.0, gamma: float = 0.0,
                 n_jobs: int = 4, tree_method: str = "hist",
                 early_stopping_rounds: int = 50, **_):
        self.params = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda, gamma=gamma,
            n_jobs=n_jobs, tree_method=tree_method,
            objective="binary:logistic", eval_metric="auc",
            early_stopping_rounds=early_stopping_rounds, verbosity=0,
        )
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from xgboost import XGBClassifier
        self.model = XGBClassifier(**self.params)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        return {"best_iteration": int(getattr(self.model, "best_iteration", -1))}

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        self.model.save_model(path)


@register_backbone("catboost")
class CatBoost(_GBMBase):
    framework = "catboost"

    def __init__(self, iterations: int = 1000, depth: int = 6,
                 learning_rate: float = 0.05, l2_leaf_reg: float = 3.0,
                 bagging_temperature: float = 1.0, random_strength: float = 1.0,
                 thread_count: int = 4, early_stopping_rounds: int = 100, **_):
        self.params = dict(
            iterations=iterations, depth=depth, learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg, bagging_temperature=bagging_temperature,
            random_strength=random_strength, thread_count=thread_count,
            loss_function="Logloss", eval_metric="AUC",
            early_stopping_rounds=early_stopping_rounds, verbose=0,
        )
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from catboost import CatBoostClassifier
        self.model = CatBoostClassifier(**self.params)
        eval_set = (X_val, y_val) if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        return {"best_iteration": int(self.model.get_best_iteration() or self.params["iterations"])}

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        self.model.save_model(path)
