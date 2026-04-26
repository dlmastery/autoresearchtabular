"""sklearn baselines: LogisticRegression + RandomForest."""
from __future__ import annotations
import numpy as np
from .registry import register_backbone


@register_backbone("logistic_regression")
class LR:
    framework = "sklearn"

    def __init__(self, C: float = 1.0, max_iter: int = 1000,
                 solver: str = "lbfgs", n_jobs: int = 4,
                 penalty: str = "l2", **_):
        self.params = dict(C=C, max_iter=max_iter, solver=solver,
                           n_jobs=n_jobs, penalty=penalty, random_state=0)
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(**self.params)
        self.model.fit(X_train, y_train)
        return {}

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self.model, path)


@register_backbone("random_forest")
class RF:
    framework = "sklearn"

    def __init__(self, n_estimators: int = 500, max_depth: int = 20,
                 max_features: str = "sqrt", n_jobs: int = 4,
                 random_state: int = 0, **_):
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth,
                           max_features=max_features, n_jobs=n_jobs,
                           random_state=random_state)
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_train, y_train)
        return {}

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self.model, path)
