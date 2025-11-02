from __future__ import annotations
import joblib
from typing import Any, Dict, Optional
from .base import BaseRegressor, Capabilities

class SklearnAdapter(BaseRegressor):
    """Wrap a sklearn-like estimator to match our BaseRegressor API."""
    _cls = None  # must be set by subclass

    def __init__(self, **params: Any):
        if self._cls is None:
            raise RuntimeError("SklearnAdapter subclass must set _cls")
        self._params = dict(params)
        self.model = self._cls(**params)

    def fit(self, X, y, **kwargs: Any):
        # pass-through kwargs so ES etc. work
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X, **kwargs: Any):
        return self.model.predict(X, **kwargs)


    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return self.model.get_params(deep=deep)

    def set_params(self, **params: Any):
        self._params.update(params)
        self.model.set_params(**params)
        return self

    def save(self, path: str) -> None:
        joblib.dump(self.model, path if path.endswith(".joblib") else path + ".joblib")

    @classmethod
    def load(cls, path: str) -> "SklearnAdapter":
        obj = cls()
        obj.model = joblib.load(path if path.endswith(".joblib") else path + ".joblib")
        obj._params = obj.model.get_params()
        return obj

    def get_feature_importance(self) -> Optional[Dict[str, Any]]:
        if hasattr(self.model, "feature_importances_"):
            return {"type": "importance", "values": self.model.feature_importances_}
        if hasattr(self.model, "coef_"):
            return {"type": "coef", "values": self.model.coef_}
        return None

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            backend="sklearn",
            supports_early_stopping=False,
            supports_gpu=False,
            supports_feature_importance=False,
            handles_categoricals=False,
            deterministic=True,
            notes="Generic sklearn adapter",
        )
