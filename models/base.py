# base.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Protocol

@dataclass(frozen=True)
class Capabilities:
    backend: str                    # "sklearn" | "boosting"
    supports_early_stopping: bool = False
    supports_gpu: bool = False
    supports_feature_importance: bool = False
    handles_categoricals: bool = False
    deterministic: bool = True
    notes: str = ""

class BaseRegressor(Protocol):
    def fit(self, X, y, **kwargs: Any) -> "BaseRegressor": ...
    def predict(self, X, **kwargs: Any): ...
    def get_params(self, deep: bool = True) -> Dict[str, Any]: ...
    def set_params(self, **params: Any) -> "BaseRegressor": ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "BaseRegressor": ...
    @classmethod
    def capabilities(cls) -> Capabilities: ...
