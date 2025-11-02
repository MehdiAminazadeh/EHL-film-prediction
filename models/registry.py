# file: models/registry.py
from __future__ import annotations
from typing import Dict, Type, Any, Optional, List, Literal, Callable, Union
from .base import BaseRegressor, Capabilities

# Role clarifies how the model is consumed by the pipeline
Role = Literal["estimator", "backbone"]

_REGISTRY: Dict[str, Type[Any]] = {}
_META: Dict[str, Dict[str, Any]] = {}   # includes role/tags/metadata
# Optional factory override for creation; useful when constructor is awkward
_FACTORIES: Dict[str, Callable[..., Any]] = {}


def register_model(
    name: str,
    *,
    role: Role = "estimator",
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    factory: Optional[Callable[..., Any]] = None,
):
    """
    Decorator to register a model class or backbone.

    Args:
        name: registration name (lowercased)
        role: "estimator" (sklearn-like / BaseRegressor) or "backbone" (e.g., torch nn.Module)
        tags: free-form tags, e.g. ["torch", "gpu", "tree_based"]
        metadata: any extra info you want to track
        factory: optional constructor override (callable) for create()

    Notes:
      • For role="estimator": class should implement BaseRegressor-like API
      • For role="backbone": class may be a raw module (e.g., nn.Module); training happens in a strategy
    """
    def deco(cls: Type[Any]):
        key = name.lower()
        if key in _REGISTRY:
            raise ValueError(f"Model '{name}' already registered")

        _REGISTRY[key] = cls
        _META[key] = {
            "role": role,
            "tags": tags or [],
            "metadata": metadata or {},
        }
        if factory is not None:
            _FACTORIES[key] = factory
        return cls
    return deco


def create(name: str, **params) -> Any:
    """
    Instantiate a registered entry.

    Behavior:
      • If a factory was provided, we call it(**params).
      • Else we call the registered class constructor directly: cls(**params)
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    if key in _FACTORIES:
        return _FACTORIES[key](**params)
    return _REGISTRY[key](**params)  # type: ignore


def get_class(name: str) -> Type[Any]:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'")
    return _REGISTRY[key]


def get_role(name: str) -> Role:
    key = name.lower()
    if key not in _META:
        raise ValueError(f"Unknown model '{name}'")
    return _META[key].get("role", "estimator")  # default for backward compat


def list_models() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, cls in _REGISTRY.items():
        caps: Optional[Capabilities] = None
        # Try to query capabilities if provided
        if hasattr(cls, "capabilities") and callable(getattr(cls, "capabilities")):
            try:
                caps = cls.capabilities()  # type: ignore
            except Exception:
                caps = None
        entry = {
            "role": _META.get(k, {}).get("role", "estimator"),
            "tags": _META.get(k, {}).get("tags", []),
            **_META.get(k, {}).get("metadata", {}),
        }
        if caps is not None:
            entry.update({
                "backend": caps.backend,
                "supports_early_stopping": caps.supports_early_stopping,
                "supports_gpu": caps.supports_gpu,
                "supports_feature_importance": caps.supports_feature_importance,
                "handles_categoricals": caps.handles_categoricals,
                "deterministic": caps.deterministic,
                "notes": caps.notes,
            })
        out[k] = entry
    return out


def known_models() -> List[str]:
    return sorted(_REGISTRY.keys())


KNOWN_MODELS = set(_REGISTRY.keys())
