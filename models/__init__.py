# file: models/__init__.py
from __future__ import annotations
from typing import Any, Dict
from .registry import create, register_model, get_class, get_role, list_models, known_models

def get_model(name: str, params: Dict[str, Any]):
    return create(name, **(params or {}))

# IMPORTANT: import modules that register themselves
from .dnn_adapter import DNNAdapter  # registers "dnn"
from .tabnet_adapter import TabNetSklearnAdapter
from .node_adapter import NodeSklearnAdapter
from .ft_transformer_adapter import FTTransformerSklearnAdapter

__all__ = [
    "get_model",
    "create", "register_model", "get_class", "get_role", "list_models", "known_models",
    "DNNAdapter","TabNetSklearnAdapter", "NodeSklearnAdapter", "FTTransformerSklearnAdapter",
]
