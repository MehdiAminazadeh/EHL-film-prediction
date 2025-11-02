from __future__ import annotations
import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Top-level keys we expect to exist in the YAML
REQUIRED_TOP = ["data", "experiment", "logging", "models"]
# 'optuna' and 'training' are optional globally; if absent we treat them as {}


class ConfigError(Exception):
    pass


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config:
    """
    Loads YAML, validates, and exposes:
      • data, experiment, logging (globals)
      • active: merged block for the chosen model
        {
          model_name: str,
          model: Dict (per-model param space / fixed params),
          optuna: Dict (global optuna merged with per-model optuna),
          training: Dict (global training merged with per-model training),
        }
      • convenience attrs (resolved): model_name, model, optuna, training
    """

    def __init__(self, raw: Dict[str, Any]):
        self._raw = raw or {}
        self._validate()

        # globals
        self.data: Dict[str, Any] = self._raw["data"]
        self.experiment: Dict[str, Any] = self._raw["experiment"]
        self.logging: Dict[str, Any] = self._raw["logging"]
        self._global_optuna: Dict[str, Any] = self._raw.get("optuna", {}) or {}
        self._global_training: Dict[str, Any] = self._raw.get("training", {}) or {}
        self._models: Dict[str, Any] = self._raw["models"] or {}

        # resolve model name (support both locations)
        model_name = (
            (self.experiment.get("model_name") or "")
            or (self._raw.get("model", {}) or {}).get("name", "")
        ).lower().strip()

        if not model_name:
            raise ConfigError(
                "No model selected. Provide either 'experiment.model_name' or 'model.name' in the config."
            )

        if model_name not in self._models:
            raise ConfigError(f"models.{model_name} block missing in config.")

        # initialize active block and resolved attrs
        self._set_active(model_name)

    # -----------------------------
    # internal helpers
    # -----------------------------
    def _validate(self) -> None:
        for key in REQUIRED_TOP:
            if key not in self._raw:
                raise ConfigError(f"Missing top-level key: {key}")
        if "csv_path" not in self._raw["data"]:
            raise ConfigError("data.csv_path is required.")
        if "target_column" not in self._raw["data"]:
            raise ConfigError("data.target_column is required.")

    def _set_active(self, model_name: str) -> None:
        model_block = self._models.get(model_name, {}) or {}

        self.active: Dict[str, Any] = {
            "model_name": model_name,
            "model": model_block.get("model", {}) or {},
            "optuna": _deep_merge(self._global_optuna, model_block.get("optuna", {}) or {}),
            "training": _deep_merge(self._global_training, model_block.get("training", {}) or {}),
        }

        # convenience (resolved) attributes
        self.model_name: str = self.active["model_name"]
        self.model: Dict[str, Any] = self.active["model"]
        self.optuna: Dict[str, Any] = self.active["optuna"]
        self.training: Dict[str, Any] = self.active["training"]

    # -----------------------------
    # public API
    # -----------------------------
    def switch_model(self, model_name: str) -> None:
        """Switch the active model programmatically (e.g., CLI override)."""
        model_name = (model_name or "").lower().strip()
        if model_name not in self._models:
            raise ConfigError(f"models.{model_name} block missing in config.")
        # also mirror in experiment for reproducibility
        self.experiment["model_name"] = model_name
        self._set_active(model_name)

    def dump_effective(self) -> str:
        """Pretty JSON of effective config (handy for logs/repro)."""
        blob = {
            "data": self.data,
            "experiment": self.experiment,
            "logging": self.logging,
            "active": self.active,
        }
        return json.dumps(blob, indent=2)

    @property
    def raw(self) -> Dict[str, Any]:
        return self._raw


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config(raw)
