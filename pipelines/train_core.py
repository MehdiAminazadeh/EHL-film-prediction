# file: train_core.py
from __future__ import annotations
from typing import Dict, Any
import pandas as pd

# strategies
from .train_strategies.train_classical import train_classical
from .train_strategies.train_dnn import train_dnn
from .train_strategies.train_tabnet import train_tabnet
from .train_strategies.train_ft import train_ft_transformer
from .train_strategies.train_node import train_node

# map model registry names -> strategy group
STRATEGY_MAP = {
    # classical ML models
    "xgboost": "classical",
    "lightgbm": "classical",
    "catboost": "classical",
    "gradient_boosting": "classical",
    "random_forest": "classical",
    "extra_trees": "classical",
    "svr": "classical",
    "ridge": "classical",
    "lasso": "classical",
    "elasticnet": "classical",
    "kneighbors": "classical",
    # deep learning models
    "dnn": "dnn",
    "tabnet": "tabnet",
    "ft_transformer": "ft_transformer",
    "node": "node",
}

def train_model(
    *,
    cfg,                      # Config object from utils.config_parser
    data_cfg: Dict[str, Any], # cfg.data
    active: Dict[str, Any],   # cfg.active (merged block)
    df: pd.DataFrame,         # full dataframe with target col
    logger                    # utils.logger.ExperimentLogger
) -> Dict[str, Any]:
    """
    Route to the right training strategy by model name.
    Returns standardized result dict for logging.
    """
    model_name = active["model_name"].lower()
    group = STRATEGY_MAP.get(model_name)
    if group is None:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Known: {', '.join(sorted(STRATEGY_MAP.keys()))}"
        )

    if group == "classical":
        return train_classical(cfg=cfg, data_cfg=data_cfg, active=active, df=df, logger=logger)

    if group == "dnn":  
        return train_dnn(cfg=cfg, df=df, logger=logger)
    
    if group == "tabnet":
        return train_tabnet(cfg=cfg, df=df, logger=logger)
    
    if group == "ft_transformer":
        return train_ft_transformer(cfg=cfg, df=df, logger=logger)

    if group == "node":
        return train_node(cfg=cfg, df=df, logger=logger)

    # safety
    raise RuntimeError(f"No training strategy bound for group '{group}'")
