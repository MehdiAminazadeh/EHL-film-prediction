# file: pipelines/train_strategies/train_tabnet.py
from __future__ import annotations
from typing import Dict, Any
import time
import copy
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import KFold

from pipelines.make_pipeline import make_pipeline
from utils.optuna_search import run_optuna_search, _fit_with_es_pipeline  # ← reuse ES helper
from utils.metrics import (
    evaluate_metrics,
    aggregate_cv_metrics,
    r2_gap_overfitting_label,
)

# ---------- helpers  ----------

def _evaluate_cv(
    pipe,
    X_df: pd.DataFrame,
    y: np.ndarray,
    k: int = 5,
    patience: int | None = None,
) -> Dict[str, float]:
    """
    K-fold evaluation using metrics.py:
      - averages TRAIN metrics across folds (R2 only for reporting)
      - averages VAL metrics across folds (R2/MAE/RMSE/MAPE)
      - uses the same early-stopping logic as HPO when patience>0
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    train_metrics_list, val_metrics_list = [], []

    es_enabled = bool(patience and int(patience) > 0)
    es_rounds = int(patience or 0)

    for tr_idx, va_idx in kf.split(X_df):
        Xtr, Xv = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        ytr, yv = y[tr_idx], y[va_idx]

        pipe_fold = copy.deepcopy(pipe)

        # use the same helper as in Optuna search (handles pipeline namespacing)
        _fit_with_es_pipeline(
            pipe_fold,
            Xtr, ytr,
            Xv,  yv,
            enabled=es_enabled,
            rounds=es_rounds,
            final_step_name="model",
        )

        ytr_hat = pipe_fold.predict(Xtr)
        yv_hat  = pipe_fold.predict(Xv)

        train_metrics_list.append(evaluate_metrics(ytr, ytr_hat))
        val_metrics_list.append(evaluate_metrics(yv,  yv_hat))

    train_avg = aggregate_cv_metrics(train_metrics_list)
    val_avg   = aggregate_cv_metrics(val_metrics_list)

    return {
        "Train_R2": float(train_avg["R2"]),
        "CV_R2":    float(val_avg["R2"]),
        "MAE":      float(val_avg["MAE"]),
        "RMSE":     float(val_avg["RMSE"]),
        "MAPE":     float(val_avg["MAPE"]),
    }


# ---------- strategy ----------

def train_tabnet(
    *, cfg, df: pd.DataFrame, logger
) -> Dict[str, Any]:
    """
    TabNet strategy:
      - HPO via generic Optuna engine (arrays path)
      - Build preprocessing+TabNet pipeline (force scaling)
      - K-Fold CV evaluation with early stopping (patience from config)
      - Log and return standardized row
    """
    t0 = time.time()

    target = cfg.data["target_column"]
    X_df = df.drop(columns=[target])
    y = df[target].values

    model_name: str = "tabnet"
    model_space: Dict[str, Any] = cfg.model
    optuna_cfg: Dict[str, Any] = cfg.optuna
    train_cfg:  Dict[str, Any] = cfg.training

    # 1) Optional HPO
    best_params: Dict[str, Any] = {}
    n_trials = int(optuna_cfg.get("n_trials", 0))
    if n_trials > 0:
        best_params = run_optuna_search(
            model_name=model_name,
            model_space=model_space,
            optuna_cfg=optuna_cfg,
            training_cfg=train_cfg,
            X=X_df.values,   # arrays path is fine
            y=y,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    # 2) Final pipeline (defaults + best_params)
    defaults = {k: v for k, v in model_space.items() if not isinstance(v, dict)}
    final_params = {**defaults, **(best_params or {})}

    # normalize alias from config
    if "lr" in final_params and "learning_rate" not in final_params:
        final_params["learning_rate"] = final_params.pop("lr")

    pipe = make_pipeline(
        model_name,
        final_params,
        df_train=df,
        target_col=target,
        preproc_cfg={
            "force_scale": True,   # match your original TabNet script behavior
            "ohe_sparse": False,   # dense arrays for TabNet
        },
        target_scaling=False,
    )

    # 3) CV evaluation with ES (same patience used in HPO)
    k = int(optuna_cfg.get("kfold_splits", 5))
    patience = int((train_cfg.get("early_stopping", {}) or {}).get("patience", 0))
    cv = _evaluate_cv(pipe, X_df, y, k=k, patience=patience)

    # 4) Overfitting assessment
    r2_gap, overfit = r2_gap_overfitting_label(cv["Train_R2"], cv["CV_R2"])

    row = {
        "Model": model_name.upper(),
        "Train_R2": round(cv["Train_R2"], 4),
        "CV_R2":    round(cv["CV_R2"], 4),
        "R2_Gap":   round(r2_gap, 4),
        "Overfitting": overfit,
        "MAE":  round(cv["MAE"], 4),
        "RMSE": round(cv["RMSE"], 4),
        "MAPE": round(cv["MAPE"], 4),
        "Time": round(time.time() - t0, 2),
        "Best_Params": best_params or {},
    }

    if logger:
        logger.append_row(row)

    return row
