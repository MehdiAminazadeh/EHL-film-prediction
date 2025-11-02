# file: pipelines/train_strategies/train_classical.py
from __future__ import annotations
from typing import Dict, Any
import time
import copy
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from pipelines.make_pipeline import make_pipeline
from utils.optuna_search import run_optuna_search  # your generic HPO
from utils.metrics import (
    evaluate_metrics,
    aggregate_cv_metrics,
    r2_gap_overfitting_label,
)

# ---------- helpers ----------

def _evaluate_cv(pipe, X_df: pd.DataFrame, y: np.ndarray, k: int = 5) -> Dict[str, float]:
    """
    K-fold evaluation using metrics.py:
      - averages TRAIN metrics across folds (R2 only for reporting)
      - averages VAL metrics across folds (R2/MAE/RMSE/MAPE)
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    train_metrics_list = []
    val_metrics_list = []

    for tr_idx, va_idx in kf.split(X_df):
        Xtr, Xv = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        ytr, yv = y[tr_idx], y[va_idx]

        # fresh copy per fold
        pipe_fold = copy.deepcopy(pipe)

        pipe_fold.fit(Xtr, ytr)
        ytr_hat = pipe_fold.predict(Xtr)
        yv_hat  = pipe_fold.predict(Xv)

        # use unified metrics
        train_m = evaluate_metrics(ytr, ytr_hat)
        val_m   = evaluate_metrics(yv,  yv_hat)

        train_metrics_list.append(train_m)
        val_metrics_list.append(val_m)

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

def train_classical(
    *, cfg, data_cfg: Dict[str, Any], active: Dict[str, Any], df: pd.DataFrame, logger
) -> Dict[str, Any]:
    """
    Classical strategy:
      - Optional HPO via Optuna
      - Build final preprocessing+model pipeline
      - K-Fold CV evaluation
      - Log and return a standardized row
    """
    t0 = time.time()
    target = data_cfg["target_column"]
    X_df = df.drop(columns=[target])
    y = df[target].values

    model_name: str = active["model_name"].lower()
    model_space: Dict[str, Any] = active["model"]     # fixed or search-space specs
    optuna_cfg: Dict[str, Any] = active["optuna"]
    train_cfg:  Dict[str, Any] = active["training"]

    # 1) Optional hyperparameter search.
    best_params: Dict[str, Any] = {}
    n_trials = int(optuna_cfg.get("n_trials", 0))
    if n_trials > 0:
        best_params = run_optuna_search(
            model_name=model_name,
            model_space=model_space,
            optuna_cfg=optuna_cfg,
            training_cfg=train_cfg,
            X=X_df.values,   # HPO uses arrays; fine for classical
            y=y,
            device="cpu",
        )

    # 2) Final pipeline (defaults + best_params)
    defaults = {k: v for k, v in model_space.items() if not isinstance(v, dict)}
    final_params = {**defaults, **(best_params or {})}

    pipe = make_pipeline(
        model_name,
        final_params,
        df_train=df,
        target_col=target,
        preproc_cfg={"drop_unused_cats_for_trees": False},
        target_scaling=False,
    )

    # 3) CV evaluation (metrics.py)
    k = int(optuna_cfg.get("kfold_splits", 5))
    cv = _evaluate_cv(pipe, X_df, y, k=k)

    # 4) Overfitting assessment (metrics.py)
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
