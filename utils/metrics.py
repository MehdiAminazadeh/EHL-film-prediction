# file: utils/metrics.py

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _to_1d_numpy(x) -> np.ndarray:
    """Coerce arrays/Series/Lists to 1D float numpy arrays."""
    if x is None:
        raise ValueError("Input is None.")
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim > 1:
        arr = np.ravel(arr)
    return arr


def _align_and_filter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    drop_nan_inf: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure y_true/y_pred are same length and optionally drop NaN/Inf pairs."""
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"Length mismatch: y_true={y_true.shape[0]} vs y_pred={y_pred.shape[0]}")
    if not drop_nan_inf:
        return y_true, y_pred

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.all():
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    return y_true, y_pred


def evaluate_metrics(
    y_true_in,
    y_pred_in,
    *,
    r2_decimals: int = 4,
    mae_decimals: int = 4,
    rmse_decimals: int = 4,
    mape_decimals: int = 4,
    zero_epsilon: float = 1e-8,
    drop_nan_inf: bool = True,
) -> Dict[str, float]:
    """
    Compute MAE, RMSE, R2, MAPE with robust handling.

    Parameters
    ----------
    y_true_in, y_pred_in : array-like
        Ground truth and predictions (1D).
    zero_epsilon : float
        Small constant to avoid division by zero in MAPE.
    drop_nan_inf : bool
        If True, drop pairs where either value is NaN/Inf before scoring.

    Returns
    -------
    dict with keys: "MAE", "RMSE", "R2", "MAPE"
    """
    y_true = _to_1d_numpy(y_true_in)
    y_pred = _to_1d_numpy(y_pred_in)
    y_true, y_pred = _align_and_filter(y_true, y_pred, drop_nan_inf=drop_nan_inf)

    if y_true.size == 0:
        # Edge-case: nothing to score
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    denom = np.maximum(np.abs(y_true), zero_epsilon)  # safe MAPE
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

    return {
        "MAE": round(float(mae), mae_decimals),
        "RMSE": round(float(rmse), rmse_decimals),
        "R2": round(float(r2), r2_decimals),
        "MAPE": round(float(mape), mape_decimals),
    }


def r2_gap_overfitting_label(train_r2: float, cv_r2: float) -> Tuple[float, str]:
    """
    Compute R² gap and an overfitting label using your thresholds.
    Returns (gap, label).
    """
    gap = round(float(train_r2) - float(cv_r2), 4)
    if gap < 0.02:
        label = "No (Excellent)"
    elif gap < 0.05:
        label = "Low (Acceptable)"
    elif gap < 0.10:
        label = "Moderate (Needs Tuning)"
    else:
        label = "High (Likely Overfitting)"
    return gap, label


def aggregate_cv_metrics(metrics_list: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """
    Average a list of fold metrics dicts (from evaluate_metrics) into one dict.
    Ignores keys not in {MAE, RMSE, R2, MAPE}. Ignores NaNs safely.
    """
    keys = ("MAE", "RMSE", "R2", "MAPE")
    acc = {k: [] for k in keys}
    for m in metrics_list:
        if not m:
            continue
        for k in keys:
            v = m.get(k, np.nan)
            if np.isfinite(v):
                acc[k].append(float(v))

    out = {}
    for k in keys:
        out[k] = round(float(np.mean(acc[k])) if acc[k] else np.nan, 4)
    return out


def regression_report(
    y_true_in,
    y_pred_in,
    *,
    include: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Convenience wrapper around evaluate_metrics with optional field selection.
    """
    res = evaluate_metrics(y_true_in, y_pred_in, **kwargs)
    if include is None:
        return res
    return {k: res[k] for k in include if k in res}
