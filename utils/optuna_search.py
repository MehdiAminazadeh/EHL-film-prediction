# utils/optuna_search.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import inspect
import numpy as np
import optuna

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_is_fitted

from models import get_model
from pipelines.make_pipeline import make_pipeline

# ============================
# Param sampling helpers
# ============================

def _sample_param(trial: optuna.Trial, key: str, spec: Any):
    """
    spec can be:
      - literal (fixed value) -> returned as-is
      - dict with one of:
          * choices: [...]
          * min/max (+ optional step / log)
    """
    if not isinstance(spec, dict):
        return spec

    if "choices" in spec:
        return trial.suggest_categorical(key, spec["choices"])

    low = spec.get("min", None)
    high = spec.get("max", None)
    if low is None or high is None:
        raise ValueError(f"Param '{key}' spec needs either 'choices' or both 'min' and 'max'.")

    if spec.get("log", False):
        return trial.suggest_float(key, float(low), float(high), log=True)

    step = spec.get("step", None)
    if step is not None:
        is_int = isinstance(step, int) and isinstance(low, int) and isinstance(high, int)
        if is_int:
            return trial.suggest_int(key, int(low), int(high), step=int(step))
        return trial.suggest_float(key, float(low), float(high), step=float(step))

    return trial.suggest_float(key, float(low), float(high))


def _extract_fixed_defaults(model_space: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only fixed (non-dict) values from the model_space block."""
    return {k: v for k, v in (model_space or {}).items() if not isinstance(v, dict)}


# ============================
# Early stopping helpers
# ============================

def _es_config(training_cfg: Dict[str, Any]) -> Tuple[bool, int, float]:
    """
    Read ES config. Returns (enabled, rounds, min_delta).
    We map `patience` -> rounds for booster-style APIs.
    """
    es = (training_cfg or {}).get("early_stopping", {}) or {}
    enabled = bool(es.get("enabled", False))
    rounds = int(es.get("rounds", es.get("patience", 0)))
    min_delta = float(es.get("min_delta", 0.0))  # reserved for custom loops
    if enabled and rounds <= 0:
        rounds = 50
    return enabled, rounds, min_delta


def _accepts_param(func, name: str) -> bool:
    """Check if callable's signature accepts a given parameter name (or **kwargs)."""
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return False
    params = sig.parameters
    if name in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _fit_with_es_model(model, Xtr, ytr, Xv, yv, enabled: bool, rounds: int):
    """
    Fit a direct estimator with early stopping if supported.
    Always ends fitted (falls back to plain fit on any exception).
    """
    fit_fn = getattr(model, "fit")
    if not enabled:
        fit_fn(Xtr, ytr)
        return

    cand = {}
    if _accepts_param(fit_fn, "eval_set"):
        cand["eval_set"] = [(Xv, yv)]
    if _accepts_param(fit_fn, "early_stopping_rounds"):
        cand["early_stopping_rounds"] = rounds
    if _accepts_param(fit_fn, "verbose"):
        cand["verbose"] = False

    try:
        if cand:
            fit_fn(Xtr, ytr, **cand)
        else:
            fit_fn(Xtr, ytr)
    except Exception:
        # Final fallback
        fit_fn(Xtr, ytr)


def _fit_with_es_pipeline(pipe, Xtr, ytr, Xv, yv, enabled: bool, rounds: int, final_step_name: str = "model"):
    """
    Fit a sklearn Pipeline with ES if supported by the final estimator.
    Tries namespaced kwargs, then plain, then falls back to plain fit.
    Always ends fitted (verified with check_is_fitted).
    """
    if not enabled:
        pipe.fit(Xtr, ytr)
    else:
        # 1) namespaced attempt
        ns = {
            f"{final_step_name}__eval_set": [(Xv, yv)],
            f"{final_step_name}__early_stopping_rounds": rounds,
            f"{final_step_name}__verbose": False,
        }
        try:
            pipe.fit(Xtr, ytr, **ns)
        except Exception:
            # 2) plain attempt
            try:
                pipe.fit(Xtr, ytr, eval_set=[(Xv, yv)], early_stopping_rounds=rounds, verbose=False)
            except Exception:
                # 3) fallback: no ES
                pipe.fit(Xtr, ytr)

    # Safety: ensure fitted (prevents FutureWarning / future error in sklearn>=1.8)
    try:
        check_is_fitted(pipe)
    except Exception:
        pipe.fit(Xtr, ytr)


def _extract_best_iteration(obj) -> Optional[int]:
    """
    Try different conventions to get best iteration/tree/epoch.
    Returns None if not available.
    """
    for attr in ("best_iteration_", "best_iteration"):
        if hasattr(obj, attr):
            try:
                val = getattr(obj, attr)
                if isinstance(val, (int, np.integer)):
                    return int(val)
            except Exception:
                pass
    # CatBoost
    if hasattr(obj, "get_best_iteration"):
        try:
            return int(obj.get_best_iteration())
        except Exception:
            pass
    return None


# ============================
# Simple penalty helpers
# ============================

def _cfg_bool(d: Dict[str, Any], k: str, default=False) -> bool:
    return bool(d.get(k, default))


def _cfg_float(d: Dict[str, Any], k: str, default=0.0) -> float:
    try:
        return float(d.get(k, default))
    except Exception:
        return default


# ============================
# Core API
# ============================

def run_optuna_search(
    *,
    model_name: str,
    model_space: Dict[str, Any],
    optuna_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    # Arrays path (fallback / current usage)
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    # Pipeline path (recommended): if provided, overrides arrays path
    df_train=None,
    target_col: Optional[str] = None,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    preproc_cfg: Optional[Dict[str, Any]] = None,
    target_scaling: bool = False,
    # misc
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter search with K-Fold CV.
    - Early stopping for booster-style estimators when enabled in training_cfg.
    - Records early stop iteration per trial in user_attrs["early_stop_iter"] (max across folds).
    - Applies a SIMPLE overfitting penalty if configured:
        adjusted = val_R2 - alpha * max(0, (train_R2 - val_R2) - eps)
    """
    n_trials = int(optuna_cfg.get("n_trials", 0))
    if n_trials <= 0:
        return {}

    k_splits = int(optuna_cfg.get("kfold_splits", 5))
    direction = str(optuna_cfg.get("direction", "maximize")).lower()
    if direction not in {"maximize", "minimize"}:
        direction = "maximize"

    use_pipeline = df_train is not None and target_col is not None
    fixed_defaults = _extract_fixed_defaults(model_space)

    # Early stopping config
    es_enabled, es_rounds, _ = _es_config(training_cfg)

    # Simple penalty config
    use_of_pen = _cfg_bool(optuna_cfg, "use_overfit_penalty", False)
    of_alpha   = _cfg_float(optuna_cfg, "overfit_alpha", 5.0)
    of_margin  = _cfg_float(optuna_cfg, "overfit_margin", 0.02)

    def objective(trial: optuna.Trial):
        # 1) sample params
        sampled = {pname: _sample_param(trial, pname, spec) for pname, spec in (model_space or {}).items()}
        params = {**fixed_defaults, **sampled}

        best_es_iter_over_folds: Optional[int] = None

        if use_pipeline:
            # Prepare CV data
            kf = KFold(n_splits=k_splits, shuffle=True, random_state=42)
            X_df = df_train.drop(columns=[target_col])
            y_vec = df_train[target_col].values

            val_scores: List[float] = []
            train_scores: List[float] = []

            for tr_idx, va_idx in kf.split(X_df):
                Xtr, Xv = X_df.iloc[tr_idx], X_df.iloc[va_idx]
                ytr, yv = y_vec[tr_idx], y_vec[va_idx]

                # fresh pipeline each fold (no leakage)
                pipe = make_pipeline(
                    model_name,
                    params,
                    df_train=df_train,  # used inside pipeline if needed
                    target_col=target_col,
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                    preproc_cfg=preproc_cfg,
                    target_scaling=target_scaling,
                )

                _fit_with_es_pipeline(pipe, Xtr, ytr, Xv, yv, es_enabled, es_rounds, final_step_name="model")

                # ensure fitted (safety against future sklearn errors)
                try:
                    check_is_fitted(pipe)
                except Exception:
                    pipe.fit(Xtr, ytr)

                pred_val = pipe.predict(Xv)
                pred_tr  = pipe.predict(Xtr)

                val_scores.append(r2_score(yv, pred_val))
                train_scores.append(r2_score(ytr, pred_tr))

                # ES iteration (if available) — avoid named_steps; use steps[-1][1]
                try:
                    est = pipe.steps[-1][1] if hasattr(pipe, "steps") and pipe.steps else None
                    if est is not None:
                        es_iter = _extract_best_iteration(est)
                        if es_iter is not None:
                            best_es_iter_over_folds = (
                                es_iter if best_es_iter_over_folds is None
                                else max(best_es_iter_over_folds, es_iter)
                            )
                except Exception:
                    pass

            val_mean = float(np.mean(val_scores))
            train_mean = float(np.mean(train_scores))

        else:
            if X is None or y is None:
                raise ValueError("X/y must be provided when df_train/target_col are not used.")

            kf = KFold(n_splits=k_splits, shuffle=True, random_state=42)
            val_scores: List[float] = []
            train_scores: List[float] = []

            for tr_idx, va_idx in kf.split(X):
                Xtr, Xv = X[tr_idx], X[va_idx]
                ytr, yv = y[tr_idx], y[va_idx]

                # fresh model each fold (no leakage)
                model = get_model(model_name, params)
                _fit_with_es_model(model, Xtr, ytr, Xv, yv, es_enabled, es_rounds)

                pred_val = model.predict(Xv)
                pred_tr  = model.predict(Xtr)

                val_scores.append(r2_score(yv, pred_val))
                train_scores.append(r2_score(ytr, pred_tr))

                es_iter = _extract_best_iteration(model)
                if es_iter is not None:
                    best_es_iter_over_folds = (
                        es_iter if best_es_iter_over_folds is None
                        else max(best_es_iter_over_folds, es_iter)
                    )

            val_mean = float(np.mean(val_scores))
            train_mean = float(np.mean(train_scores))

        # ---- simple overfitting penalty ----
        gap = max(0.0, train_mean - val_mean)
        if use_of_pen:
            adjusted = val_mean - of_alpha * max(0.0, gap - of_margin)
        else:
            adjusted = val_mean

        # Attach diagnostics to the trial
        trial.set_user_attr("early_stop_iter", int(best_es_iter_over_folds) if best_es_iter_over_folds is not None else None)
        trial.set_user_attr("val_mean", val_mean)
        trial.set_user_attr("train_mean", train_mean)
        trial.set_user_attr("gap", gap)
        trial.set_user_attr("adjusted_score", adjusted)

        return adjusted if direction == "maximize" else -adjusted

    # Callback: concise ES/gap note per trial
    def _print_es_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        es_iter = trial.user_attrs.get("early_stop_iter", None)
        gap = trial.user_attrs.get("gap", None)
        if gap is None:
            print("  ↪ (no gap)")
            return
        if es_iter is not None:
            print(f"  ↪ ES best_iter={es_iter}; gap={gap:.3f}")
        else:
            print(f"  ↪ ES n/a; gap={gap:.3f}")

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=int(optuna_cfg.get("n_trials", 0)), show_progress_bar=False, callbacks=[_print_es_callback])

    return study.best_params if study.best_trial else {}
