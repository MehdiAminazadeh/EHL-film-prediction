from __future__ import annotations

import io
import re
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

PX_THEME = "plotly_dark"


def _numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).copy()


def _metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def _available_models():
    return {
        "LinearRegression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "Ridge": Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]
        ),
        "Lasso": Pipeline(
            [("scaler", StandardScaler()), ("model", Lasso(alpha=0.01, max_iter=10000))]
        ),
        "RandomForest": Pipeline(
            [("model", RandomForestRegressor(n_estimators=400, random_state=42))]
        ),
        "SVR (RBF)": Pipeline(
            [("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=10.0, gamma="scale"))]
        ),
    }


def _find_average_film_column(columns):
    pattern = re.compile(r"\baverage\s*film\b", re.IGNORECASE)
    for column in columns:
        if pattern.search(str(column)):
            return column
    return None


def guidance_text(cv_folds: int, test_size: float, n: int) -> str:
    train_n = int((1 - test_size) * n)
    lines = [
        f"• Samples: {n}   → train ≈ {train_n}, test ≈ {n - train_n}",
        f"• CV folds = {cv_folds} → each fold trains on ≈ {(cv_folds - 1) / cv_folds:.0%} of training data and validates on ≈ {1 / cv_folds:.0%}.",
        "• Higher CV folds → better generalization estimate, more compute.",
        "• Larger test size → stronger final evaluation, less training data.",
        "• With small data, prefer lower test size (0.15–0.2) and moderate folds (5).",
    ]
    return "\n".join(lines)


def render_ml_panel(df_clean: pd.DataFrame):
    st.markdown("### Machine Learning — Predict Average Film")

    if df_clean is None or df_clean.empty:
        st.info("No cleaned data yet.")
        return

    numeric_df = _numeric_only(df_clean)

    banned_patterns = [
        r"\border\b",
        r"\bavg(?:erage)?\s*wavelength\b",
        r"\baverage\s*wavelength\b",
        r"\blube\s*ref\s*index\b",
    ]
    numeric_df = numeric_df.drop(
        columns=[
            c
            for c in numeric_df.columns
            if any(re.search(pattern, str(c), re.IGNORECASE) for pattern in banned_patterns)
        ],
        errors="ignore",
    )

    if numeric_df.shape[1] < 2:
        st.warning("Need at least 2 numeric columns.")
        return

    auto_target = _find_average_film_column(list(numeric_df.columns))
    manual_mode = auto_target is None

    left, right = st.columns([1.25, 1.75], gap="large")

    with left:
        if manual_mode:
            st.warning("Couldn’t auto-detect ‘Average Film’. Please select the target.")
            target = st.selectbox("Target", list(numeric_df.columns))
        else:
            target = auto_target
            st.write(f"**Target:** `{target}` (auto-detected)")

        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("CV folds", 3, 10, 5, 1)
        model_name = st.selectbox("Model", list(_available_models().keys()), index=3)
        do_perm = st.checkbox("Permutation importance", True)
        run_clicked = st.button("Train & Evaluate", use_container_width=True)

    with right:
        st.caption(
            "Training uses numeric features only; any ‘Average Film’ columns are removed from features to avoid leakage."
        )
        st.markdown("#### What your settings mean")
        st.code(guidance_text(cv_folds, test_size, numeric_df.shape[0]), language="text")

    if not run_clicked:
        return

    def _normalize_name(col_name: str) -> str:
        normalized = unicodedata.normalize("NFKC", str(col_name)).replace("\xa0", " ")
        return re.sub(r"\s+", " ", normalized).strip().lower()

    def _is_banned_feature(col_name: str) -> bool:
        normalized = _normalize_name(col_name)
        has_order = "order" in normalized
        has_avg_wl = "average wavelength" in normalized or "avg wavelength" in normalized
        has_lube = "lube ref index" in normalized
        return has_order or has_avg_wl or has_lube

    banned_in_pool = [c for c in numeric_df.columns if _is_banned_feature(c)]
    if banned_in_pool:
        numeric_df = numeric_df.drop(columns=banned_in_pool, errors="ignore")

    feature_df = numeric_df.drop(columns=[target], errors="ignore")

    hidden_target_cols = [
        c for c in feature_df.columns if "average film" in _normalize_name(c)
    ]
    if hidden_target_cols:
        feature_df = feature_df.drop(columns=hidden_target_cols, errors="ignore")

    final_banned = [c for c in feature_df.columns if _is_banned_feature(c)]
    if final_banned:
        feature_df = feature_df.drop(columns=final_banned, errors="ignore")

    excluded = sorted(set(banned_in_pool + hidden_target_cols + final_banned))
    if excluded:
        st.caption("Excluded from ML features: " + ", ".join(excluded))

    assert not any(_is_banned_feature(c) for c in feature_df.columns), f"BANNED FEATURE LEAKED: {list(feature_df.columns)}"

    if feature_df.shape[1] == 0:
        st.error("No features left after exclusions.")
        return

    X = feature_df.values
    y = numeric_df[target].values

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    pipe = _available_models()[model_name]

    scoring = {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "r2": "r2",
    }
    cv_result = cross_validate(pipe, X, y, cv=kf, scoring=scoring, return_train_score=False)
    cv_table = pd.DataFrame(
        {
            "MAE (cv)": -cv_result["test_mae"],
            "RMSE (cv)": np.sqrt(-cv_result["test_mse"]),
            "R2 (cv)": cv_result["test_r2"],
        }
    )

    st.markdown("**Cross-validation metrics**")
    summary_df = cv_table.describe().loc[["mean", "std"]].T
    st.dataframe(summary_df, use_container_width=True)

    buffer = io.StringIO()
    summary_df.to_csv(buffer)
    st.download_button("Download CV summary", buffer.getvalue().encode(), "cv_summary.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    st.markdown("**Holdout metrics**")
    st.dataframe(pd.Series(_metrics(y_test, y_pred), name="value").to_frame(), use_container_width=True)

    figure = go.Figure()
    figure.add_scatter(x=y_test, y=y_pred, mode="markers", name="pred")
    lo = float(np.nanmin(y_test))
    hi = float(np.nanmax(y_test))
    figure.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="ideal"))
    figure.update_layout(
        template=PX_THEME,
        height=360,
        title="Predicted vs Actual — Average Film",
        xaxis_title="Actual",
        yaxis_title="Predicted",
    )
    st.popover("Show Predicted vs Actual").write(st.plotly_chart(figure, use_container_width=True))

    residuals = y_test - y_pred
    residual_figure = px.histogram(
        residuals,
        nbins=40,
        template=PX_THEME,
        title="Residuals histogram",
    )
    residual_figure.update_layout(height=320, xaxis_title="Residual", yaxis_title="Count")
    st.popover("Show Residuals histogram").write(
        st.plotly_chart(residual_figure, use_container_width=True)
    )

    residual_vs_fitted = go.Figure()
    residual_vs_fitted.add_scatter(x=y_pred, y=residuals, mode="markers")
    residual_vs_fitted.add_hline(y=0)
    residual_vs_fitted.update_layout(
        template=PX_THEME,
        height=360,
        title="Residuals vs Fitted",
        xaxis_title="Fitted",
        yaxis_title="Residual",
    )
    st.popover("Show Residuals vs Fitted").write(
        st.plotly_chart(residual_vs_fitted, use_container_width=True)
    )

    train_sizes, train_scores, test_scores = learning_curve(
        pipe, X, y, cv=kf, scoring="r2", train_sizes=np.linspace(0.2, 1.0, 5)
    )
    lc_fig = go.Figure()
    lc_fig.add_scatter(
        x=train_sizes,
        y=train_scores.mean(axis=1),
        mode="lines+markers",
        name="Train R²",
    )
    lc_fig.add_scatter(
        x=train_sizes,
        y=test_scores.mean(axis=1),
        mode="lines+markers",
        name="CV R²",
    )
    lc_fig.update_layout(
        template=PX_THEME,
        height=360,
        title="Learning curve (R²)",
        xaxis_title="Training samples",
        yaxis_title="R²",
    )
    st.popover("Show Learning Curve").write(st.plotly_chart(lc_fig, use_container_width=True))

    with st.expander("Feature importance", expanded=False):
        feature_names = list(feature_df.columns)

        if model_name in ["LinearRegression", "Ridge", "Lasso"]:
            coefs = pipe.named_steps["model"].coef_
            importances = (
                pd.Series(coefs, index=feature_names)
                .sort_values(key=np.abs, ascending=False)
                .to_frame("coefficient")
            )
            st.dataframe(importances, use_container_width=True, height=260)
        elif model_name == "RandomForest":
            forest = pipe.named_steps["model"]
            importances = (
                pd.Series(forest.feature_importances_, index=feature_names)
                .sort_values(ascending=False)
                .to_frame("rf_importance")
            )
            st.dataframe(importances, use_container_width=True, height=260)
        else:
            st.caption("SVR has no direct coefficients; using permutation importance below.")

        permutation_result = permutation_importance(
            pipe, X_test, y_test, n_repeats=10, random_state=42, scoring="r2"
        )
        permutation_series = pd.Series(
            permutation_result.importances_mean, index=feature_names
        ).sort_values(ascending=False)

        perm_fig = go.Figure()
        perm_fig.add_bar(
            y=permutation_series.index[::-1],
            x=permutation_series.values[::-1],
            orientation="h",
        )
        perm_fig.update_layout(template=PX_THEME, height=360, title="Permutation importance")
        st.plotly_chart(perm_fig, use_container_width=True)
