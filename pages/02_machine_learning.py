import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from ui_shared import (
    topbar,
    inline_plotly_fig,
    guidance_text,
    find_avg_film,
    suitability_report,
)

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_validate,
    learning_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.inspection import permutation_importance


st.set_page_config(page_title="Machine Learning • EHL", layout="wide")
topbar("ML", show_sidebar=True)

st.markdown('<div class="wrap">', unsafe_allow_html=True)
st.title("Machine Learning — Predict Average Film")

df = st.session_state.get("last_clean_df")
if df is None or df.empty:
    st.info("Go to **Home** to upload and auto-clean first.")
    st.stop()

numeric_df = df.select_dtypes(include=[np.number]).copy()
if numeric_df.shape[1] < 2:
    st.warning("Need at least 2 numeric columns.")
    st.stop()


def available_models() -> dict[str, Pipeline]:
    return {
        "LinearRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "Lasso": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=0.01, max_iter=10000)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("model", RandomForestRegressor(n_estimators=400, random_state=42)),
            ]
        ),
        "SVR (RBF)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=10.0, gamma="scale")),
            ]
        ),
    }


target_column = find_avg_film(list(numeric_df.columns))
if target_column is None:
    target_column = st.selectbox(
        "Target (couldn’t auto-detect ‘Average Film’)",
        list(numeric_df.columns),
    )
else:
    st.write(f"**Target:** `{target_column}` (auto-detected)")

test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
cv_folds = st.slider("CV folds", 3, 10, 5, 1)
model_name = st.selectbox("Model", list(available_models().keys()), index=3)

st.code(guidance_text(cv_folds, test_size, numeric_df.shape[0]), language="text")

feature_df = numeric_df.drop(columns=[target_column], errors="ignore")

leak_candidates = []
for col in feature_df.columns:
    matched = find_avg_film([col])
    if matched:
        leak_candidates.append(col)

if leak_candidates:
    feature_df = feature_df.drop(columns=leak_candidates, errors="ignore")

X = feature_df.values
y = numeric_df[target_column].values

kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
pipeline = available_models()[model_name]

scoring = {
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
    "r2": "r2",
}

cv_result = cross_validate(
    pipeline,
    X,
    y,
    cv=kf,
    scoring=scoring,
    return_train_score=False,
)

cv_table = pd.DataFrame(
    {
        "MAE (cv)": -cv_result["test_mae"],
        "RMSE (cv)": np.sqrt(-cv_result["test_mse"]),
        "R2 (cv)": cv_result["test_r2"],
    }
)

st.markdown("**Cross-validation metrics**")
cv_summary = cv_table.describe().loc[["mean", "std"]].T
st.dataframe(cv_summary, use_container_width=True)

with st.popover("Explain these CV metrics"):
    r2_cv = cv_summary.loc["R2 (cv)", "mean"] if "R2 (cv)" in cv_summary.index else np.nan
    rmse_cv = cv_summary.loc["RMSE (cv)", "mean"] if "RMSE (cv)" in cv_summary.index else np.nan

    if np.isfinite(r2_cv):
        if r2_cv > 0.98:
            st.write(
                "Your model explains almost all variation in this cleaned dataset. "
                "This often happens when speed, temperature and load are consistent."
            )
        elif r2_cv > 0.9:
            st.write(
                "Model explains most of the variance. Some runs are different → check experimental scatter."
            )
        else:
            st.write(
                "Model sees heterogeneous runs. Try a nonlinear model or tighten experiment ranges."
            )

    if np.isfinite(rmse_cv):
        st.write(f"Typical error (cv RMSE) is about **{rmse_cv:,.2f} nm**.")

    st.write(
        "If these numbers move a lot when you change the fold count, the dataset is small or has outliers."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42,
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

mse_value = mean_squared_error(y_test, y_pred)
holdout_metrics = pd.Series(
    {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": float(np.sqrt(mse_value)),
        "R2": r2_score(y_test, y_pred),
    }
)

st.markdown("**Holdout metrics**")
st.dataframe(holdout_metrics.to_frame("value"), use_container_width=True)

with st.popover("What does this holdout tell me?"):
    r2_h = holdout_metrics["R2"]
    rmse_h = holdout_metrics["RMSE"]

    if r2_h >= 0.98:
        st.write("Very high agreement on unseen data. Cleaning pattern looks consistent.")
    elif r2_h >= 0.9:
        st.write("Good generalization. A few rows are harder to predict → check temperature / SRR.")
    else:
        st.write("Accuracy drops on unseen rows. Dataset may be small or files vary in format.")
    st.write(f"Typical error on new TXT files is about **{rmse_h:,.2f} nm**.")
    st.write("You can reduce this if the most important features are measured more consistently.")

fig_pred = go.Figure()
fig_pred.add_scatter(x=y_test, y=y_pred, mode="markers", name="predicted")

actual_min = float(np.nanmin(y_test))
actual_max = float(np.nanmax(y_test))
fig_pred.add_trace(
    go.Scatter(
        x=[actual_min, actual_max],
        y=[actual_min, actual_max],
        mode="lines",
        name="ideal",
    )
)
fig_pred.update_layout(
    title="Predicted vs Actual — Average Film",
    xaxis_title="Actual",
    yaxis_title="Predicted",
)
inline_plotly_fig(fig_pred, height=360)

with st.popover("Describe this plot"):
    st.write("Points on the diagonal → prediction is close.")
    st.write(
        "If high film values are always underpredicted, collect more runs in that region."
    )
    st.write("A wide cloud → noisy or missing values in key columns.")

residuals = y_test - y_pred

fig_resid = go.Figure()
fig_resid.add_scatter(x=y_pred, y=residuals, mode="markers", name="residuals")
fig_resid.add_hline(y=0)
fig_resid.update_layout(
    title="Residuals vs Fitted",
    xaxis_title="Fitted",
    yaxis_title="Residual",
)
inline_plotly_fig(fig_resid, height=360)

with st.popover("Residuals insight"):
    st.write("Residuals should be around 0.")
    st.write("A funnel shape means the error grows for big films → collect more data there.")
    st.write("Horizontal bands can come from rounded or differently formatted TXT files.")

train_sizes, train_scores, test_scores = learning_curve(
    pipeline,
    X,
    y,
    cv=kf,
    scoring="r2",
    train_sizes=np.linspace(0.2, 1.0, 5),
)

fig_lc = go.Figure()
fig_lc.add_scatter(
    x=train_sizes,
    y=train_scores.mean(axis=1),
    mode="lines+markers",
    name="Train R²",
)
fig_lc.add_scatter(
    x=train_sizes,
    y=test_scores.mean(axis=1),
    mode="lines+markers",
    name="CV R²",
)
fig_lc.update_layout(
    title="Learning curve (R²)",
    xaxis_title="Training samples",
    yaxis_title="R²",
)
inline_plotly_fig(fig_lc, height=360)

with st.popover("Learning curve explanation"):
    st.write("If both curves meet near 1.0, the data is clean and the model is not overfitting.")
    st.write("If train is high but CV lower, the model memorizes small variations.")
    st.write("Add TXT files with rare speed / temperature combinations to improve CV.")

feature_names = list(feature_df.columns)

perm_result = permutation_importance(
    pipeline,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="r2",
)
perm_series = pd.Series(
    perm_result.importances_mean,
    index=feature_names,
).sort_values(ascending=False)

fig_perm = go.Figure()
fig_perm.add_bar(
    y=perm_series.index[::-1],
    x=perm_series.values[::-1],
    orientation="h",
)
fig_perm.update_layout(
    title="Permutation importance",
)
inline_plotly_fig(fig_perm, height=360)

with st.popover("How to improve the model from this chart"):
    if not perm_series.empty:
        top_feature = perm_series.index[0]
        st.write(f"The model is most sensitive to **{top_feature}**.")
        st.write(
            "If this column is noisy or very wide, the prediction will also be noisy."
        )
        st.write(
            f"Try to stabilize or narrow the range of `{top_feature}` in future experiments."
        )
        other_features = list(perm_series.index[1:3])
        if other_features:
            st.write(f"Next important features: {', '.join(other_features)}")
    else:
        st.write("Importance could not be computed. Train again with a nonlinear model.")

st.header("Cleaning methodology & dataset suitability")
st.write(
    "• Duplicate column coalescing.\n"
    "• Artifact row removal.\n"
    "• Robust scaling → KNN imputation → inverse scale.\n"
    "• Constant / empty column drop."
)
st.code(suitability_report(df), language="text")
