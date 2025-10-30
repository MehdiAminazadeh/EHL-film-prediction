from __future__ import annotations

import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from ui_shared import topbar, suitability_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


st.set_page_config(page_title="Deep Learning • EHL Film Prediction", layout="wide")
topbar("DL", show_sidebar=True)

st.title("Deep Learning (Order-grouped modeling)")

st.info(
    "**Important**\n\n"
    "Order isn’t an independent variable. It is derived from the film thickness itself "
    "(each time the film grows by ≈ λ / 2n, the order increments).\n\n"
    "This page trains per **Order group** (bins of 4 orders): 3–6, 7–10, 11–14, 15–18, … "
    "The last group can contain fewer orders depending on your data. "
    "All thresholds, predictions and suggestions are computed within the selected group."
)

df = st.session_state.get("last_clean_df")
if df is None or df.empty:
    st.warning("No cleaned data found. Upload/clean on **Home** first.")
    st.stop()


def find_column(pattern: str) -> str | None:
    for col in df.columns:
        if re.search(pattern, str(col), re.I):
            return col
    return None


col_avg_film = find_column(r"\baverage\s*film\b")
col_order = find_column(r"\border\b")
col_avg_wl = find_column(r"\baverage\s*wavelength\b")

if col_avg_film is None or col_order is None:
    st.error("Missing required columns: 'Average Film' and/or 'Order'.")
    st.stop()

order_series = pd.to_numeric(df[col_order], errors="coerce").dropna().astype(int)
if order_series.empty:
    st.error("No integer Order values detected.")
    st.stop()

min_order = int(order_series.min())
max_order = int(order_series.max())

start_order = 3
if min_order < start_order:
    offset = (min_order - start_order) % 4
    start_order = min_order - offset

order_bins: list[tuple[int, int]] = []
current_lo = start_order
while current_lo <= max_order:
    current_hi = current_lo + 3
    order_bins.append((current_lo, min(current_hi, max_order)))
    current_lo = current_hi + 1

bin_labels = [f"{lo}-{hi}" for (lo, hi) in order_bins]

selected_label = st.selectbox("Select Order group", bin_labels, index=0)
selected_index = bin_labels.index(selected_label)
selected_lo, selected_hi = order_bins[selected_index]

mask_group = (order_series >= selected_lo) & (order_series <= selected_hi)
subset = df.loc[mask_group.values].copy()

if subset.empty:
    st.warning(f"No rows for Order group {selected_label}.")
    st.stop()

avg_wavelength = None
if col_avg_wl and col_avg_wl in subset.columns:
    avg_wavelength = (
        pd.to_numeric(subset[col_avg_wl], errors="coerce")
        .dropna()
        .mean()
    )

header_text = f"**Rows in group {selected_label}:** {len(subset)}"
if avg_wavelength is not None and np.isfinite(avg_wavelength):
    header_text += f" • **Avg Wavelength:** {avg_wavelength:.2f} nm"
st.markdown(header_text)

y_series = pd.to_numeric(subset[col_avg_film], errors="coerce")
subset = subset.loc[y_series.notna()].copy()
y_series = y_series.loc[y_series.notna()]

if len(y_series) == 0:
    st.error("No valid 'Average Film' values in this order group.")
    st.stop()

group_threshold = float(y_series.mean())

numeric_subset = subset.select_dtypes(include=[np.number]).copy()

drop_in_features: list[str] = []
for col in numeric_subset.columns:
    col_lower = str(col).lower()
    if "average film" in col_lower:
        drop_in_features.append(col)
    elif "order" in col_lower:
        drop_in_features.append(col)
    elif "average wavelength" in col_lower:
        drop_in_features.append(col)
    elif "lube ref index" in col_lower:
        drop_in_features.append(col)

feature_df = numeric_subset.drop(columns=drop_in_features, errors="ignore")

if feature_df.shape[1] == 0:
    st.error("No usable features after excluding target/order/wavelength.")
    st.stop()

left_col, right_col = st.columns([1.25, 1.75], gap="large")

with left_col:
    st.subheader(f"What-if simulator (Orders {selected_label})")
    slider_values: dict[str, float] = {}
    at_least_one_slider = False

    for feature_name in feature_df.columns:
        series_values = pd.to_numeric(subset[feature_name], errors="coerce").dropna()
        if series_values.empty:
            continue

        low = float(np.nanpercentile(series_values, 1))
        high = float(np.nanpercentile(series_values, 99))
        median = float(np.nanmedian(series_values))

        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            st.caption(
                f"• `{feature_name}` cannot be adjusted for this order group (no variation)."
            )
            continue

        slider_values[feature_name] = st.slider(
            feature_name,
            low,
            high,
            value=median,
        )
        at_least_one_slider = True

    if not at_least_one_slider:
        st.error(
            "Dataset is not valid for this selected order range. "
            "Try a lower order range or upload richer data."
        )
        st.stop()

with right_col:
    st.subheader("Train group regressor & predict")

    X = feature_df.values
    y = y_series.values

    regressor = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
    )

    if len(y) < 8:
        regressor.fit(X, y)
        y_hat_train = regressor.predict(X)
        mae_resub = float(np.mean(np.abs(y - y_hat_train)))
        st.caption(
            f"Train size: {len(y)} (tiny). Resubstitution MAE ≈ {mae_resub:.3f} nm."
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )
        regressor.fit(X_train, y_train)
        y_pred_test = regressor.predict(X_test)
        mae_holdout = mean_absolute_error(y_test, y_pred_test)
        st.caption(
            f"Train size: {len(y_train)}, Test size: {len(y_test)}. "
            f"Holdout MAE ≈ {mae_holdout:.3f} nm."
        )

    feature_names = list(feature_df.columns)

    prediction_row: list[float] = []
    for name in feature_names:
        if name in slider_values:
            prediction_row.append(slider_values[name])
        else:
            col_vals = pd.to_numeric(subset[name], errors="coerce").dropna()
            if len(col_vals):
                prediction_row.append(float(np.nanmedian(col_vals)))
            else:
                prediction_row.append(0.0)

    prediction_input = np.array([prediction_row], dtype=float)
    predicted_film = float(regressor.predict(prediction_input)[0])

    prediction_is_ok = predicted_film >= group_threshold

    st.markdown(
        f"**Predicted Average Film (nm):** "
        f"<span style='background:#153e27;color:#b7ffd1;padding:2px 6px;border-radius:6px'>{predicted_film:,.4f}</span> "
        f"&nbsp;|&nbsp; **Threshold:** "
        f"<span style='background:#16361f;color:#b6f5bf;padding:2px 6px;border-radius:6px'>{group_threshold:,.4f}</span>",
        unsafe_allow_html=True,
    )

    if not np.isnan(group_threshold):
        if prediction_is_ok:
            st.success("Prediction is at or above the threshold.")
        else:
            st.warning("Prediction is below the threshold. To increase Average Film (nm):")

    corr_df = subset[feature_df.columns.tolist() + [col_avg_film]].corr(
        method="spearman"
    )
    corr_series = corr_df[col_avg_film].drop(col_avg_film)

    positive_features = corr_series[corr_series > 0].sort_values(
        ascending=False
    ).index.tolist()
    negative_features = corr_series[corr_series < 0].sort_values().index.tolist()

    if not prediction_is_ok:
        if positive_features:
            st.markdown("**Increase:** " + ", ".join(positive_features))
        if negative_features:
            st.markdown("**Decrease:** " + ", ".join(negative_features))

    bar_fig = go.Figure()
    bar_fig.add_bar(
        x=["Predicted", "Threshold"],
        y=[predicted_film, group_threshold],
    )
    bar_fig.update_layout(
        template="plotly_dark",
        height=260,
        title=f"Orders {selected_label}: Predicted vs Threshold (nm)",
    )
    st.plotly_chart(bar_fig, use_container_width=True)

st.divider()
st.caption(suitability_report(subset.select_dtypes(include=[np.number])))
