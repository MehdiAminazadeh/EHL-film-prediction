import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui_shared import topbar

st.set_page_config(page_title="Statistics • EHL", layout="wide")
topbar("Statistics", show_sidebar=True)

st.markdown('<div class="wrap">', unsafe_allow_html=True)
st.title("Statistics")

df = st.session_state.get("last_clean_df")
if df is None or df.empty:
    st.info("Go to **Home** to upload data first.")
    st.stop()

numeric_df = df.select_dtypes(include=[np.number]).copy()
if numeric_df.empty:
    st.warning("No numeric columns to analyze.")
    st.stop()

tabs = st.tabs(
    [
        "Summary & Percentiles",
        "Correlation / Heatmap",
        "Distributions",
        "Outlier Diagnostics",
    ]
)

with tabs[0]:
    percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    description = numeric_df.describe(percentiles=percentiles).T
    coverage = numeric_df.notna().mean().to_frame("coverage")
    missing = (1 - coverage["coverage"]).to_frame("missing_rate")
    output = description.join(coverage).join(missing)
    st.dataframe(output, use_container_width=True, height=420)

    buffer = io.StringIO()
    output.to_csv(buffer)
    st.download_button("Download summary CSV", buffer.getvalue().encode(), "summary_stats.csv")

with tabs[1]:
    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        method = st.selectbox("Method", ["pearson", "spearman", "kendall"])
        target = st.selectbox("Rank features vs… (optional)", ["(none)"] + list(numeric_df.columns))
        sort_abs = st.checkbox("Sort by |corr|", True)

    corr = numeric_df.corr(method=method)

    fig = px.imshow(corr, color_continuous_scale="Viridis", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(title=f"Correlation heatmap ({method})")

    with col_right:
        st.plotly_chart(fig, use_container_width=True)

    if target and target != "(none)":
        series = corr[target].drop(labels=[target])
        if sort_abs:
            series = series.reindex(series.abs().sort_values(ascending=False).index)
        else:
            series = series.sort_values(ascending=False)
        st.markdown("**Feature ranking**")
        st.dataframe(series.to_frame("corr").style.format(precision=3), use_container_width=True, height=300)

with tabs[2]:
    st.markdown("#### Single distribution")
    main_column = st.selectbox("Column", list(numeric_df.columns))
    bins = st.slider("Bins", 10, 120, 40, 5)
    main_series = numeric_df[main_column].dropna()

    hist = px.histogram(main_series, nbins=bins, title=f"Histogram — {main_column}")
    hist.update_layout(height=320, xaxis_title=main_column, yaxis_title="count")
    st.plotly_chart(hist, use_container_width=True)

    box_fig = go.Figure()
    box_fig.add_box(y=main_series.values, name=main_column)
    box_fig.update_layout(height=260, title="Boxplot")
    st.popover("Show boxplot").write(st.plotly_chart(box_fig, use_container_width=True))

    st.markdown("#### Compare with other columns")
    other_columns = [c for c in numeric_df.columns if c != main_column]

    compare_cols = st.multiselect(
        "Add columns to overlay (each only once)",
        options=other_columns,
    )

    if compare_cols:
        base = main_series
        base_lo = float(base.quantile(0.01))
        base_hi = float(base.quantile(0.99))
        base_clipped = base.clip(lower=base_lo, upper=base_hi)

        overlay_fig = go.Figure()
        overlay_fig.add_histogram(
            x=base_clipped.values,
            name=main_column,
            opacity=0.55,
            nbinsx=bins,
        )

        for col_name in compare_cols:
            series = numeric_df[col_name].dropna()
            if series.empty:
                continue
            lo = float(series.quantile(0.01))
            hi = float(series.quantile(0.99))
            series_clipped = series.clip(lower=lo, upper=hi)
            overlay_fig.add_histogram(
                x=series_clipped.values,
                name=col_name,
                opacity=0.45,
                nbinsx=bins,
            )

        overlay_fig.update_layout(
            barmode="overlay",
            height=360,
            title="Overlay comparison (clipped to 1–99%)",
            xaxis_title="value",
            yaxis_title="count",
        )
        st.plotly_chart(overlay_fig, use_container_width=True)
        st.caption("Clipped each selected column to its own 1–99% range to make the shapes comparable.")
    else:
        st.caption("Select 1 or more columns above to overlay them under the main plot.")

with tabs[3]:
    method = st.radio("Method", ["Z-score", "IQR"], horizontal=True)
    column_name = st.selectbox("Column", list(numeric_df.columns), key="outlier_col")
    series = pd.to_numeric(numeric_df[column_name], errors="coerce").dropna()

    if series.empty:
        st.warning("No data in selected column.")
    else:
        if method == "Z-score":
            z_threshold = st.slider("Z threshold", 2.0, 6.0, 3.0, 0.1)
            std = series.std() if series.std() != 0 else 1
            z_scores = (series - series.mean()) / std
            mask = z_scores.abs() > z_threshold
            details = pd.DataFrame({"value": series[mask], "zscore": z_scores[mask]}).sort_values(
                "zscore", key=np.abs
            )
        else:
            iqr_multiplier = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1)
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            mask = (series < lower_bound) | (series > upper_bound)
            details = pd.DataFrame(
                {"value": series[mask], "lower": lower_bound, "upper": upper_bound}
            )

        percent_flagged = 100.0 * mask.mean()
        st.markdown(f"**Flagged:** {mask.sum()} rows ({percent_flagged:.2f}%)")

        scatter_fig = go.Figure()
        scatter_fig.add_scatter(x=series.index, y=series.values, mode="markers", name="data")
        scatter_fig.add_scatter(x=series.index[mask], y=series[mask], mode="markers", name="outliers")
        scatter_fig.update_layout(height=360, title=f"{column_name}: Outliers ({method})")
        st.plotly_chart(scatter_fig, use_container_width=True)
