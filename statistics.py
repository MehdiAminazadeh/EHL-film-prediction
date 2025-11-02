from __future__ import annotations

import io
import re
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PX_THEME = "plotly_dark"


def _normalize_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(name)).replace("\xa0", " ")
    return re.sub(r"\s+", " ", normalized).strip().lower()


def _is_banned(name: str) -> bool:
    normalized = _normalize_name(name)
    is_order = "order" in normalized
    is_avg_wl = "average wavelength" in normalized or "avg wavelength" in normalized
    is_lube = "lube ref index" in normalized
    return is_order or is_avg_wl or is_lube


def _numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    drop_cols = [c for c in numeric_df.columns if _is_banned(c)]
    return numeric_df.drop(columns=drop_cols, errors="ignore")


def _download_button(df: pd.DataFrame, label: str, filename: str):
    buffer = io.StringIO()
    df.to_csv(buffer, index=True)
    st.download_button(label, buffer.getvalue().encode("utf-8"), filename, mime="text/csv")


def render_statistics_panel(df_clean: pd.DataFrame):
    st.markdown("### Statistics")

    if df_clean is None or df_clean.empty:
        st.info("No cleaned data yet.")
        return

    numeric_df = _numeric_only(df_clean)

    if numeric_df.empty:
        st.warning("No numeric columns to analyze.")
        return

    tabs = st.tabs(
        [
            "Summary & Percentiles",
            "Correlation / Heatmap",
            "Distributions",
            "Outlier Diagnostics",
        ]
    )
    #tab 1
    with tabs[0]:
        st.caption("Descriptive statistics with coverage.")
        percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        description = numeric_df.describe(percentiles=percentiles).T
        coverage = numeric_df.notna().mean().to_frame("coverage")
        missing = (1 - coverage["coverage"]).to_frame("missing_rate")
        output = description.join(coverage).join(missing)
        st.dataframe(output, use_container_width=True, height=420)
        _download_button(output, "Download summary CSV", "summary_stats.csv")

    
    #tab 2
    with tabs[1]:
        col_left, col_right = st.columns([1.0, 1.0], gap="large")

        with col_left:
            method = st.selectbox("Method", ["pearson", "spearman", "kendall"], index=0)
            target = st.selectbox("Rank features vs… (optional)", ["(none)"] + list(numeric_df.columns))
            sort_abs = st.checkbox("Sort by |corr|", True)

        correlations = numeric_df.corr(method=method)

        # dynamic, square heatmap
        n_features = len(correlations.columns)
        base_size = max(400, 40 * n_features)

        heatmap_fig = go.Figure(
            data=go.Heatmap(
                z=correlations.values,
                x=list(correlations.columns),
                y=list(correlations.index),
                colorscale="Viridis",
                zmin=-1,
                zmax=1,
            )
        )
        heatmap_fig.update_layout(
            template=PX_THEME,
            title=f"Correlation heatmap ({method})",
            width=base_size,
            height=base_size,
            xaxis_title="Features",
            yaxis_title="Features",
            xaxis=dict(tickangle=45, side="bottom"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=70, r=40, t=60, b=80),
        )

        with col_right:
           
            st.plotly_chart(heatmap_fig, use_container_width=False)

        if target and target != "(none)":
            series = correlations[target].drop(labels=[target])
            if sort_abs:
                series = series.reindex(series.abs().sort_values(ascending=False).index)
            else:
                series = series.sort_values(ascending=False)
            st.markdown("**Feature ranking**")
            st.dataframe(series.to_frame("corr").style.format(precision=3), use_container_width=True, height=300)
            _download_button(series.to_frame("corr"), "Download correlations CSV", f"corr_vs_{target}.csv")

    #tab 3
    with tabs[2]:
        st.markdown("#### Single distribution")
        main_column = st.selectbox("Column", list(numeric_df.columns), key="dist_main_col")
        bin_count = st.slider("Bins", 10, 120, 40, 5, key="dist_bins")

        main_series = numeric_df[main_column].dropna()

        hist_fig = px.histogram(
            main_series,
            nbins=bin_count,
            template=PX_THEME,
            title=f"Histogram — {main_column}",
        )
        hist_fig.update_layout(height=320, xaxis_title=main_column, yaxis_title="count")
        st.plotly_chart(hist_fig, use_container_width=True)

        box_fig = go.Figure()
        box_fig.add_box(y=main_series.values, name=main_column)
        box_fig.update_layout(template=PX_THEME, height=260, title="Boxplot")
        st.popover("Show boxplot").write(st.plotly_chart(box_fig, use_container_width=True))

        st.markdown("#### Compare with other columns")
        other_candidates = [c for c in numeric_df.columns if c != main_column]

        compare_cols = st.multiselect(
            "Add columns to overlay (each only once)",
            options=other_candidates,
            key="dist_compare_cols",
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
                nbinsx=bin_count,
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
                    nbinsx=bin_count,
                )

            overlay_fig.update_layout(
                barmode="overlay",
                template=PX_THEME,
                height=360,
                title="Overlay comparison (clipped to 1–99%)",
                xaxis_title="value",
                yaxis_title="count",
            )
            st.plotly_chart(overlay_fig, use_container_width=True)
            st.caption("Clipped each selected column to its 1–99% range to avoid long tails dominating.")
        else:
            st.caption("Select 1 or more columns above to overlay them under the main plot.")

    #tab 4
    with tabs[3]:
        method = st.radio("Method", ["Z-score", "IQR"], horizontal=True)
        column_to_check = st.selectbox("Column", list(numeric_df.columns), key="outlier_col")
        series = numeric_df[column_to_check].dropna()

        if series.empty:
            st.warning("No data in selected column.")
            return

        if method == "Z-score":
            z_threshold = st.slider("Z threshold", 2.0, 6.0, 3.0, 0.1)
            std = series.std()
            if std == 0:
                std = 1.0
            z_scores = (series - series.mean()) / std
            mask = z_scores.abs() > z_threshold
            details = (
                pd.DataFrame(
                    {
                        "value": series[mask],
                        "zscore": z_scores[mask],
                    }
                )
                .sort_values("zscore", key=np.abs, ascending=False)
            )
        else:
            iqr_factor = st.slider("IQR factor (k)", 1.0, 3.0, 1.5, 0.1)
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            low = q1 - iqr_factor * iqr
            high = q3 + iqr_factor * iqr
            mask = (series < low) | (series > high)
            details = (
                pd.DataFrame(
                    {
                        "value": series[mask],
                        "lower_bound": low,
                        "upper_bound": high,
                    }
                )
                .sort_values("value")
            )

        percent_flagged = 100.0 * mask.mean()
        st.markdown(f"**Flagged:** {mask.sum()} rows  ({percent_flagged:.2f}%)")

        scatter_fig = go.Figure()
        scatter_fig.add_scatter(x=series.index, y=series.values, mode="markers", name="data")
        scatter_fig.add_scatter(x=series.index[mask], y=series[mask], mode="markers", name="outliers")
        scatter_fig.update_layout(
            template=PX_THEME,
            height=360,
            title=f"{column_to_check}: Outliers ({method})",
            xaxis_title="Row index",
            yaxis_title=column_to_check,
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

        box_outlier_fig = go.Figure()
        box_outlier_fig.add_box(y=series.values, name=column_to_check)
        box_outlier_fig.update_layout(template=PX_THEME, height=320, title="Boxplot")
        st.plotly_chart(box_outlier_fig, use_container_width=True)

        with st.expander("Outlier rows", expanded=False):
            st.dataframe(details, use_container_width=True, height=240)
            _download_button(details, "Download outliers CSV", f"outliers_{column_to_check}_{method.lower()}.csv")
