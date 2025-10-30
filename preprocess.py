from __future__ import annotations

import io
import re
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler


def _normalize_column_name(name: str) -> str:
    text = str(name)
    text = text.replace("Â°C", "°C")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _base_column_name(name: str) -> str:
    text = _normalize_column_name(name)
    text = re.sub(r"\s*\(\d+\)\s*$", "", text)
    text = re.sub(r"\s*#\d+\s*$", "", text)
    text = re.sub(r"\s*\bcopy\b\s*$", "", text, flags=re.IGNORECASE)
    return text


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_column_name(c) for c in df.columns]

    groups: dict[str, list[str]] = {}
    for column in df.columns:
        base = _base_column_name(column)
        groups.setdefault(base, []).append(column)

    merged = pd.DataFrame(index=df.index)

    for base_name, cols in groups.items():
        if len(cols) == 1:
            merged[base_name] = df[cols[0]]
        else:
            merged[base_name] = df[cols].bfill(axis=1).ffill(axis=1).iloc[:, 0]

    return merged


STATUS_PATTERN = re.compile(
    r"(instrument has gone offline|test stopped|all steps completed|status|error|during step)",
    re.IGNORECASE,
)


def _remove_artifact_rows(original: pd.DataFrame, numeric_view: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    def is_artifact(row) -> bool:
        for value in row:
            if isinstance(value, str):
                long_text = len(value) > 32
                has_status = STATUS_PATTERN.search(value or "") is not None
                if long_text or has_status:
                    return True
        return False

    mask_artifacts = original.apply(is_artifact, axis=1)
    filtered = numeric_view.loc[~mask_artifacts].reset_index(drop=True)
    return filtered, int(mask_artifacts.sum())


def _clean_cell(value):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"", "-"}:
            return np.nan
        return stripped
    return value


DEFAULT_MUST_KEEP_PATTERNS: List[str] = [
    r"\baverage\s*film\b",
    r"\brolling\s*speed\b",
    r"\blube\s*temp\b|\blubricant\s*temp\b",
    r"\bpot\s*temp\b|\bplate\s*temp\b",
    r"\bball\s*load\b",
    r"\bsrr\b|\bslide\s*to\s*roll\b",
    r"\btraction\b",
    r"\border\b",
    r"\baverage\s*wavelength\b",
]


def _matches_any(column_name: str, patterns: List[str]) -> bool:
    lower = column_name.lower()
    return any(re.search(pattern, lower) for pattern in patterns)


def preprocess_dataframe(
    df_raw: pd.DataFrame,
    numeric_threshold: float = 0.80,
    must_keep_patterns: List[str] = None,
    keep_top_k: int = 12,
) -> Tuple[pd.DataFrame, str]:
    if must_keep_patterns is None:
        must_keep_patterns = DEFAULT_MUST_KEEP_PATTERNS

    report_lines: list[str] = []

    df_cleaned = df_raw.applymap(_clean_cell)
    df_cleaned.columns = [_normalize_column_name(c) for c in df_cleaned.columns]

    df_merged = _coalesce_duplicate_columns(df_cleaned)

    numeric_coverage: dict[str, float] = {}
    for col in df_merged.columns:
        series = pd.to_numeric(df_merged[col], errors="coerce")
        numeric_coverage[col] = series.notna().mean()

    passing_columns = {c for c, frac in numeric_coverage.items() if frac >= numeric_threshold}
    must_keep_columns = {c for c in df_merged.columns if _matches_any(c, must_keep_patterns)}

    sorted_by_coverage = sorted(numeric_coverage.items(), key=lambda item: item[1], reverse=True)
    topk_columns = {c for c, _ in sorted_by_coverage[:keep_top_k]}

    selected_columns = list(passing_columns | must_keep_columns | topk_columns)
    dropped_columns = sorted(set(df_merged.columns) - set(selected_columns))

    df_numeric = df_merged[selected_columns].apply(pd.to_numeric, errors="coerce")

    df_numeric, removed_rows_count = _remove_artifact_rows(df_cleaned, df_numeric)

    scaler = RobustScaler()
    scaled_values = scaler.fit_transform(df_numeric.values)

    imputer = KNNImputer(n_neighbors=5, weights="distance")
    imputed_values = imputer.fit_transform(scaled_values)

    restored_values = scaler.inverse_transform(imputed_values)
    df_imputed = pd.DataFrame(restored_values, columns=df_numeric.columns)

    constant_columns = [
        c for c in df_imputed.columns if pd.Series(df_imputed[c]).nunique(dropna=True) <= 1
    ]
    if constant_columns:
        df_imputed = df_imputed.drop(columns=constant_columns)

    df_imputed = df_imputed.dropna(axis=1, how="all").reset_index(drop=True)

    must_keep_hit = sorted([c for c in selected_columns if _matches_any(c, must_keep_patterns)])
    report_lines.append(f"Kept by numeric threshold (≥ {int(numeric_threshold * 100)}%): {len(passing_columns)} columns")
    report_lines.append(
        "Always-kept key signals (domain rules): "
        + (", ".join(must_keep_hit) if must_keep_hit else "none")
    )
    report_lines.append(f"Top-{keep_top_k} by usable values: {len(topk_columns)}")
    report_lines.append(f"Dropped (low coverage / non-numeric): {len(dropped_columns)}")
    report_lines.append(f"Rows removed with text/status artifacts: {removed_rows_count}")
    report_lines.append("Imputation pipeline: RobustScaler → KNNImputer(k=5, distance weights) → inverse-scale back to original units")
    report_lines.append("Rationale: robust to outliers and preserves physical units after imputation")
    report_lines.append(f"Constant columns dropped after imputation: {len(constant_columns)}")
    report_lines.append(f"Final shape: {df_imputed.shape[0]} rows × {df_imputed.shape[1]} columns")

    drop_lube_cols = [
        c for c in df_imputed.columns if re.search(r"\blube\s*ref\s*index\b", str(c), re.IGNORECASE)
    ]
    if drop_lube_cols:
        df_imputed = df_imputed.drop(columns=drop_lube_cols, errors="ignore")

    report_text = "\n".join(report_lines)
    return df_imputed, report_text


def preprocess_csv_bytes(csv_bytes: bytes, **kwargs) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(io.BytesIO(csv_bytes))
    return preprocess_dataframe(df, **kwargs)
