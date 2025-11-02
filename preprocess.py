from __future__ import annotations
import re
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

NONNEG_PATTERNS = [
    r"average\s*film",
    r"average\s*film\s*\(nm\)",
    r"rolling\s*speed",
    r"sliding\s*speed",
    r"ball\s*speed",
    r"disc\s*speed",
    r"lube\s*temp",
    r"pot\s*temp",
    r"ball\s*load",
    r"traction\s*force",
]

DEFAULT_MUST_KEEP_PATTERNS = [
    r"average\s*film",
    r"average\s*film\s*\(nm\)",
    r"rolling\s*speed",
    r"sliding\s*speed",
    r"traction",
    r"temperature",
    r"load",
]


def _normalize_column_name(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    name = re.sub(r"[\r\n\t]", " ", name)
    name = re.sub(r"\s+", " ", name.strip())
    return name


def _clean_cell(x):
    if isinstance(x, str):
        x = x.strip().replace(",", ".")
        if x == "":
            return np.nan
    return x


def _matches_any(col: str, patterns: List[str]) -> bool:
    for p in patterns:
        if re.search(p, col, re.IGNORECASE):
            return True
    return False


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        base = re.sub(r"\.\d+$", "", col)
        if base not in out:
            out[base] = df[col]
        else:
            out[base] = out[base].combine_first(df[col])
    return out


def _remove_artifact_rows(df_orig: pd.DataFrame, df_num: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    pattern = re.compile(r"(test\s+stopped|all\s+steps\s+completed|during\s+step)", re.IGNORECASE)
    mask = np.ones(len(df_num), dtype=bool)
    for i in range(len(df_orig)):
        row_txt = " ".join(str(x) for x in df_orig.iloc[i].values)
        if pattern.search(row_txt):
            mask[i] = False
    removed = int((~mask).sum())
    return df_num[mask], removed


def preprocess_dataframe(
    df_raw: pd.DataFrame,
    numeric_threshold: float = 0.5,
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
        s = pd.to_numeric(df_merged[col], errors="coerce")
        numeric_coverage[col] = s.notna().mean()

    passing = {c for c, frac in numeric_coverage.items() if frac >= numeric_threshold}
    must_keep = {c for c in df_merged.columns if _matches_any(c, must_keep_patterns)}
    sorted_cov = sorted(numeric_coverage.items(), key=lambda x: x[1], reverse=True)
    topk = {c for c, _ in sorted_cov[:keep_top_k]}

    selected_cols = list(passing | must_keep | topk)
    dropped_cols = sorted(set(df_merged.columns) - set(selected_cols))

   
    df_numeric = df_merged[selected_cols].apply(pd.to_numeric, errors="coerce")

    
    df_numeric, removed_rows = _remove_artifact_rows(df_cleaned, df_numeric)

    if df_numeric.shape[1] == 0:
        report_lines.append("All columns empty after artifact removal.")
        return pd.DataFrame(), "\n".join(report_lines)

    #remember cols that were non-negative before imputation
    nonneg_cols = []
    for c in df_numeric.columns:
        s = df_numeric[c].dropna()
        if len(s) > 0 and s.min() >= 0:
            nonneg_cols.append(c)

    # KNN
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    imputed = imputer.fit_transform(df_numeric)


    n_rows_arr, n_cols_arr = imputed.shape
    n_cols_df = df_numeric.shape[1]
    if n_cols_arr != n_cols_df:
        min_cols = min(n_cols_arr, n_cols_df)
        imputed = imputed[:, :min_cols]
        cols = list(df_numeric.columns)[:min_cols]
        df_imputed = pd.DataFrame(imputed, columns=cols)
        report_lines.append(
            f"[warn] column mismatch after imputation (array {n_cols_arr} vs df {n_cols_df}); trimmed to {min_cols}."
        )
    else:
        df_imputed = pd.DataFrame(imputed, columns=df_numeric.columns)

    
    for c in df_imputed.columns:
        if c in nonneg_cols or _matches_any(c, NONNEG_PATTERNS):
            df_imputed[c] = df_imputed[c].clip(lower=0)

    #drop constants and all-null
    const_cols = [c for c in df_imputed.columns if pd.Series(df_imputed[c]).nunique(dropna=True) <= 1]
    if const_cols:
        df_imputed = df_imputed.drop(columns=const_cols, errors="ignore")

    df_imputed = df_imputed.dropna(axis=1, how="all").reset_index(drop=True)

    #report
    must_keep_hit = sorted([c for c in selected_cols if _matches_any(c, must_keep_patterns)])
    report_lines.append(f"Kept ≥{int(numeric_threshold*100)}% numeric: {len(passing)}")
    report_lines.append("Always-kept key signals: " + (", ".join(must_keep_hit) if must_keep_hit else "none"))
    report_lines.append(f"Top-{keep_top_k} coverage cols: {len(topk)}")
    report_lines.append(f"Dropped (low coverage / non-numeric): {len(dropped_cols)}")
    report_lines.append(f"Rows removed (text/status artifacts): {removed_rows}")
    report_lines.append("Imputation: KNNImputer on original scale (no RobustScaler)")
    report_lines.append(f"Constant columns dropped: {len(const_cols)}")
    report_lines.append(f"Final shape: {df_imputed.shape[0]} × {df_imputed.shape[1]}")

    return df_imputed, "\n".join(report_lines)
