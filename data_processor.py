from __future__ import annotations

import io
import re
import typing as t
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

EHS_MAGIC = "EHS102 Test Data File"

HEADER_FIRST_CANDIDATES = {
    "Rolling Speed (mm/s)",
    "Step Time (s)",
    "Rolling Speed (mm/s)\tAverage Film (nm)",
    "Rolling Speed (mm/s)\tTraction Coeff (-)",
    "Rolling Speed (mm/s)\tImage file name",
}

STEP_START_RE = re.compile(r'^Step\s+\d+\s+"[^"]*"\s+started at\b', re.IGNORECASE)
STEP_TYPE_RE = re.compile(r"^Step type\t([A-Za-z]+)\tStep description", re.IGNORECASE)
ALL_STEPS_RE = re.compile(r"^\s*All steps completed", re.IGNORECASE)
STATUS_RE = re.compile(
    r"(All\s+steps\s+completed|test\s+completed\s+normally|Test\s+stopped\s+by\s+the\s+user|during\s+step\s+\d+)",
    re.IGNORECASE,
)
NUMERIC_NAME_RE = re.compile(r"^\s*-?\d+(\.\d+)?\s*$", re.IGNORECASE)


def extract_supported_from_zip(zip_bytes: bytes) -> t.List[t.Tuple[str, bytes]]:
    extracted: t.List[t.Tuple[str, bytes]] = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            name_lower = name.lower()
            is_plain_file = not name.endswith("/")
            if name_lower.endswith(".txt") and is_plain_file:
                base_name = name.split("/")[-1]
                extracted.append((base_name, zf.read(name)))

    return extracted


def _make_unique_headers(raw_columns: t.Iterable[str]) -> t.List[str]:
    seen_counts: dict[str, int] = {}
    unique_columns: list[str] = []

    for raw in raw_columns:
        col_name = "" if raw is None else str(raw)
        if col_name in seen_counts:
            seen_counts[col_name] += 1
            unique_columns.append(f"{col_name} ({seen_counts[col_name]})")
        else:
            seen_counts[col_name] = 0
            unique_columns.append(col_name)

    return unique_columns


def _looks_like_header(line: str) -> bool:
    return any(line.startswith(candidate) for candidate in HEADER_FIRST_CANDIDATES)


def _split_tabrow(line: str) -> t.List[str]:
    return line.rstrip("\n").split("\t")


def _is_ehs_text(text: str) -> bool:
    return text.lstrip().startswith(EHS_MAGIC)


def extract_tables_from_txt(text: str, source_name: str) -> t.List[pd.DataFrame]:
    lines = text.splitlines()
    tables: list[pd.DataFrame] = []
    index = 0
    current_step_type = None

    while index < len(lines):
        line = lines[index]

        match_step_type = STEP_TYPE_RE.match(line)
        if match_step_type:
            current_step_type = match_step_type.group(1)
            index += 1
            continue

        if _looks_like_header(line):
            header = _make_unique_headers(_split_tabrow(line))
            data_rows: list[t.List[str]] = []
            index += 1

            while index < len(lines):
                next_line = lines[index]

                if STEP_START_RE.match(next_line) or _looks_like_header(next_line):
                    break

                if not next_line.strip() or ALL_STEPS_RE.match(next_line):
                    index += 1
                    continue

                row = _split_tabrow(next_line)
                if len(row) > len(header):
                    row = row[: len(header)]
                elif len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))

                data_rows.append(row)
                index += 1

            if data_rows:
                df = pd.DataFrame(data_rows, columns=header)
                df.insert(0, "source_file", source_name)
                if current_step_type:
                    df.insert(1, "StepType", current_step_type)
                tables.append(df)

            continue

        index += 1

    return tables


def parse_txt_file(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    text = file_bytes.decode("utf-8", errors="replace")

    if not _is_ehs_text(text):
        return pd.DataFrame(columns=["source_file"])

    tables = extract_tables_from_txt(text, file_name)
    if not tables:
        return pd.DataFrame(columns=["source_file"])

    merged = pd.concat(tables, ignore_index=True, sort=False)

    pattern = re.compile(r"^\(\d+\)$")
    redundant_cols = [c for c in merged.columns if pattern.match(str(c))]
    merged = merged.drop(columns=redundant_cols, errors="ignore")
    merged = merged.dropna(axis=1, how="all")

    for col in list(merged.columns):
        unique_vals = merged[col].dropna().unique()
        if len(unique_vals) == 1:
            as_text = str(unique_vals[0]).strip().lower()
            is_noise = (
                as_text in ["film", "timed"]
                or as_text.startswith("fva")
                or as_text.startswith("step")
            )
            if is_noise:
                merged = merged.drop(columns=[col])

    noise_keywords = [
        "image",
        "file name",
        "step type",
        "steptype",
        "unnamed",
        "arbitrary",
        "-9",
    ]
    to_drop = [c for c in merged.columns if any(k in str(c).lower() for k in noise_keywords)]
    merged = merged.drop(columns=to_drop, errors="ignore")

    for col in list(merged.columns):
        try:
            numeric_vals = pd.to_numeric(merged[col], errors="coerce")
            if numeric_vals.notna().sum() > 0 and numeric_vals.dropna().nunique() == 1:
                only_val = numeric_vals.dropna().iloc[0]
                if only_val in (-9, 0):
                    merged = merged.drop(columns=[col])
        except Exception:
            continue

    for col in merged.columns:
        is_common_numeric = any(
            key in col
            for key in [
                "Speed",
                "Load",
                "Temp",
                "(nm)",
                "SRR",
                "Order",
                "Traction",
                "Time",
            ]
        )
        if is_common_numeric:
            merged[col] = pd.to_numeric(merged[col], errors="ignore")

    merged = merged.reset_index(drop=True)
    merged["source_file"] = file_name
    return merged


def parse_single_file(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    lower_name = file_name.lower()
    if lower_name.endswith(".txt"):
        return parse_txt_file(file_name, file_bytes)
    return pd.DataFrame(columns=["source_file"])


def clean_merged(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.dropna(axis=1, how="all").copy()

    original_cols = list(df.columns)
    normalized = {c: str(c).strip() for c in original_cols}
    lower = {c: normalized[c].lower() for c in original_cols}
    to_drop: set[str] = set()

    for col in original_cols:
        is_source = lower[col] in {"source file", "source_file", "sourcefile"}
        if is_source:
            to_drop.add(col)

    negative_numeric_header = re.compile(r"^\s*-\d+(\.\d+)?\s*$")
    for col in original_cols:
        if negative_numeric_header.match(normalized[col]):
            to_drop.add(col)

    for col in original_cols:
        lower_col = lower[col]
        image_like = "image" in lower_col and "file" in lower_col and "name" in lower_col
        if image_like:
            to_drop.add(col)
        if lower_col.startswith("step time"):
            to_drop.add(col)
        if "sd repeat" in lower_col or "user repeat" in lower_col:
            to_drop.add(col)

    if to_drop:
        df = df.drop(columns=list(to_drop), errors="ignore")

    df = df.dropna(axis=1, how="all").reset_index(drop=True)
    return df


def remove_text_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    def row_as_text(row) -> str:
        return " ".join(row.values).lower()

    text_mask = df.astype(str).apply(row_as_text, axis=1)
    pattern = re.compile(
        r"(step\s+\d+|test\s+stopped|new\s+film\s+step|started\s+at)",
        flags=re.IGNORECASE,
    )
    filtered = df[~text_mask.str.contains(pattern)].copy()
    filtered.reset_index(drop=True, inplace=True)
    return filtered


def drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.dropna(axis=1, how="all")
    kept_columns: list[str] = []

    for col in df.columns:
        series = df[col]
        if series.dropna().nunique() > 1:
            kept_columns.append(col)

    df = df[kept_columns]

    pattern = re.compile(r"^\(\d+\)$")
    redundant = [c for c in df.columns if pattern.match(str(c))]
    df = df.drop(columns=redundant, errors="ignore")
    df.reset_index(drop=True, inplace=True)
    return df


def process_files(
    files: t.List[t.Tuple[str, bytes]],
    parallel: bool = True,
    max_workers: int = 8,
) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()

    if parallel and len(files) > 1:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(files))) as executor:
            dataframes = list(executor.map(lambda pair: parse_single_file(pair[0], pair[1]), files))
    else:
        dataframes = [parse_single_file(name, data) for name, data in files]

    if dataframes:
        merged = pd.concat(dataframes, ignore_index=True, sort=False)
    else:
        merged = pd.DataFrame()

    merged = clean_merged(merged)
    merged = remove_text_rows(merged)
    merged = drop_redundant_columns(merged)
    return merged
