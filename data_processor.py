from __future__ import annotations

import io
import re
import zipfile
import hashlib
import typing as t

import pandas as pd

TARGET_COLS = [
    "Rolling Speed (mm/s)",
    "Sliding Speed (mm/s)",
    "Average Film (nm)",
    "Traction Force (N)",
    "Traction Force 1 (N)",
    "Traction Force 2 (N)",
    "Traction Coeff (-)",
    "Lube Temp (°C)",
    "Pot Temp (°C)",
    "Ball Load (N)",
    "Disc Speed 1",
    "Disc Speed 2",
    "Ball Speed 1",
    "Ball Speed 2",
    "SRR (%)",
]


def extract_supported_from_zip(zip_bytes: bytes) -> list[tuple[str, bytes]]:
    result: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            is_file = not name.endswith("/")
            is_txt = name.lower().endswith(".txt")
            if is_txt and is_file:
                short_name = name.split("/")[-1]
                result.append((short_name, zf.read(name)))
    return result


def _looks_like_header(line: str) -> bool:
    patterns = [
        r"Rolling Speed\s*\(mm/s\)",
        r"Step Time\s*\(s\)",
        r"Traction\s*Coeff",
        r"Average\s*Film",
    ]
    return any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns)


def _split_columns(line: str) -> list[str]:
    parts = re.split(r"\t+|\s{2,}", line.strip())
    return [part.strip() for part in parts if part.strip()]


def _parse_table_block(header: str, lines: list[str]) -> pd.DataFrame:
    buffer = io.StringIO("\n".join([header] + lines))
    df = pd.read_csv(buffer, sep=r"\t+|\s{2,}", engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df


def _extract_tables_from_text(text: str) -> list[pd.DataFrame]:
    lines = text.splitlines()
    tables: list[pd.DataFrame] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if _looks_like_header(line):
            header = line
            data_rows: list[str] = []
            index += 1
            while index < len(lines):
                next_line = lines[index]
                empty_line = not next_line.strip()
                is_step = re.match(r"^(Step\s+\d+|Test\s+stopped|All\s+steps\s+completed)", next_line, re.I)
                is_header = _looks_like_header(next_line)
                if empty_line or is_step or is_header:
                    break
                data_rows.append(next_line)
                index += 1
            if data_rows:
                tables.append(_parse_table_block(header, data_rows))
        else:
            index += 1

    return tables


def _normalize_to_target(df: pd.DataFrame) -> pd.DataFrame:
    normalized = {}
    for expected_col in TARGET_COLS:
        candidates = [
            c
            for c in df.columns
            if re.sub(r"\s+", " ", c.strip().lower())
            == re.sub(r"\s+", " ", expected_col.strip().lower())
        ]
        if candidates:
            normalized[expected_col] = pd.to_numeric(df[candidates[0]], errors="coerce")
        else:
            normalized[expected_col] = pd.Series([pd.NA] * len(df))
    return pd.DataFrame(normalized)


def _filter_film_if_present(df: pd.DataFrame) -> pd.DataFrame:
    if "Average Film (nm)" in df.columns and df["Average Film (nm)"].notna().any():
        df = df[df["Average Film (nm)"].notna()]
    return df.reset_index(drop=True)


def parse_single_file(name: str, content: bytes) -> pd.DataFrame:
    text = content.decode("utf-8", errors="ignore")
    tables = _extract_tables_from_text(text)

    if not tables:
        return pd.DataFrame(columns=TARGET_COLS)

    merged = pd.concat([_normalize_to_target(table) for table in tables], ignore_index=True)
    merged = _filter_film_if_present(merged)
    merged["__source_file"] = name
    return merged


def process_files(files: list[tuple[str, bytes]], parallel: bool = False) -> pd.DataFrame:
    dataframes: list[pd.DataFrame] = []
    seen_hashes: set[str] = set()

    for name, data in files:
        if not name.lower().endswith(".txt"):
            continue

        digest = hashlib.sha1(data).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)

        df = parse_single_file(name, data)
        if not df.empty:
            dataframes.append(df)

    if not dataframes:
        return pd.DataFrame(columns=TARGET_COLS)

    merged = pd.concat(dataframes, ignore_index=True)
    merged = merged.dropna(how="all").reset_index(drop=True)
    return merged[TARGET_COLS + ["__source_file"]]
