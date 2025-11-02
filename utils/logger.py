from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class ExperimentLogger:
    """
    Simple in-memory accumulator that can flush to Excel.
    Use your run script to also emit a CSV if you like.
    """
    def __init__(self, enabled: bool, excel_path: str, verbosity: int = 1):
        self.enabled = bool(enabled)
        self.verbosity = int(verbosity)
        self.path: Path = Path(excel_path)
        self._rows: List[Dict[str, Any]] = []

    def append_row(self, row: Dict[str, Any]) -> None:
        self._rows.append(dict(row))
        if self.verbosity > 0:
            print(row)

    def flush(self) -> None:
        if not self.enabled:
            return
        df_new = pd.DataFrame(self._rows)
        if self.path.exists():
            try:
                df_old = pd.read_excel(self.path)
                df = pd.concat([df_old, df_new], ignore_index=True)
            except Exception:
                df = df_new
        else:
            df = df_new
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(self.path, index=False)

    def clear(self) -> None:
        self._rows.clear()
