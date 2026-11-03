"""
loader.py — read any CSV or Excel file into pandas DataFrames.

* CSV: delimiter (comma, semicolon, tab, pipe) and encoding are detected automatically.
* Excel: every sheet becomes its own table.
Returns a dict {table_name: DataFrame}.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import pandas as pd

MAX_ROWS = 1_000_000          # safety limit for the web app
ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]


def table_name(name: str) -> str:
    """'Retail Orders (2025).csv' -> 'retail_orders_2025' (safe to use as a SQL table name)."""
    stem = Path(name).stem
    s = re.sub(r"[^0-9a-zA-Z]+", "_", stem).strip("_").lower()
    if not s or s[0].isdigit():
        s = "t_" + s
    return s[:50]


def _read_csv(raw: bytes) -> pd.DataFrame:
    last_err = None
    for enc in ENCODINGS:
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
        try:
            dialect = csv.Sniffer().sniff(text[:20000], delimiters=",;\t|")
            sep = dialect.delimiter
        except csv.Error:
            sep = ","
        return pd.read_csv(io.StringIO(text), sep=sep, nrows=MAX_ROWS, low_memory=False)
    raise ValueError(f"Could not decode the file: {last_err}")


def load_bytes(raw: bytes, filename: str) -> dict[str, pd.DataFrame]:
    """Load an uploaded file (bytes + original filename)."""
    ext = Path(filename).suffix.lower()
    if ext in (".csv", ".txt", ".tsv"):
        return {table_name(filename): _read_csv(raw)}
    if ext in (".xlsx", ".xlsm", ".xls"):
        sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None, nrows=MAX_ROWS)
        out = {}
        for sheet, df in sheets.items():
            df = df.dropna(how="all").dropna(axis=1, how="all")
            if len(df):
                key = table_name(filename) if len(sheets) == 1 else table_name(f"{Path(filename).stem}_{sheet}")
                out[key] = df
        return out
    raise ValueError(f"Unsupported file type '{ext}'. Please upload a .csv or .xlsx file.")


def load_path(path: str | Path) -> dict[str, pd.DataFrame]:
    path = Path(path)
    return load_bytes(path.read_bytes(), path.name)


def load_many(paths) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for p in paths:
        tables.update(load_path(p))
    return tables
