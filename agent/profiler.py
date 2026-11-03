"""
profiler.py — understand every column before touching it.

For each column the agent works out a *semantic type*:
    id · numeric · numeric_text (numbers stored as text) · datetime · date_text
    (dates stored as text) · boolean · categorical · text
and measures missing values, uniqueness, ranges, outliers and suspicious values.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--", "?", "missing", "#n/a"}
NON_NEGATIVE_HINTS = ("qty", "quantity", "price", "amount", "cost", "revenue", "sales", "age",
                      "units", "volume", "stock", "weight", "salary", "income")
ID_HINTS = re.compile(r"(^id$|_id$|^id_|_code$|_no$|_number$|^sku$|_sku$|_key$)", re.I)  # matched on snake_case names


def _snake(name: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name).strip())
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    return s or "column"


def to_number(s: pd.Series) -> pd.Series:
    """'$1,234.50' -> 1234.5 ; '15%' -> 15 ; '(200)' -> -200."""
    t = s.astype("string").str.strip()
    t = t.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    t = t.str.replace(r"[,$€£৳%\s]", "", regex=True)
    return pd.to_numeric(t, errors="coerce")


def to_datetime(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, errors="coerce", format="mixed")


def is_null_token(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower().isin(NULL_TOKENS).fillna(False)


def semantic_type(s: pd.Series, name: str) -> str:
    non_null = s[~s.isna()]
    if len(non_null) == 0:
        return "empty"
    n = len(non_null)
    unique_ratio = non_null.nunique() / n

    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        vals = non_null.astype(float)
        integer_like = np.all(np.mod(vals, 1) == 0)
        if ID_HINTS.search(name) and integer_like:
            return "id"
        if set(np.unique(vals)).issubset({0, 1}) and non_null.nunique() <= 2:
            return "boolean"
        return "numeric"

    # object / string columns
    str_vals = non_null.astype("string").str.strip()
    str_vals = str_vals[~str_vals.str.lower().isin(NULL_TOKENS)]
    if len(str_vals) == 0:
        return "empty"
    if str_vals.str.lower().isin({"true", "false", "yes", "no", "y", "n"}).mean() > 0.95:
        return "boolean"
    if ID_HINTS.search(name) and unique_ratio > 0.5:
        return "id"
    num = to_number(str_vals)
    if num.notna().mean() > 0.9:
        return "numeric_text"
    sample = str_vals.head(500)
    looks_date = sample.str.contains(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}", regex=True).mean()
    if looks_date >= 0.8 and to_datetime(sample).notna().mean() >= 0.8:
        return "date_text"
    if str_vals.nunique() <= max(30, 0.05 * n):
        return "categorical"
    return "text"


def iqr_outliers(x: pd.Series, k: float = 3.0) -> pd.Series:
    x = x.dropna()
    if len(x) < 10:
        return pd.Series(False, index=x.index)
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=x.index)
    return (x < q1 - k * iqr) | (x > q3 + k * iqr)


def profile_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column with type, missing %, uniqueness, range and issue counts."""
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        stype = semantic_type(s, _snake(col))
        null_tokens = int(is_null_token(s).sum()) if s.dtype == object or pd.api.types.is_string_dtype(s) else 0
        missing = int(s.isna().sum()) + null_tokens
        row = dict(column=col, clean_name=_snake(col), semantic_type=stype, dtype=str(s.dtype),
                   missing=missing, missing_pct=round(100 * missing / n, 2) if n else 0.0,
                   unique=int(s.nunique(dropna=True)),
                   sample=", ".join(map(str, s.dropna().astype(str).unique()[:3])),
                   min=None, max=None, mean=None, outliers=0, negatives=0)
        if stype in ("numeric", "numeric_text"):
            x = s if stype == "numeric" else to_number(s)
            row.update(min=float(x.min()) if x.notna().any() else None,
                       max=float(x.max()) if x.notna().any() else None,
                       mean=round(float(x.mean()), 4) if x.notna().any() else None,
                       outliers=int(iqr_outliers(x).sum()),
                       negatives=int((x < 0).sum()))
        elif stype in ("datetime", "date_text"):
            x = s if stype == "datetime" else to_datetime(s)
            if x.notna().any():
                row.update(min=str(x.min().date()), max=str(x.max().date()))
        rows.append(row)
    return pd.DataFrame(rows)


def find_key_column(df: pd.DataFrame, prof: pd.DataFrame) -> str | None:
    """The table's primary key = first id-type column that is (almost) unique."""
    for _, r in prof.iterrows():
        if r["semantic_type"] == "id" and r["unique"] >= 0.97 * max(len(df) - r["missing"], 1):
            return r["column"]
    return None


def non_negative_expected(name: str) -> bool:
    n = name.lower()
    return any(h in n for h in NON_NEGATIVE_HINTS) and "change" not in n and "growth" not in n
