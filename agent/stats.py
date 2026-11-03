"""
stats.py — automatic statistical analysis of a cleaned table.

* Descriptive statistics for numeric columns (mean, median, std, skew, quartiles)
* Categorical breakdowns (top values and their share)
* Correlations (Pearson) — strongest pairs only
* Time trend (monthly counts / sums) when a date column exists
* Plain-English insights generated from the numbers (no AI needed)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import profiler as P


def column_roles(df: pd.DataFrame) -> dict:
    prof = P.profile_table(df)
    t = dict(zip(prof.column, prof.semantic_type))
    return dict(
        numeric=[c for c, v in t.items() if v == "numeric"],
        categorical=[c for c, v in t.items() if v in ("categorical", "boolean")],
        datetime=[c for c, v in t.items() if v == "datetime"],
        ids=[c for c, v in t.items() if v == "id"],
    )


def describe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame()
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    d = x.describe().T
    d["median"] = x.median()
    d["skew"] = x.skew()
    d["missing"] = x.isna().sum()
    d = d.rename(columns={"25%": "q1", "50%": "q2", "75%": "q3"})
    return d[["count", "mean", "median", "std", "min", "q1", "q3", "max", "skew", "missing"]].round(3)


def categorical_breakdown(df: pd.DataFrame, cols: list[str], top: int = 5) -> dict[str, pd.DataFrame]:
    out = {}
    for c in cols:
        vc = df[c].astype(str).value_counts()
        t = vc.head(top).rename("count").to_frame()
        t["share_pct"] = (100 * t["count"] / vc.sum()).round(1)
        out[c] = t.reset_index().rename(columns={"index": c})
    return out


def top_correlations(df: pd.DataFrame, cols: list[str], min_abs: float = 0.3, top: int = 10) -> pd.DataFrame:
    if len(cols) < 2:
        return pd.DataFrame(columns=["var_1", "var_2", "r", "strength"])
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr()
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= min_abs:
                strength = "strong" if abs(r) >= 0.7 else "moderate"
                pairs.append(dict(var_1=a, var_2=b, r=round(float(r), 3), strength=strength))
    return pd.DataFrame(pairs, columns=["var_1", "var_2", "r", "strength"]).sort_values(
        "r", key=abs, ascending=False).head(top)


def monthly_trend(df: pd.DataFrame, date_col: str, value_col: str | None = None) -> pd.DataFrame:
    d = df[[date_col] + ([value_col] if value_col else [])].dropna(subset=[date_col])
    month = pd.to_datetime(d[date_col]).dt.to_period("M").dt.to_timestamp()
    if value_col:
        out = d.groupby(month)[value_col].sum().rename("value")
    else:
        out = d.groupby(month).size().rename("value")
    out = out.reset_index().rename(columns={date_col: "month"})
    return out


def insights(df: pd.DataFrame, table: str = "data") -> list[str]:
    """Rule-based plain-English findings — these also become context for the AI."""
    r = column_roles(df)
    notes = [f"`{table}` has {len(df):,} rows and {df.shape[1]} columns "
             f"({len(r['numeric'])} numeric, {len(r['categorical'])} categorical, {len(r['datetime'])} date)."]
    desc = describe_numeric(df, r["numeric"])
    for c, s in desc.iterrows():
        consistent = (s["skew"] > 0 and s["mean"] > s["median"]) or (s["skew"] < 0 and s["mean"] < s["median"])
        if abs(s["skew"]) >= 1 and consistent:
            side = "right (a few very large values)" if s["skew"] > 0 else "left (a few very small values)"
            notes.append(f"`{c}` is skewed to the {side}: mean {s['mean']:,.2f} vs median {s['median']:,.2f} — "
                         f"use the median when reporting a typical value.")
    for c, t in categorical_breakdown(df, r["categorical"], top=1).items():
        if len(t) and t.share_pct.iloc[0] >= 50 and df[c].nunique() > 1:
            notes.append(f"`{c}` is dominated by '{t.iloc[0, 0]}' ({t.share_pct.iloc[0]}% of rows).")
    for _, p in top_correlations(df, r["numeric"]).iterrows():
        direction = "rise together" if p.r > 0 else "move in opposite directions"
        notes.append(f"`{p.var_1}` and `{p.var_2}` {direction} ({p.strength} correlation, r = {p.r}).")
    for dc in r["datetime"]:
        tr = monthly_trend(df, dc)
        if len(tr) >= 3:
            first, last = tr.month.min(), tr.month.max()
            notes.append(f"`{dc}` covers {first:%b %Y} – {last:%b %Y} ({len(tr)} months); busiest month "
                         f"{tr.loc[tr.value.idxmax(), 'month']:%b %Y} with {tr.value.max():,.0f} rows.")
    return notes


def analyze(df: pd.DataFrame, table: str = "data") -> dict:
    r = column_roles(df)
    return dict(
        roles=r,
        numeric=describe_numeric(df, r["numeric"]),
        categorical=categorical_breakdown(df, r["categorical"]),
        correlations=top_correlations(df, r["numeric"]),
        trends={c: monthly_trend(df, c) for c in r["datetime"]},
        insights=insights(df, table),
    )
