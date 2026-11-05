"""
cleaner.py — automatic, explainable data cleaning.

Every change is written to a log (step, column, rows affected, what was done),
so the Data Quality Report can show exactly what the agent did and why.

Cleaning steps, in order
    1. Standardise column names to snake_case
    2. Trim spaces; turn null-like text ("N/A", "null", "-") into real missing values
    3. Fix types: numbers stored as text, dates stored as text, yes/no -> boolean
    4. Remove exact duplicate rows
    5. Remove duplicate primary-key rows (keep the first)
    6. Invalid values: negatives in columns that can't be negative, impossible ages -> missing
    7. Missing values: drop columns > 60 % empty, drop rows with missing key,
       numeric -> median, categorical -> mode or "Unknown", dates left empty (flagged)
    8. Flag extreme outliers (kept, not deleted — a human should decide)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import profiler as P

DROP_COLUMN_MISSING_PCT = 60.0


def similar_labels(s: pd.Series, cutoff: float = 0.85, max_labels: int = 60) -> list[tuple[str, str, int]]:
    """Pairs of category labels that look like typos of each other (rarer label first)."""
    from difflib import SequenceMatcher
    vc = s.dropna().astype(str).value_counts()
    if len(vc) > max_labels:
        return []
    labs, out = list(vc.index), []
    for i, a in enumerate(labs):
        for b in labs[i + 1:]:
            if len(a) >= 4 and len(b) >= 4 and SequenceMatcher(None, a.lower(), b.lower()).ratio() >= cutoff:
                rare, common = (a, b) if vc[a] < vc[b] else (b, a)
                out.append((rare, common, int(vc[rare])))
    return out


@dataclass
class CleanResult:
    table: str
    raw: pd.DataFrame
    clean: pd.DataFrame
    profile_before: pd.DataFrame
    profile_after: pd.DataFrame
    log: list = field(default_factory=list)
    key_column: str | None = None
    column_map: dict = field(default_factory=dict)

    @property
    def log_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.log, columns=["step", "column", "rows_affected", "action"])


def clean_table(df: pd.DataFrame, table: str = "data", fill_numeric: str = "median") -> CleanResult:
    raw = df.copy()
    log: list[dict] = []

    def add(step, column, rows, action):
        if rows:
            log.append(dict(step=step, column=column, rows_affected=int(rows), action=action))

    prof_before = P.profile_table(raw)
    d = raw.copy()

    # 1. column names -------------------------------------------------------
    names, seen = {}, {}
    for c in d.columns:
        new = P._snake(c)
        if new in seen:
            seen[new] += 1
            new = f"{new}_{seen[new]}"
        else:
            seen[new] = 0
        names[c] = new
    renamed = sum(1 for k, v in names.items() if k != v)
    d = d.rename(columns=names)
    add("1. Column names", "(all)", renamed, f"Renamed {renamed} column(s) to snake_case")

    types = dict(zip(prof_before.column.map(names), prof_before.semantic_type))

    # 2. whitespace + null tokens ---------------------------------------------
    for c in d.columns:
        if d[c].dtype == object or pd.api.types.is_string_dtype(d[c]):
            before = d[c].copy()
            s = d[c].astype("string").str.strip()
            s = s.str.replace(r"\s+", " ", regex=True)
            tokens = s.str.lower().isin(P.NULL_TOKENS).fillna(False)
            changed_ws = int(((before.astype("string") != s) & before.notna() & ~tokens).sum())
            s = s.mask(tokens)
            d[c] = s.astype(object).where(s.notna(), None)
            add("2. Text tidy", c, changed_ws, "Trimmed extra spaces")
            add("2. Text tidy", c, tokens.sum(), "Converted null-like text (N/A, null, -) to missing")

    # 3. types -----------------------------------------------------------------
    for c, t in types.items():
        if t == "numeric_text":
            conv = P.to_number(d[c])
            bad = int((conv.isna() & d[c].notna()).sum())
            d[c] = conv
            add("3. Fix types", c, int(conv.notna().sum()), "Converted text numbers ($, %, commas) to numeric")
            add("3. Fix types", c, bad, "Values that were not numbers set to missing")
        elif t == "date_text":
            conv = P.to_datetime(d[c])
            bad = int((conv.isna() & d[c].notna()).sum())
            d[c] = conv
            add("3. Fix types", c, int(conv.notna().sum()), "Converted text to dates")
            add("3. Fix types", c, bad, "Unreadable dates set to missing")
        elif t == "boolean" and not pd.api.types.is_bool_dtype(d[c]) and not pd.api.types.is_numeric_dtype(d[c]):
            m = {"true": True, "yes": True, "y": True, "false": False, "no": False, "n": False}
            d[c] = d[c].astype("string").str.lower().map(m)
            add("3. Fix types", c, int(d[c].notna().sum()), "Converted yes/no text to True/False")
        elif t == "categorical":
            # unify case variants: "completed" / "Completed " -> most common spelling
            s = d[c].astype("string")
            canon = s.dropna().groupby(s.dropna().str.lower()).agg(lambda x: x.value_counts().index[0])
            new = s.str.lower().map(canon)
            changed = int(((new != s) & s.notna()).sum())
            if changed:
                d[c] = new.astype(object).where(new.notna(), None)
                add("3. Fix types", c, changed, "Unified spelling / case of category labels")
            # near-duplicate labels (e.g. "Mashad" vs "Mashhad"): flag for a human, don't guess
            for a_lab, b_lab, n_rows in similar_labels(d[c]):
                add("3. Fix types", c, n_rows, f"Possible duplicate labels '{a_lab}' and '{b_lab}' — flagged for review")

    # 4. exact duplicates -------------------------------------------------------
    dup = d.duplicated(keep="first")
    add("4. Duplicates", "(row)", dup.sum(), "Removed exact duplicate rows")
    d = d[~dup]

    # 5. duplicate keys ----------------------------------------------------------
    prof_mid = P.profile_table(d)
    key = P.find_key_column(d, prof_mid)
    if key:
        kdup = d[key].notna() & d.duplicated(subset=[key], keep="first")
        add("5. Duplicate keys", key, kdup.sum(), f"Removed rows whose {key} was already used (kept first)")
        d = d[~kdup]

    # 6. invalid values -------------------------------------------------------------
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c]):
            if P.non_negative_expected(c):
                neg = d[c] < 0
                add("6. Invalid values", c, neg.sum(), "Negative values are impossible here -> set to missing")
                d.loc[neg, c] = pd.NA
            if c == "age" or c.endswith("_age"):
                bad = (d[c] > 120) | (d[c] < 0)
                add("6. Invalid values", c, bad.sum(), "Age outside 0-120 -> set to missing")
                d.loc[bad, c] = pd.NA

    # 7. missing values -----------------------------------------------------------------
    for c in list(d.columns):
        pct = 100 * d[c].isna().mean() if len(d) else 0
        if pct > DROP_COLUMN_MISSING_PCT:
            add("7. Missing values", c, len(d), f"Dropped column: {pct:.0f}% empty")
            d = d.drop(columns=c)
    if key and key in d.columns:
        miss_key = d[key].isna()
        add("7. Missing values", key, miss_key.sum(), "Dropped rows with no key value")
        d = d[~miss_key]

    prof_now = P.profile_table(d)
    stype = dict(zip(prof_now.column, prof_now.semantic_type))
    for c in d.columns:
        n_miss = int(d[c].isna().sum())
        if not n_miss:
            continue
        t = stype.get(c)
        if t == "numeric":
            val = d[c].median() if fill_numeric == "median" else d[c].mean()
            d[c] = d[c].fillna(val)
            add("7. Missing values", c, n_miss, f"Filled with {fill_numeric} ({val:,.4g})")
        elif t in ("categorical", "boolean"):
            mode = d[c].mode(dropna=True)
            share = d[c].value_counts(normalize=True).iloc[0] if d[c].notna().any() else 0
            if share >= 0.5 and len(mode):
                d[c] = d[c].fillna(mode.iloc[0])
                add("7. Missing values", c, n_miss, f"Filled with most common value '{mode.iloc[0]}'")
            else:
                d[c] = d[c].fillna("Unknown")
                add("7. Missing values", c, n_miss, "Filled with 'Unknown'")
        elif t == "text":
            d[c] = d[c].fillna("")
            add("7. Missing values", c, n_miss, "Filled with empty text")
        else:  # dates, ids that are not the key
            add("7. Missing values", c, n_miss, "Left empty (flagged) — guessing a date or ID would invent data")

    # 8. outliers (flag only) ---------------------------------------------------------------
    for c in d.columns:
        if stype.get(c) == "numeric":
            out = P.iqr_outliers(pd.to_numeric(d[c], errors="coerce"))
            add("8. Outliers", c, out.sum(), "Extreme values flagged (3×IQR) — kept for review, not deleted")

    d = d.reset_index(drop=True)
    return CleanResult(table=table, raw=raw, clean=d, profile_before=prof_before,
                       profile_after=P.profile_table(d), log=log, key_column=key, column_map=names)


def clean_all(tables: dict[str, pd.DataFrame]) -> dict[str, CleanResult]:
    return {name: clean_table(df, name) for name, df in tables.items()}
