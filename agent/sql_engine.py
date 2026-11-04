"""
sql_engine.py — run SQL on the cleaned tables with DuckDB (an in-process SQL database).

Safety: the agent only ever runs ONE read-only statement (SELECT / WITH).
Anything that could change or leave the database — INSERT, UPDATE, DELETE, DROP,
CREATE, ATTACH, COPY, file readers, PRAGMA … — is refused before it reaches DuckDB,
and every result is capped at MAX_ROWS rows.
"""

from __future__ import annotations

import re

import duckdb
import pandas as pd

MAX_ROWS = 5000
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|export|import|install|load|pragma|"
    r"call|set|reset|vacuum|checkpoint|grant|revoke|truncate|merge|"
    r"read_csv\w*|read_parquet|read_json\w*|read_text|read_blob|glob|httpfs|getenv)\b",
    re.I)


class UnsafeSQL(ValueError):
    pass


def clean_sql(sql: str) -> str:
    """Strip markdown fences, comments and a trailing semicolon."""
    s = sql.strip()
    s = re.sub(r"^```(?:sql)?\s*|\s*```$", "", s, flags=re.I)
    s = re.sub(r"--[^\n]*", "", s)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    return s.strip().rstrip(";").strip()


def check_sql(sql: str) -> str:
    s = clean_sql(sql)
    if not s:
        raise UnsafeSQL("Empty query.")
    if ";" in s:
        raise UnsafeSQL("Only one SQL statement is allowed.")
    if not re.match(r"^(select|with)\b", s, re.I):
        raise UnsafeSQL("Only SELECT queries are allowed.")
    # ignore words inside string literals when looking for forbidden keywords
    no_strings = re.sub(r"'(?:[^']|'')*'", "''", s)
    m = FORBIDDEN.search(no_strings)
    if m:
        raise UnsafeSQL(f"Keyword '{m.group(0)}' is not allowed in a read-only query.")
    return s


class SQLEngine:
    def __init__(self, tables: dict[str, pd.DataFrame]):
        self.con = duckdb.connect(database=":memory:")
        self.tables = {}
        for name, df in tables.items():
            self.con.register(f"_df_{name}", df)
            self.con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM "_df_{name}"')
            self.con.unregister(f"_df_{name}")
            self.tables[name] = df
        # lock it down: no file system or extensions after setup
        self.con.execute("SET enable_external_access = false")
        self.con.execute("SET lock_configuration = true")

    def schema(self, sample_values: int = 3) -> str:
        """Compact schema text for the AI prompt: tables, columns, types, example values."""
        lines = []
        for name, df in self.tables.items():
            lines.append(f"TABLE {name} ({len(df):,} rows)")
            info = self.con.execute(f"DESCRIBE \"{name}\"").fetchdf()
            for _, r in info.iterrows():
                col = r["column_name"]
                ex = df[col].dropna().astype(str).unique()[:sample_values]
                ex_txt = ", ".join(e[:25] for e in ex)
                lines.append(f"  - {col} {r['column_type']}  e.g. {ex_txt}")
        return "\n".join(lines)

    def run(self, sql: str) -> pd.DataFrame:
        s = check_sql(sql)
        df = self.con.execute(f"SELECT * FROM ({s}) AS q LIMIT {MAX_ROWS}").fetchdf()
        return df
