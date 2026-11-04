"""
ask.py — answer a business question in plain English.

    question ──▶ AI writes SQL (with the real schema) ──▶ safety check ──▶ DuckDB runs it
             ──▶ if it fails, the error goes back to the AI for ONE fix ──▶ result table
             ──▶ chart chosen ──▶ AI writes a 2-3 sentence answer from the numbers

Without an API key a rule-based planner writes simple single-table SQL instead,
so the demo still works offline (it handles "total X by Y", "top N", "trend by month",
"how many").
"""

from __future__ import annotations

import numbers
import re
from dataclasses import dataclass, field

import pandas as pd

from .llm import LLM, MockLLM
from .sql_engine import SQLEngine

SQL_SYSTEM = """You are a senior data analyst who writes DuckDB SQL.
Rules:
- Use ONLY the tables and columns in the schema. Quote nothing unless needed.
- One SELECT (CTEs allowed). Never modify data.
- Use date_trunc('month', col) for monthly trends; ROUND numbers to 2 decimals.
- Give every computed column a short snake_case alias.
- Return at most 50 rows unless the question needs more; order results meaningfully.
- Join tables on the keys listed under RELATIONSHIPS when the question needs several tables.
Return JSON: {"sql": "...", "chart": {"type": "bar|line|pie|table", "x": "<column>", "y": "<column>"},
"title": "<short chart title>", "assumption": "<one sentence on any assumption, or empty>"}"""

ANSWER_SYSTEM = """You are a data analyst explaining a query result to a business manager.
Write 2-3 short sentences: the direct answer first with its key numbers, then one notable
pattern. Use only numbers that appear in the result. No markdown headings."""


@dataclass
class Answer:
    question: str
    sql: str = ""
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    chart: dict = field(default_factory=dict)
    title: str = ""
    text: str = ""
    assumption: str = ""
    error: str = ""
    provider: str = ""


# ---------------------------------------------------------------------------
# Offline planner
# ---------------------------------------------------------------------------
def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _col_mentioned(question: str, cols: list[str]) -> list[str]:
    q = question.lower()
    hits = []
    for c in cols:
        parts = c.lower().split("_")
        if c.lower() in q or c.lower().replace("_", " ") in q or (len(parts) > 1 and all(p in q for p in parts)):
            hits.append(c)
        elif len(parts) == 1 and re.search(rf"\b{re.escape(c.lower())}s?\b", q):
            hits.append(c)
    return hits


def rule_based_sql(question: str, engine: SQLEngine) -> dict:
    q = question.lower()
    best, best_score = None, -1
    for name, df in engine.tables.items():
        score = len(_col_mentioned(q, list(df.columns))) * 10 + (3 if any(w in name for w in _words(q)) else 0)
        score += len(df) / 1e7
        if score > best_score:
            best, best_score = name, score
    df = engine.tables[best]
    cols = list(df.columns)
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and not c.endswith("_id")
               and not pd.api.types.is_bool_dtype(df[c])]
    dates = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    cats = [c for c in cols if (df[c].dtype == object or pd.api.types.is_bool_dtype(df[c])) and df[c].nunique() <= 60]
    mentioned = _col_mentioned(q, cols)

    metric_col = next((c for c in mentioned if c in numeric), None)
    agg = "AVG" if re.search(r"\b(average|avg|mean)\b", q) else "SUM"
    metric = f"ROUND({agg}({metric_col}), 2) AS {agg.lower()}_{metric_col}" if metric_col else "COUNT(*) AS row_count"
    metric_alias = metric.split(" AS ")[-1]
    m = re.search(r"\btop\s+(\d+)", q)
    limit = int(m.group(1)) if m else 20

    if dates and re.search(r"\b(trend|month|monthly|over time|per month|by month)\b", q):
        d = dates[0]
        sql = (f"SELECT date_trunc('month', {d}) AS month, {metric} FROM {best} "
               f"WHERE {d} IS NOT NULL GROUP BY 1 ORDER BY 1")
        chart = {"type": "line", "x": "month", "y": metric_alias}
    else:
        group = next((c for c in mentioned if c in cats), None)
        if group is None and re.search(r"\b(by|per|each|which|top)\b", q) and cats:
            group = cats[0]
        if group:
            sql = (f"SELECT {group}, {metric} FROM {best} GROUP BY 1 ORDER BY 2 DESC LIMIT {limit}")
            chart = {"type": "bar", "x": group, "y": metric_alias}
        else:
            sql = f"SELECT {metric} FROM {best}"
            chart = {"type": "table"}
    return {"sql": sql, "chart": chart, "title": question.strip().rstrip("?").capitalize(),
            "assumption": f"Offline mode: answered from table `{best}` with a rule-based query."}


def _fmt(v) -> str:
    if isinstance(v, numbers.Number) and not isinstance(v, bool):
        return f"{int(v):,}" if float(v).is_integer() else f"{float(v):,.2f}"
    if hasattr(v, "strftime"):
        return v.strftime("%b %Y")
    return str(v)


def _fallback_text(ans: Answer) -> str:
    d = ans.data
    if d.empty:
        return "The query returned no rows."
    if d.shape == (1, 1):
        return f"The answer is **{_fmt(d.iloc[0, 0])}**."
    num = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
    if not num or d.shape[1] < 2:
        return f"The query returned {len(d)} rows."
    y, x = num[-1], d.columns[0]
    additive = y.startswith(("sum_", "row_count", "count", "total"))
    if pd.api.types.is_datetime64_any_dtype(d[x]):
        s = d.sort_values(x)
        peak = s.loc[s[y].idxmax()]
        first, last = s.iloc[0], s.iloc[-1]
        return (f"Peak month was **{_fmt(peak[x])}** with {_fmt(peak[y])}. "
                f"It went from {_fmt(first[y])} in {_fmt(first[x])} to {_fmt(last[y])} in {_fmt(last[x])}.")
    s = d.sort_values(y, ascending=False)
    top, low = s.iloc[0], s.iloc[-1]
    txt = f"**{_fmt(top[x])}** is highest with {_fmt(top[y])}"
    if additive and d[y].sum():
        txt += f" ({100 * top[y] / d[y].sum():.1f}% of the total)"
    return txt + f"; **{_fmt(low[x])}** is lowest with {_fmt(low[y])} ({len(d)} groups compared)."


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def ask(question: str, engine: SQLEngine, llm: LLM, relationships: str = "", context: str = "") -> Answer:
    ans = Answer(question=question, provider=llm.name)
    prompt = (f"SCHEMA\n{engine.schema()}\n\nRELATIONSHIPS\n{relationships or 'none found'}\n\n"
              f"DATA NOTES\n{context or '-'}\n\nQUESTION\n{question}")
    try:
        plan = rule_based_sql(question, engine) if isinstance(llm, MockLLM) else llm.complete_json(SQL_SYSTEM, prompt)
    except Exception as e:  # AI unreachable -> offline planner
        plan = rule_based_sql(question, engine)
        ans.assumption = f"AI unavailable ({type(e).__name__}); used the offline planner."
    ans.sql, ans.chart = plan.get("sql", ""), plan.get("chart") or {}
    ans.title = plan.get("title", question)
    ans.assumption = ans.assumption or plan.get("assumption", "")

    for attempt in range(2):
        try:
            ans.data = engine.run(ans.sql)
            ans.error = ""
            break
        except Exception as e:
            ans.error = str(e).splitlines()[0][:300]
            if attempt == 0 and not isinstance(llm, MockLLM):
                fix = llm.complete_json(SQL_SYSTEM, prompt + f"\n\nYOUR PREVIOUS SQL\n{ans.sql}\n\nIT FAILED WITH\n"
                                                           f"{ans.error}\nReturn corrected JSON.")
                ans.sql, ans.chart = fix.get("sql", ans.sql), fix.get("chart") or ans.chart
            else:
                break
    if ans.error:
        ans.text = f"I couldn't run that query: {ans.error}"
        return ans

    if isinstance(llm, MockLLM):
        ans.text = _fallback_text(ans)
    else:
        try:
            preview = ans.data.head(30).to_csv(index=False)
            ans.text = llm.complete(ANSWER_SYSTEM, f"QUESTION: {question}\nSQL: {ans.sql}\nRESULT (CSV):\n{preview}",
                                    max_tokens=400).strip()
        except Exception:
            ans.text = _fallback_text(ans)
    return ans
