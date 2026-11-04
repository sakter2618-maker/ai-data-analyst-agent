"""
powerbi.py — prepare everything needed to build the Power BI dashboard by hand:

* the star schema (which table is the fact table, which are dimensions, the relationships)
* a Date table in DAX
* DAX measures: totals, averages, counts, distinct counts, rate of the dominant status,
  month-over-month %, year-over-year %, running total
* a page-by-page design structure (cards, charts, slicers) with a colour theme

It is rule-based (no AI needed), so the output is predictable and always valid DAX syntax.
"""

from __future__ import annotations

import pandas as pd

from .cleaner import CleanResult
from .stats import column_roles

THEME = {"name": "Agent Clean", "dataColors": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                 "#008300", "#6250d6", "#e34948"],
         "background": "#FFFFFF", "foreground": "#0B0B0B", "tableAccent": "#2A78D6"}


def _q(table: str) -> str:
    return f"'{table}'"


def _title(col: str) -> str:
    return col.replace("_", " ").title()


def pick_fact(results: dict[str, CleanResult], relationships: pd.DataFrame) -> str:
    def score(name):
        df = results[name].clean
        r = column_roles(df)
        children = 0 if relationships is None or relationships.empty else (relationships.child_table == name).sum()
        return (children * 3) + (2 if r["datetime"] else 0) + len(r["numeric"]) + len(df) / 1e6
    return max(results, key=score)


def build_model(results: dict[str, CleanResult], relationships: pd.DataFrame) -> dict:
    fact = pick_fact(results, relationships)
    df = results[fact].clean
    roles = column_roles(df)
    date_col = roles["datetime"][0] if roles["datetime"] else None
    F = _q(fact)
    measures = []

    def add(name, dax, fmt, why):
        measures.append(dict(name=name, dax=f"{name} = {dax}", format=fmt, purpose=why))

    noun = fact.split("_")[-1].rstrip("s").title() or "Row"
    add(f"Total {noun}s", f"COUNTROWS({F})", "#,0", f"How many {noun.lower()}s")
    for c in roles["numeric"][:4]:
        add(f"Total {_title(c)}", f"SUM({F}[{c}])", "#,0.00", f"Sum of {c}")
        add(f"Avg {_title(c)}", f"AVERAGE({F}[{c}])", "#,0.00", f"Average {c} per row")
    for c in roles["ids"]:
        if c != results[fact].key_column:
            add(f"Distinct {_title(c)}s", f"DISTINCTCOUNT({F}[{c}])", "#,0", f"Unique {c} values")
    for c in roles["categorical"][:3]:
        top = df[c].value_counts().index[0]
        if df[c].nunique() <= 8:
            add(f"{top} %", f'DIVIDE(CALCULATE(COUNTROWS({F}), {F}[{c}] = "{top}"), COUNTROWS({F}))',
                "0.0%", f"Share of rows where {c} = {top}")
    base = measures[0]["name"]
    if date_col:
        add(f"{base} MoM %", f"VAR cur = [{base}] VAR prev = CALCULATE([{base}], DATEADD('Date'[Date], -1, MONTH)) "
                             f"RETURN DIVIDE(cur - prev, prev)", "0.0%", "Change vs previous month")
        add(f"{base} YoY %", f"VAR cur = [{base}] VAR prev = CALCULATE([{base}], SAMEPERIODLASTYEAR('Date'[Date])) "
                             f"RETURN DIVIDE(cur - prev, prev)", "0.0%", "Change vs same period last year")
        add(f"{base} Running Total", f"CALCULATE([{base}], FILTER(ALL('Date'[Date]), 'Date'[Date] <= MAX('Date'[Date])))",
            "#,0", "Cumulative total over time")

    date_table = None
    if date_col:
        date_table = (f"Date = ADDCOLUMNS(CALENDAR(DATE(YEAR(MIN({F}[{date_col}])), 1, 1), "
                      f"DATE(YEAR(MAX({F}[{date_col}])), 12, 31)), \"Year\", YEAR([Date]), "
                      f"\"Month No\", MONTH([Date]), \"Month\", FORMAT([Date], \"MMM\"), "
                      f"\"Year-Month\", FORMAT([Date], \"YYYY-MM\"), \"Quarter\", \"Q\" & QUARTER([Date]))")

    rels = []
    if relationships is not None and not relationships.empty:
        for _, r in relationships.iterrows():
            rels.append(f"{r.child_table}[{r.column}]  *──1  {r.parent_table}[{r.column}]  (single direction)")
    if date_col:
        rels.append(f"{fact}[{date_col}]  *──1  Date[Date]  (mark 'Date' as date table)")

    cards = [m["name"] for m in measures[:4]]
    cats = roles["categorical"][:3]
    pages = [
        dict(page="1. Executive Overview", visuals=[f"KPI cards: {', '.join(cards)}"]
             + ([f"Line chart: {base} by Date[Year-Month]"] if date_col else [])
             + [f"Bar chart: {base} by {fact}[{c}]" for c in cats[:2]],
             slicers=(["Date[Year]"] if date_col else []) + [f"{fact}[{c}]" for c in cats[:2]]),
        dict(page="2. Trends", visuals=([f"Line chart: {base} and running total by month",
                                         f"Column chart: {base} MoM % by month",
                                         "Matrix: Year × Month with conditional colour"] if date_col else
                                        ["(no date column — skip this page)"]), slicers=["Date[Year]"] if date_col else []),
        dict(page="3. Breakdown", visuals=[f"Bar chart sorted descending: {m} by {fact}[{c}]"
                                           for c in cats for m in cards[:1]] + ["Decomposition tree on the main measure"],
             slicers=[f"{fact}[{c}]" for c in cats]),
        dict(page="4. Data Quality", visuals=["Cards: quality score before/after, rows removed, missing filled",
                                              "Table: cleaning log (step, column, rows affected, action)"],
             slicers=[]),
        dict(page="5. Detail", visuals=[f"Table: all {fact} columns with conditional formatting", "Drill-through from pages 1–3"],
             slicers=[]),
    ]
    return dict(fact=fact, dimensions=[t for t in results if t != fact], date_table=date_table,
                relationships=rels, measures=measures, pages=pages, theme=THEME)


def model_md(m: dict) -> str:
    out = ["# Power BI build guide", "",
           f"**Fact table:** `{m['fact']}` · **Dimensions:** {', '.join(f'`{d}`' for d in m['dimensions']) or 'none'}", "",
           "## 1. Relationships (Model view)"] + [f"- {r}" for r in m["relationships"] or ["none"]]
    if m["date_table"]:
        out += ["", "## 2. Date table (Modeling → New table)", "```DAX", m["date_table"], "```"]
    out += ["", "## 3. Measures (Modeling → New measure)", ""]
    for x in m["measures"]:
        out += [f"**{x['name']}** — {x['purpose']} · format `{x['format']}`", "```DAX", x["dax"], "```"]
    out += ["", "## 4. Pages"]
    for p in m["pages"]:
        out += ["", f"### {p['page']}"] + [f"- {v}" for v in p["visuals"]]
        if p["slicers"]:
            out.append(f"- Slicers: {', '.join(p['slicers'])}")
    out += ["", "## 5. Theme", "Colours (View → Themes → Customize): " + ", ".join(m["theme"]["dataColors"])]
    return "\n".join(out)
