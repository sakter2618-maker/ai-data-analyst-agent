"""
pipeline.py — the whole agent in one call, for automation (n8n, cron, the API).

    files ──▶ load ──▶ clean ──▶ quality scores ──▶ relationships ──▶ questions answered
          ──▶ final report with recommendations ──▶ one JSON-friendly result

The Streamlit app runs the same steps interactively; this module runs them headless.
"""

from __future__ import annotations

import html
import re
from datetime import datetime

from .ask import ask
from .cleaner import clean_all
from .llm import LLM, MockLLM
from .loader import load_bytes
from .quality import find_relationships, grade, quality_report_md, quality_scores
from .report import build_report, rule_recommendations
from .sql_engine import SQLEngine


def relationships_text(rel) -> str:
    if rel is None or rel.empty:
        return ""
    return "\n".join(f"{x.child_table}.{x.column} -> {x.parent_table}.{x.column}" for _, x in rel.iterrows())


def run_pipeline(files: dict[str, bytes], llm: LLM | None = None, questions: list[str] | None = None) -> dict:
    """files: {"orders.csv": b"...", ...}. Returns scores, reports (markdown + HTML) and recommendations."""
    llm = llm or MockLLM()
    tables = {}
    for name, raw in files.items():
        tables.update(load_bytes(raw, name))
    if not tables:
        raise ValueError("No readable tables (send CSV or Excel files).")

    results = clean_all(tables)
    rel = find_relationships(results)
    engine = SQLEngine({t: r.clean for t, r in results.items()})
    answers = [ask(q, engine, llm, relationships_text(rel)) for q in (questions or []) if q.strip()]

    report = build_report(results, rel, answers, llm)
    quality = quality_report_md(results, rel)
    tables_out = []
    for t, r in results.items():
        b, a = quality_scores(r.raw), quality_scores(r.clean)
        tables_out.append(dict(table=t, rows_before=b["rows"], rows_after=a["rows"],
                               score_before=b["overall"], score_after=a["overall"],
                               grade_before=grade(b["overall"]), grade_after=grade(a["overall"]),
                               fixes=len(r.log)))
    avg_after = round(sum(x["score_after"] for x in tables_out) / len(tables_out), 1)
    recs = rule_recommendations(results, rel)
    summary = (f"{len(tables_out)} table(s), {sum(x['rows_before'] for x in tables_out):,} rows analysed; "
               f"data quality {avg_after}/100 ({grade(avg_after)}) after "
               f"{sum(x['fixes'] for x in tables_out)} cleaning steps. Top action: {recs[0]['title']}.")
    return dict(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        ai=("offline rules" if isinstance(llm, MockLLM) else llm.name),
        summary=summary,
        quality_score=avg_after,
        quality_grade=grade(avg_after),
        tables=tables_out,
        orphan_rows=int(rel.orphan_rows.sum()) if rel is not None and not rel.empty else 0,
        recommendations=recs,
        answers=[dict(question=a.question, answer=a.text, sql=a.sql, error=a.error) for a in answers],
        report_md=report,
        quality_report_md=quality,
        report_html=md_to_html(report),
    )


# ---------------------------------------------------------------------------
# Tiny Markdown -> HTML (headings, lists, tables, bold, italics, code) for e-mail
# ---------------------------------------------------------------------------
def _inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])_(.+?)_(?!\w)", r"<i>\1</i>", s)
    return s


def md_to_html(md: str) -> str:
    out, lines, i = [], md.split("\n"), 0
    td = 'style="border:1px solid #d7dce5;padding:6px 8px;text-align:left;vertical-align:top"'
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            code = html.escape("\n".join(lines[i + 1:j]))
            out.append(f'<pre style="background:#f3f5f9;padding:10px;border-radius:6px;overflow:auto">{code}</pre>')
            i = j + 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            rows = [line]
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                rows.append(lines[j])
                j += 1
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            head = "".join(f'<th {td} bgcolor="#eef2f8">{_inline(c)}</th>' for c in cells[0])
            body = "".join("<tr>" + "".join(f"<td {td}>{_inline(c)}</td>" for c in r) + "</tr>" for r in cells[1:])
            out.append(f'<table style="border-collapse:collapse;margin:8px 0;font-size:14px"><tr>{head}</tr>{body}</table>')
            i = j
            continue
        if line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        m = re.match(r"^(#{1,4}) (.*)", line)
        if m:
            n = len(m.group(1))
            out.append(f"<h{n} style=\"color:#0B1220\">{_inline(m.group(2))}</h{n}>")
        elif line.strip():
            out.append(f"<p>{_inline(line)}</p>")
        i += 1
    return ('<div style="font-family:Segoe UI,Arial,sans-serif;color:#1f2937;line-height:1.5;max-width:900px">'
            + "\n".join(out) + "</div>")
