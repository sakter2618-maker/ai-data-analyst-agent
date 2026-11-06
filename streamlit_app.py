"""
AI-Powered Data Analyst Automation Agent — web app
Run:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import hashlib
import html
import os
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent import clean_all, find_relationships, load_bytes, quality_report_md, quality_scores
from agent.ask import ask
from agent.charts import PALETTE, make_chart
from agent.llm import MockLLM, get_llm
from agent.powerbi import build_model, model_md
from agent.quality import grade
from agent.report import build_report
from agent.sql_engine import SQLEngine
from agent.stats import analyze

ROOT = Path(__file__).parent
SAMPLES = sorted((ROOT / "sample_data").glob("retail_*.csv"))
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = PALETTE

st.set_page_config(page_title="AI Data Analyst Agent", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 3.4rem; max-width: 1400px; }
.hero { background: linear-gradient(120deg,#2b1055 0%,#6250d6 45%,#2a78d6 80%,#1baf7a 125%); color:#fff;
        border-radius:18px; padding:22px 26px; margin-bottom:16px; box-shadow:0 8px 24px rgba(43,16,85,.2); }
.hero h1 { color:#fff; font-size:1.7rem; margin:0 0 4px; font-weight:800; line-height:1.25; }
.hero p { color:rgba(255,255,255,.9); margin:0; }
.hero .tag { display:inline-block; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.3);
             padding:2px 10px; border-radius:999px; font-size:.75rem; margin:0 6px 8px 0; }
.kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:4px 0 14px; }
.kpi { background:#fff; border:1px solid rgba(11,11,11,.08); border-radius:14px; padding:14px 16px 12px;
       position:relative; overflow:hidden; box-shadow:0 2px 8px rgba(11,11,11,.04); }
.kpi::before { content:""; position:absolute; left:0; top:0; bottom:0; width:5px; background:var(--a); }
.kpi .l { color:#52514e; font-size:.76rem; font-weight:700; text-transform:uppercase; letter-spacing:.04em; }
.kpi .v { color:#0b0b0b; font-size:1.6rem; font-weight:800; line-height:1.2; margin:2px 0; }
.kpi .n { color:#898781; font-size:.78rem; }
.answer { background:#fff; border:1px solid rgba(11,11,11,.08); border-left:5px solid #6250d6; border-radius:14px;
          padding:14px 18px; margin:8px 0 10px; font-size:1.02rem; }
.badge { display:inline-block; padding:3px 10px; border-radius:999px; font-weight:700; font-size:.8rem; }
.badge.on { background:#e3f5e3; color:#0b6b0b; } .badge.off { background:#fff3d6; color:#7a5200; }
.chart-title { font-weight:700; margin:10px 0 4px; }
[data-testid="stMarkdownContainer"] table { display:block; overflow-x:auto; max-width:100%; }
[data-testid="stPlotlyChart"] { background:#fff; border:1px solid rgba(11,11,11,.08); border-radius:14px; padding:6px; }
@media (max-width:640px) { .hero h1 { font-size:1.25rem; } .kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .kpi .v { font-size:1.3rem; } .block-container { padding-left:1rem; padding-right:1rem; } }
</style>""", unsafe_allow_html=True)


def kpis(cards):
    html = "".join(f'<div class="kpi" style="--a:{a}"><div class="l">{l}</div><div class="v">{v}</div>'
                   f'<div class="n">{n}</div></div>' for l, v, n, a in cards)
    st.markdown(f'<div class="kpi-grid">{html}</div>', unsafe_allow_html=True)


def chart_title(t):
    st.markdown(f'<div class="chart-title">{t}</div>', unsafe_allow_html=True)


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


@st.cache_data(show_spinner="The agent is reading and cleaning your data…")
def run_pipeline(files: tuple[tuple[str, bytes], ...]):
    tables = {}
    for name, raw in files:
        tables.update(load_bytes(raw, name))
    results = clean_all(tables)
    return results, find_relationships(results)


@st.cache_resource(show_spinner=False, max_entries=3)
def get_engine(_key: str, _results) -> SQLEngine:
    return SQLEngine({k: v.clean for k, v in _results.items()})


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Show midnight timestamps as plain dates in tables."""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]) and (d[c].dropna().dt.normalize() == d[c].dropna()).all():
            d[c] = d[c].dt.date
    return d


def secrets() -> dict:
    try:
        return dict(st.secrets)
    except Exception:
        return {}


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 1 · Your data")
    uploads = st.file_uploader("Upload CSV or Excel files", type=["csv", "xlsx", "xls"], accept_multiple_files=True,
                               help="Upload several related files (e.g. orders + customers) and the agent links them.")
    use_sample = st.toggle("Use the sample retail data", value=not uploads)
    st.markdown("### 2 · AI model")
    choice = st.radio("Provider", ["Gemini", "Claude", "Offline"], horizontal=True, label_visibility="collapsed")
    llm = MockLLM() if choice == "Offline" else get_llm(choice.lower(), secrets())
    if isinstance(llm, MockLLM):
        st.markdown('<span class="badge off">● Offline mode</span>', unsafe_allow_html=True)
        st.caption("No API key found, so simple questions are answered with built-in rules. "
                   "Add GEMINI_API_KEY (or ANTHROPIC_API_KEY) to the app's secrets for full AI answers.")
    else:
        st.markdown(f'<span class="badge on">● {llm.name.title()} connected</span>', unsafe_allow_html=True)
    st.divider()
    st.caption("Your files stay in this session; they are not saved. Built with Python · pandas · DuckDB · "
               "Plotly · Streamlit · Gemini / Claude.")

st.markdown('<div class="hero"><span class="tag">AI agent</span><span class="tag">Python · SQL · LLM</span>'
            '<span class="tag">Data quality → insight → report</span>'
            '<h1>AI-Powered Data Analyst Agent</h1>'
            '<p>Upload any CSV or Excel file. The agent cleans it, explains every fix, answers your questions with SQL '
            'and writes the report — plus a ready Power BI build guide.</p></div>', unsafe_allow_html=True)

if uploads:
    files = tuple((u.name, u.getvalue()) for u in uploads)
elif use_sample:
    files = tuple((p.name, p.read_bytes()) for p in SAMPLES)
else:
    st.info("Upload a file in the sidebar, or switch on the sample retail data to try the agent.")
    st.stop()

try:
    results, rel = run_pipeline(files)
except Exception as e:
    st.error(f"Couldn't read that file: {e}")
    st.stop()

key = hashlib.md5(b"".join(n.encode() + r[:2000] for n, r in files)).hexdigest()
if st.session_state.get("data_key") != key:
    st.session_state.update(data_key=key, answers=[])

before = {t: quality_scores(r.raw, r.profile_before) for t, r in results.items()}
after = {t: quality_scores(r.clean, r.profile_after) for t, r in results.items()}
rows_b = sum(b["rows"] for b in before.values())
rows_a = sum(a["rows"] for a in after.values())
avg_b = sum(b["overall"] for b in before.values()) / len(before)
avg_a = sum(a["overall"] for a in after.values()) / len(after)
orphans = int(rel.orphan_rows.sum()) if not rel.empty else 0
kpis([
    ("Tables", f"{len(results)}", ", ".join(list(results)[:3]) + ("…" if len(results) > 3 else ""), BLUE),
    ("Rows", f"{rows_a:,}", f"{rows_b - rows_a:,} duplicate rows removed", AQUA),
    ("Quality score", f"{avg_b:.1f} → {avg_a:.1f}", "average of all tables (0–100)", GREEN),
    ("Missing values", f"{sum(b['missing_cells'] for b in before.values()):,}", "found and handled", YELLOW),
    ("Fixes applied", f"{sum(len(r.log) for r in results.values())}", "every one logged", VIOLET),
    ("Orphan records", f"{orphans:,}", "IDs with no match in parent table", RED),
])

tab_q, tab_s, tab_a, tab_p, tab_r = st.tabs(["🧹 Data Quality", "📈 Statistics", "💬 Ask a Question",
                                             "📊 Power BI Guide", "📝 Final Report"])

# ---------------------------------------------------------------- 1. quality
with tab_q:
    t = st.selectbox("Table", list(results), key="q_table")
    r = results[t]
    b, a = before[t], after[t]
    c1, c2 = st.columns([1, 1])
    with c1:
        chart_title(f"Quality score {b['overall']} ({grade(b['overall'])}) → {a['overall']} ({grade(a['overall'])})")
        dims = ["completeness", "uniqueness", "validity", "consistency"]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Before", x=[d.title() for d in dims], y=[b[d] for d in dims], marker_color="#c3c2b7",
                             hovertemplate="%{x} before: %{y:.1f}<extra></extra>"))
        fig.add_trace(go.Bar(name="After", x=[d.title() for d in dims], y=[a[d] for d in dims], marker_color=AQUA,
                             hovertemplate="%{x} after: %{y:.1f}<extra></extra>"))
        lo = max(0, min(b[d] for d in dims) - 10)
        fig.update_layout(barmode="group", height=330, template="plotly_white", margin=dict(l=8, r=8, t=10, b=8),
                          legend=dict(orientation="h", y=1.08, x=0), yaxis=dict(range=[lo, 101], gridcolor="#e1e0d9"))
        show(fig)
    with c2:
        chart_title("Issues found per column")
        iss = r.profile_before[["column", "semantic_type", "missing", "negatives", "outliers"]]
        iss = iss[(iss.missing > 0) | (iss.negatives > 0) | (iss.outliers > 0) |
                  r.profile_before.semantic_type.isin(["numeric_text", "date_text"])]
        st.dataframe(iss if len(iss) else pd.DataFrame({"result": ["No issues found in this table"]}),
                     hide_index=True, width="stretch")
        st.caption(f"Key column detected: **{r.key_column or 'none'}** · {b['duplicate_rows']:,} duplicate rows · "
                   f"{b['invalid_values']:,} invalid values")
    chart_title("What the agent did (and why)")
    st.dataframe(r.log_df, hide_index=True, width="stretch")
    if not rel.empty:
        chart_title("Relationships the agent found between tables")
        st.dataframe(rel, hide_index=True, width="stretch")
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Cleaned data (CSV)", r.clean.to_csv(index=False).encode(), f"{t}_clean.csv", "text/csv",
                       width="stretch")
    d2.download_button("⬇ Data Quality Report (Markdown)", quality_report_md(results, rel).encode(),
                       "data_quality_report.md", width="stretch")

# ---------------------------------------------------------------- 2. statistics
with tab_s:
    t = st.selectbox("Table", list(results), key="s_table")
    df = results[t].clean
    an = analyze(df, t)
    chart_title("What the numbers say")
    for line in an["insights"]:
        st.markdown(f"- {line}")
    if len(an["numeric"]):
        chart_title("Descriptive statistics")
        st.dataframe(an["numeric"], width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        if an["categorical"]:
            cat = st.selectbox("Category breakdown", list(an["categorical"]), key="s_cat")
            cb = an["categorical"][cat]
            fig = make_chart(cb.rename(columns={cb.columns[0]: cat}), {"type": "bar", "x": cat, "y": "count"})
            if fig:
                show(fig)
    with c2:
        if an["trends"]:
            dc = st.selectbox("Trend over time", list(an["trends"]), key="s_trend")
            tr = an["trends"][dc].rename(columns={"value": "rows"})
            fig = make_chart(tr, {"type": "line", "x": "month", "y": "rows"})
            if fig:
                show(fig)
    if len(an["correlations"]):
        chart_title("Strongest correlations")
        st.dataframe(an["correlations"], hide_index=True, width="stretch")

# ---------------------------------------------------------------- 3. ask
with tab_a:
    engine = get_engine(key, results)
    rel_txt = "\n".join(f"{x.child_table}.{x.column} -> {x.parent_table}.{x.column}" for x in rel.itertuples()) if not rel.empty else ""
    st.markdown("Ask in plain English. The agent writes the SQL, runs it on your cleaned data, draws a chart and "
                "explains the answer.")
    examples = (["How many orders by status?", "Monthly order trend", "Top 5 payment method by orders",
                 "Average age by customer segment"] if use_sample and not uploads else [])
    cols = st.columns(len(examples)) if examples else []
    picked = None
    for c, ex in zip(cols, examples):
        if c.button(ex, width="stretch"):
            picked = ex
    with st.form("ask", clear_on_submit=True):
        q = st.text_input("Your question", placeholder="e.g. Which city has the most customers?")
        sent = st.form_submit_button("Ask the agent", type="primary")
    question = picked or (q.strip() if sent and q.strip() else None)
    # Public demo guard: cap AI questions per visit so one visitor can't use up the API quota
    max_q = int(os.environ.get("MAX_AI_QUESTIONS", "15"))
    used = st.session_state.setdefault("ai_questions_used", 0)
    if question and not isinstance(llm, MockLLM) and used >= max_q:
        st.warning(f"This demo answers up to {max_q} AI questions per visit. Switch the sidebar to "
                   "Offline to keep exploring, or reload the page later.")
        question = None
    if question:
        with st.spinner("Writing SQL and analysing…"):
            st.session_state.answers.insert(0, ask(question, engine, llm, rel_txt))
        if not isinstance(llm, MockLLM):
            st.session_state.ai_questions_used = used + 1
    for i, ans in enumerate(st.session_state.answers):
        st.markdown(f"#### {ans.question}")
        if ans.error:
            st.error(ans.text)
        else:
            html_text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(ans.text))
            st.markdown(f'<div class="answer">{html_text}</div>', unsafe_allow_html=True)
            fig = make_chart(ans.data, ans.chart)
            c1, c2 = st.columns([3, 2]) if fig else (st.container(), None)
            if fig:
                with c1:
                    show(fig)
                with c2:
                    st.dataframe(tidy(ans.data), hide_index=True, width="stretch", height=330)
            else:
                st.dataframe(tidy(ans.data), hide_index=True, width="stretch")
        with st.expander("Show the SQL"):
            st.code(ans.sql, language="sql")
            if ans.assumption:
                st.caption(ans.assumption)
        if i == 0 and len(st.session_state.answers) > 1:
            st.divider()

# ---------------------------------------------------------------- 4. power bi
with tab_p:
    model = build_model(results, rel)
    md = model_md(model)
    st.markdown("A ready-to-follow plan for building this data as a Power BI dashboard: model, DAX measures and pages.")
    st.download_button("⬇ Power BI build guide (Markdown)", md.encode(), "powerbi_build_guide.md")
    st.markdown(md)

# ---------------------------------------------------------------- 5. report
with tab_r:
    st.markdown("The final report combines data quality, key findings, the questions you asked and prioritised "
                "recommendations.")
    if st.button("Generate the final report", type="primary"):
        with st.spinner("Writing the report…"):
            st.session_state.report = build_report(results, rel, st.session_state.answers, llm)
    if st.session_state.get("report"):
        st.download_button("⬇ Final report (Markdown)", st.session_state.report.encode(), "final_report.md")
        st.markdown(st.session_state.report)
