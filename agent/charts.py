"""
charts.py — turn a query result + chart hint into a Plotly figure.
If the hint is missing or doesn't fit the data, the chart type is chosen from the data:
date + number -> line, category + number -> bar (horizontal when labels are many), else table.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def _style(fig, title=""):
    fig.update_layout(template="plotly_white", height=400, margin=dict(l=8, r=30, t=40 if title else 10, b=8),
                      title=dict(text=title, x=0.01, font=dict(size=15, color=INK)) if title else None,
                      font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12),
                      showlegend=False, bargap=0.3)
    fig.update_xaxes(showgrid=False, tickfont=dict(color=MUTED), automargin=True)
    fig.update_yaxes(gridcolor=GRID, tickfont=dict(color=MUTED), automargin=True)
    fig.update_traces(cliponaxis=False, selector=dict(type="bar"))
    return fig


def pick_chart(df: pd.DataFrame, hint: dict | None) -> dict:
    hint = dict(hint or {})
    cols = list(df.columns)
    if df.empty or len(cols) < 2:
        return {"type": "table"}
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    dates = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    x, y = hint.get("x"), hint.get("y")
    if x not in cols:
        x = dates[0] if dates else next((c for c in cols if c not in num), cols[0])
    if y not in cols or not pd.api.types.is_numeric_dtype(df[y]):
        y = next((c for c in reversed(num) if c != x), None)
    if y is None:
        return {"type": "table"}
    t = hint.get("type")
    if t not in ("bar", "line", "pie"):
        t = "line" if x in dates else "bar"
    if t == "pie" and (len(df) > 6 or (df[y] < 0).any()):
        t = "bar"  # pies only for a few parts of a whole
    return {"type": t, "x": x, "y": y}


def make_chart(df: pd.DataFrame, hint: dict | None = None, title: str = "") -> go.Figure | None:
    spec = pick_chart(df, hint)
    if spec["type"] == "table":
        return None
    x, y, t = spec["x"], spec["y"], spec["type"]
    d = df.head(50)
    if t == "line":
        d = d.sort_values(x)
        fig = go.Figure(go.Scatter(x=d[x], y=d[y], mode="lines+markers", line=dict(color=PALETTE[0], width=2),
                                   marker=dict(size=7), hovertemplate=f"%{{x}}<br>{y}: %{{y:,.2f}}<extra></extra>"))
    elif t == "pie":
        fig = go.Figure(go.Pie(labels=d[x].astype(str), values=d[y], hole=0.55,
                               marker=dict(colors=PALETTE, line=dict(color="#fff", width=2))))
        fig.update_layout(showlegend=True)
    else:
        horizontal = len(d) > 8 or d[x].astype(str).str.len().max() > 12
        labels = d[x].astype(str)
        if horizontal:
            d = d.iloc[::-1]
            labels = labels.iloc[::-1]
            fig = go.Figure(go.Bar(x=d[y], y=labels, orientation="h", marker_color=PALETTE[0],
                                   text=[f"{v:,.2f}".rstrip("0").rstrip(".") for v in d[y]], textposition="outside",
                                   hovertemplate=f"%{{y}}<br>{y}: %{{x:,.2f}}<extra></extra>"))
            fig.update_layout(height=max(320, 26 * len(d) + 80))
        else:
            fig = go.Figure(go.Bar(x=labels, y=d[y], marker_color=PALETTE[0],
                                   text=[f"{v:,.2f}".rstrip("0").rstrip(".") for v in d[y]], textposition="outside",
                                   hovertemplate=f"%{{x}}<br>{y}: %{{y:,.2f}}<extra></extra>"))
    fig = _style(fig, title)
    if t == "bar" and len(d) > 8:
        fig.update_layout(height=max(320, 26 * len(d) + 80))
    return fig
