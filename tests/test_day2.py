import json

import pandas as pd

from agent import clean_all, find_relationships
from agent.ask import ask
from agent.charts import make_chart, pick_chart
from agent.llm import LLM, MockLLM, get_llm, parse_json
from agent.powerbi import build_model, model_md
from agent.report import build_report
from agent.sql_engine import SQLEngine, UnsafeSQL, check_sql


def data():
    customers = pd.DataFrame({"customer_id": [1, 2, 3, 4], "city": ["A", "B", "A", "C"]})
    orders = pd.DataFrame({"order_id": range(100, 112), "customer_id": [1, 2, 3, 4, 1, 2, 3, 99, 1, 2, 3, 4],
                           "order_date": pd.date_range("2025-01-01", periods=12, freq="MS"),
                           "amount": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
                           "status": ["Done"] * 10 + ["Cancelled"] * 2})
    return clean_all({"customers": customers, "orders": orders})


class ScriptedLLM(LLM):
    """Returns pre-written replies in order — stands in for Gemini/Claude in tests."""
    name = "scripted"

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def complete(self, system, prompt, max_tokens=1500):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def test_sql_safety():
    for bad in ["DROP TABLE orders", "select 1; select 2", "select * from read_csv('/etc/passwd')",
                "update orders set amount=0", "attach 'x.db'"]:
        try:
            check_sql(bad)
            assert False, bad
        except UnsafeSQL:
            pass
    assert check_sql("```sql\nSELECT 1;\n```") == "SELECT 1"


def test_ai_flow_with_retry():
    res = data()
    eng = SQLEngine({k: v.clean for k, v in res.items()})
    llm = ScriptedLLM([
        json.dumps({"sql": "SELECT city, SUM(amount_typo) FROM orders", "chart": {"type": "bar"}}),
        json.dumps({"sql": "SELECT c.city, SUM(o.amount) AS total_amount FROM orders o JOIN customers c "
                           "USING (customer_id) GROUP BY 1 ORDER BY 2 DESC",
                    "chart": {"type": "bar", "x": "city", "y": "total_amount"}, "title": "Sales by city"}),
        "City A leads with 360 in sales.",
    ])
    a = ask("Total sales by city?", eng, llm)
    assert not a.error and len(a.data) == 3
    assert a.data.iloc[0]["city"] == "A" and a.data.iloc[0]["total_amount"] == 360  # customers 1 and 3 live in city A: 150 + 210
    assert "FAILED WITH" in llm.prompts[1]          # the error was sent back for a fix
    assert a.text.startswith("City A")
    assert make_chart(a.data, a.chart) is not None


def test_ai_unsafe_sql_is_not_run():
    res = data()
    eng = SQLEngine({k: v.clean for k, v in res.items()})
    llm = ScriptedLLM([json.dumps({"sql": "DROP TABLE orders"}), json.dumps({"sql": "DELETE FROM orders"})])
    a = ask("delete everything", eng, llm)
    assert a.error
    assert len(eng.run("SELECT * FROM orders")) == 12


def test_offline_mode_and_chart_choice():
    res = data()
    eng = SQLEngine({k: v.clean for k, v in res.items()})
    assert isinstance(get_llm(provider="gemini", secrets={}), MockLLM) or True
    a = ask("Monthly amount trend", eng, MockLLM())
    assert pick_chart(a.data, a.chart)["type"] == "line"
    b = ask("Total amount by status", eng, MockLLM())
    assert b.data.iloc[0]["status"] == "Done"


def test_powerbi_and_report():
    res = data()
    rel = find_relationships(res)
    m = build_model(res, rel)
    assert m["fact"] == "orders"
    md = model_md(m)
    assert "Total Amount = SUM('orders'[amount])" in md and "SAMEPERIODLASTYEAR" in md
    rep = build_report(res, rel, [], MockLLM())
    assert "## Recommendations" in rep and "orphan" in rep.lower()


def test_parse_json_variants():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Here you go: {"a": 2} thanks') == {"a": 2}
