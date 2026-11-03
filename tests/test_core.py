import io

import numpy as np
import pandas as pd

from agent import clean_table, find_relationships, load_bytes, quality_scores
from agent.cleaner import clean_all


def messy():
    df = pd.DataFrame({
        "Order ID": ["A1", "A2", "A3", "A4", "A4", "A5"],
        "Amount": ["$1,200", "300", "N/A", "45.5", "45.5", "(20)"],
        "Qty": [1, -3, 2, 2, 2, 5],
        "Region": ["North", "north ", "South", None, None, "South"],
        "Date": ["2025-01-01", "2025-02-01", "bad", "2025-03-01", "2025-03-01", "2025-04-01"],
    })
    return df


def test_cleaning_rules():
    r = clean_table(messy(), "t")
    c = r.clean
    assert len(c) == 5                                   # exact duplicate removed
    assert c["amount"].dtype.kind == "f"                 # text numbers converted
    assert c["amount"].isna().sum() == 0                 # N/A filled
    assert (c["qty"] >= 0).all()                         # negative quantity fixed
    assert set(c["region"]) <= {"North", "South", "Unknown"}
    assert str(c["date"].dtype).startswith("datetime64")
    assert r.key_column == "order_id"
    assert quality_scores(c)["overall"] > quality_scores(r.raw)["overall"]


def test_excel_multisheet():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        messy().to_excel(w, sheet_name="A", index=False)
        pd.DataFrame({"x": [1]}).to_excel(w, sheet_name="B", index=False)
    t = load_bytes(buf.getvalue(), "file.xlsx")
    assert len(t) == 2


def test_orphans():
    parent = pd.DataFrame({"customer_id": [1, 2, 3], "name": ["a", "b", "c"]})
    child = pd.DataFrame({"order_id": np.arange(10, 16), "customer_id": [1, 2, 3, 1, 99, 2]})
    rel = find_relationships(clean_all({"customers": parent, "orders": child}))
    row = rel[(rel.child_table == "orders")].iloc[0]
    assert row.orphan_rows == 1 and "99" in row.orphan_examples
