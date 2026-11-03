"""
run_agent.py — run the agent from the command line.

    python run_agent.py sample_data/retail_orders.csv sample_data/retail_customers.csv

Writes outputs/data_quality_report.md and outputs/cleaned/<table>.csv
"""

import sys
from pathlib import Path

from agent import analyze, clean_all, find_relationships, load_many, quality_report_md


def main(paths):
    out = Path("outputs")
    (out / "cleaned").mkdir(parents=True, exist_ok=True)
    tables = load_many(paths)
    results = clean_all(tables)
    rel = find_relationships(results)
    report = quality_report_md(results, rel)
    report += "\n\n# Automatic insights\n"
    for name, r in results.items():
        r.clean.to_csv(out / "cleaned" / f"{name}.csv", index=False)
        report += f"\n## `{name}`\n" + "\n".join(f"- {s}" for s in analyze(r.clean, name)["insights"]) + "\n"
    (out / "data_quality_report.md").write_text(report, encoding="utf-8")
    print(f"Done: {len(results)} table(s) cleaned -> {out / 'data_quality_report.md'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
