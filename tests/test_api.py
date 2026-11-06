"""Tests for the headless pipeline and the HTTP API (offline, no API key needed)."""

import json
import os
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

os.environ["AI_PROVIDER"] = "mock"

import api  # noqa: E402
from agent.pipeline import md_to_html, run_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ORDERS = ROOT / "sample_data" / "retail_orders.csv"


def test_pipeline_on_retail_sample():
    files = {p.name: p.read_bytes() for p in sorted((ROOT / "sample_data").glob("retail_*.csv"))}
    out = run_pipeline(files, questions=["orders by status", "total revenue by region"])
    assert out["tables"] and 0 < out["quality_score"] <= 100
    assert all(t["score_after"] >= t["score_before"] for t in out["tables"])
    assert out["recommendations"] and out["answers"][0]["error"] == ""
    assert "can't find a column" in out["answers"][1]["error"]  # no guessing
    assert "<table" in out["report_html"] and "# Final Analysis Report" in out["report_md"]


def test_md_to_html_escapes_and_formats():
    h = md_to_html("# Title\n- **bold** <script>\n\n| a | b |\n|---|---|\n| 1 | 2 |")
    assert "<h1" in h and "<b>bold</b>" in h and "&lt;script&gt;" in h and "<td" in h


def _serve():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _post(url, data=b"", headers=None):
    req = urllib.request.Request(url, data=data, method="POST", headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_api_health_upload_and_errors():
    srv, base = _serve()
    try:
        with urllib.request.urlopen(base + "/health", timeout=10) as r:
            assert json.loads(r.read()) == {"status": "ok"}
        code, out = _post(base + "/analyze?filename=retail_orders.csv&question=how+many+orders", ORDERS.read_bytes())
        assert code == 200 and out["tables"][0]["table"] and out["answers"]
        code, out = _post(base + "/analyze?sample=retail")
        assert code == 200 and len(out["tables"]) >= 2
        assert _post(base + "/analyze")[0] == 400
        assert _post(base + "/analyze?sample=nope")[0] == 400
    finally:
        srv.shutdown()


def test_api_token():
    old = api.TOKEN
    api.TOKEN = "secret-test-token"
    srv, base = _serve()
    try:
        assert _post(base + "/analyze?sample=retail")[0] == 401
        assert _post(base + "/analyze?sample=retail", headers={"X-Agent-Token": "secret-test-token"})[0] == 200
    finally:
        api.TOKEN = old
        srv.shutdown()
