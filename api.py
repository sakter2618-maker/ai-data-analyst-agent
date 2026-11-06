"""
api.py — a small HTTP API so automation tools (n8n) can use the agent.

    GET  /health                      -> {"status": "ok"}
    POST /analyze?filename=sales.csv  -> body = the raw CSV/Excel file
         &question=...&question=...      (optional business questions)
    POST /analyze?sample=retail       -> runs on the bundled retail sample (demo / test)

Returns JSON: summary, quality score, per-table scores, recommendations, answers,
report_md and report_html (ready to e-mail).

Security
    * Optional shared token: set AGENT_API_TOKEN and send header  X-Agent-Token: <token>
    * Uploads are limited to MAX_UPLOAD_MB (default 25 MB)
    * The API is meant for the private Docker network (n8n -> agent), not the internet

Run:  python api.py            (port 8000, or PORT env var)
Uses only the Python standard library on top of the agent itself.
"""

from __future__ import annotations

import hmac
import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from agent.llm import get_llm
from agent.pipeline import run_pipeline

ROOT = Path(__file__).parent
MAX_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "25")) * 1024 * 1024)
TOKEN = os.environ.get("AGENT_API_TOKEN", "")
SAMPLES = {"retail": sorted((ROOT / "sample_data").glob("retail_*.csv")),
           "supply_chain": [ROOT / "sample_data" / "supply_chain_data.csv"]}


class Handler(BaseHTTPRequestHandler):
    server_version = "DataAnalystAgent/1.0"

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        return not TOKEN or hmac.compare_digest(self.headers.get("X-Agent-Token", ""), TOKEN)

    def do_GET(self):
        if urlparse(self.path).path.rstrip("/") == "/health":
            return self._send(200, {"status": "ok"})
        self._send(404, {"error": "Not found. Use GET /health or POST /analyze"})

    def do_POST(self):
        url = urlparse(self.path)
        if url.path.rstrip("/") != "/analyze":
            return self._send(404, {"error": "Not found. Use POST /analyze"})
        if not self._authorized():
            return self._send(401, {"error": "Missing or wrong X-Agent-Token"})
        q = parse_qs(url.query)
        try:
            if "sample" in q:
                paths = SAMPLES.get(q["sample"][0])
                if not paths:
                    return self._send(400, {"error": f"Unknown sample. Choose: {', '.join(SAMPLES)}"})
                files = {p.name: p.read_bytes() for p in paths}
            else:
                n = int(self.headers.get("Content-Length") or 0)
                if n <= 0:
                    return self._send(400, {"error": "Send the CSV/Excel file as the request body"})
                if n > MAX_BYTES:
                    return self._send(413, {"error": f"File larger than {MAX_BYTES // 1048576} MB"})
                name = (q.get("filename") or ["upload.csv"])[0]
                files = {Path(name).name: self.rfile.read(n)}
            result = run_pipeline(files, get_llm(), q.get("question", []))
            self._send(200, result)
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:  # keep the server alive, report the problem
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):  # one short line per request
        print(f"{self.address_string()} {fmt % args}", flush=True)


def main():
    port = int(os.environ.get("PORT", "8000"))
    print(f"Agent API on :{port}  (token {'on' if TOKEN else 'off'}, AI: {get_llm().name})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
