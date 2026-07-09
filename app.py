"""Minimal Flask API over the evaluation results.

    flask --app app run --port 8080

GET  /testcases   — the current test-case set
GET  /counts      — test-case counts by category
GET  /results     — latest consolidated results, grouped by model
POST /run         — execute an evaluation run (body: {"limit": N} optional)
"""

import json
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

from evaluator.runner import RESULTS_DIR, run

TESTCASES = Path("data/sample_testcases.csv")

app = Flask(__name__)


def _latest(pattern: str) -> Path | None:
    files = sorted(RESULTS_DIR.glob(pattern))
    return files[-1] if files else None


@app.get("/testcases")
def testcases():
    return pd.read_csv(TESTCASES).to_dict(orient="records")


@app.get("/counts")
def counts():
    df = pd.read_csv(TESTCASES)
    return jsonify({
        "total_testcases": len(df),
        "by_category": df.groupby("category").size().to_dict(),
    })


@app.get("/results")
def results():
    latest_csv = _latest("consolidated_results_*.csv")
    if not latest_csv:
        return jsonify({"error": "no results yet — POST /run first"}), 404
    df = pd.read_csv(latest_csv)
    latest_summary = _latest("summary_*.json")
    return jsonify({
        "results": {m: g.to_dict(orient="records") for m, g in df.groupby("model")},
        "summary": json.loads(latest_summary.read_text()) if latest_summary else None,
    })


@app.post("/run")
def run_eval():
    limit = (request.get_json(silent=True) or {}).get("limit")
    out = run(str(TESTCASES), limit=limit)
    return jsonify({"status": "complete", "results_file": str(out)})
