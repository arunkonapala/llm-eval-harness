"""Evaluation runner: load test cases from CSV, generate candidate responses,
score them with the metric suite, and write consolidated results.

    python -m evaluator.runner data/sample_testcases.csv [--limit N]

Outputs results/consolidated_results_<timestamp>.csv plus a summary JSON with
pass/fail counts per model and per metric.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from deepeval.test_case import LLMTestCase

from evaluator.metrics import MetricSuite
from evaluator.models import RateLimitExhausted, get_candidates, get_judge

RESULTS_DIR = Path("results")


def build_candidate_prompt(row: pd.Series) -> str:
    if row.get("context") and pd.notna(row.get("context")):
        return f"Context:\n{row['context']}\n\nQuestion: {row['prompt']}"
    return row["prompt"]


def build_test_case(row: pd.Series, actual_output: str) -> LLMTestCase:
    context = [row["context"]] if row.get("context") and pd.notna(row.get("context")) else None
    expected = row.get("expected_response")
    return LLMTestCase(
        input=row["prompt"],
        actual_output=actual_output,
        expected_output=expected if pd.notna(expected) else None,
        context=context,
    )


def run(testcases_path: str, limit: int | None = None) -> Path:
    df = pd.read_csv(testcases_path)
    if limit:
        df = df.head(limit)

    judge = get_judge()
    suite = MetricSuite(judge)
    rows = []
    quota_exhausted = False

    for candidate in get_candidates():
        if quota_exhausted:
            break
        model_name = candidate.get_model_name()
        print(f"\n=== Evaluating model: {model_name} ===")
        for _, tc in df.iterrows():
            if quota_exhausted:
                break
            print(f"  [{tc['case_id']}] {tc['category']}: {str(tc['prompt'])[:60]}...")
            try:
                actual = candidate.generate(build_candidate_prompt(tc))
            except Exception as exc:
                print(f"    candidate generation failed: {exc}", file=sys.stderr)
                continue

            test_case = build_test_case(tc, actual)
            for metric_name, metric_fn in suite.applicable_metrics(test_case).items():
                try:
                    score, reason, verdict = metric_fn(test_case)
                except RateLimitExhausted as exc:
                    # Every remaining judge call would fail the same way, so
                    # stop here rather than issuing dozens of doomed requests.
                    print(f"    {metric_name}: quota exhausted — {exc}", file=sys.stderr)
                    quota_exhausted = True
                except Exception as exc:  # judge/parse failure — record, keep going
                    score, reason, verdict = None, f"metric error: {exc}", "Error"
                if quota_exhausted:
                    break
                rows.append({
                    "case_id": tc["case_id"],
                    "category": tc["category"],
                    "prompt": tc["prompt"],
                    "actual_response": actual,
                    "model": model_name,
                    "metric": metric_name,
                    "score": score,
                    "reason": reason,
                    "result": verdict,
                })
                print(f"    {metric_name}: {verdict} ({score})")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = RESULTS_DIR / f"consolidated_results_{stamp}.csv"
    results = pd.DataFrame(rows)
    results.to_csv(out_csv, index=False)

    def tally(key: str) -> dict:
        if results.empty:
            return {}
        return results.groupby([key, "result"]).size().unstack(fill_value=0).to_dict("index")

    summary = {
        "run_at": stamp,
        "testcases": len(df),
        "quota_exhausted": quota_exhausted,
        "by_model": tally("model"),
        "by_metric": tally("metric"),
    }
    (RESULTS_DIR / f"summary_{stamp}.json").write_text(json.dumps(summary, indent=2))

    print(f"\nWrote {out_csv}")
    print(json.dumps(summary, indent=2))
    return out_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM evaluation over a test-case CSV")
    parser.add_argument("testcases", nargs="?", default="data/sample_testcases.csv")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N test cases")
    args = parser.parse_args()
    run(args.testcases, args.limit)
