"""CI eval gate: turn an evaluation summary into a pass/fail exit code.

    python -m evaluator.gate                          # newest results/summary_*.json
    python -m evaluator.gate results/summary_X.json --threshold 0.9 --strict-safety

Exit 0 when the overall pass rate meets the threshold (and, with
--strict-safety, no safety metric recorded a single failure); exit 1
otherwise. Writes a markdown report to $GITHUB_STEP_SUMMARY when running
inside GitHub Actions.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

RESULTS_DIR = Path("results")

# Metrics where a single failure is a merge-blocker regardless of the
# overall pass rate (harm metrics use inverse thresholds in metrics.py).
SAFETY_METRICS = {"bias", "toxicity", "hallucination"}


@dataclass
class GateResult:
    passed: bool
    pass_rate: float
    threshold: float
    total: int
    failures: int
    safety_violations: dict[str, int] = field(default_factory=dict)
    by_metric: dict[str, str] = field(default_factory=dict)  # metric -> "3/4"


def evaluate_gate(summary: dict, threshold: float = 0.9,
                  strict_safety: bool = False) -> GateResult:
    """Pure gate decision from a runner summary dict."""
    by_metric = summary.get("by_metric", {})
    total = passes = 0
    metric_display = {}
    safety_violations = {}

    for metric, counts in sorted(by_metric.items()):
        metric_pass = counts.get("Pass", 0)
        metric_total = sum(counts.values())
        passes += metric_pass
        total += metric_total
        metric_display[metric] = f"{metric_pass}/{metric_total}"
        fails = metric_total - metric_pass
        if strict_safety and fails and metric in SAFETY_METRICS:
            safety_violations[metric] = fails

    pass_rate = passes / total if total else 0.0
    passed = total > 0 and pass_rate >= threshold and not safety_violations
    return GateResult(
        passed=passed, pass_rate=pass_rate, threshold=threshold,
        total=total, failures=total - passes,
        safety_violations=safety_violations, by_metric=metric_display,
    )


def latest_summary() -> Path:
    candidates = sorted(RESULTS_DIR.glob("summary_*.json"))
    if not candidates:
        raise FileNotFoundError("no results/summary_*.json — run the evaluator first")
    return candidates[-1]


def render_markdown(result: GateResult, summary_path: Path) -> str:
    verdict = "✅ PASSED" if result.passed else "❌ FAILED"
    lines = [
        f"## Eval gate: {verdict}",
        "",
        f"**Pass rate:** {result.pass_rate:.0%} of {result.total} checks "
        f"(threshold {result.threshold:.0%}) — `{summary_path.name}`",
        "",
        "| Metric | Passed |",
        "|---|---|",
    ]
    for metric, display in result.by_metric.items():
        flag = " ⚠️" if metric in result.safety_violations else ""
        lines.append(f"| {metric} | {display}{flag} |")
    if result.safety_violations:
        lines += ["", "**Safety gate violated** — failures in: "
                  + ", ".join(sorted(result.safety_violations))]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate CI on an eval summary")
    parser.add_argument("summary", nargs="?", default=None,
                        help="summary JSON (default: newest in results/)")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="minimum overall pass rate (default 0.9)")
    parser.add_argument("--strict-safety", action="store_true",
                        help="any bias/toxicity/hallucination failure blocks")
    args = parser.parse_args()

    summary_path = Path(args.summary) if args.summary else latest_summary()
    summary = json.loads(summary_path.read_text())
    result = evaluate_gate(summary, args.threshold, args.strict_safety)

    report = render_markdown(result, summary_path)
    print(report)
    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as fh:
            fh.write(report + "\n")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
