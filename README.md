# LLM Eval Harness

A model-agnostic LLM evaluation harness built on **DeepEval**: run a CSV of
test cases against one or more candidate models, score every response across
quality and safety metrics with an LLM judge, and get consolidated pass/fail
reports per model and per metric.

```
data/testcases.csv ──► candidate model(s) ──► actual responses
                                │
                                ▼
                     DeepEval metric suite (LLM judge)
                                │
                                ▼
        results/consolidated_results_<ts>.csv + summary JSON
                                │
                                ▼
                  Flask API (/results, /counts, /run)
```

## Metrics

| Metric | Type | Passes when | Requires |
|---|---|---|---|
| Answer relevancy | quality | score ≥ 0.5 | prompt only |
| Correctness (G-Eval) | quality | score ≥ 0.5 | `expected_response` |
| Faithfulness | quality | score ≥ 0.5 | retrieval context |
| Bias | safety | score ≤ 0.5 | prompt only |
| Toxicity | safety | score ≤ 0.5 | prompt only |
| Hallucination | safety | score ≤ 0.5 | `context` |

Metrics are selected per test case based on which fields it provides —
grounded cases get hallucination checks, cases with expected answers get
G-Eval correctness, everything gets relevancy/bias/toxicity.

## Test cases

`data/sample_testcases.csv` ships with 10 finance-domain cases across five
categories: adversarial input (special characters, prompt injection), bias &
fairness (demographic neutrality), hallucination (grounded + unanswerable),
relevance, and robustness. Add your own rows — columns are
`case_id, category, subcategory, prompt, expected_response, context`.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp config.ini.example config.ini      # defaults target Groq's free tier
export LLM_API_KEY=gsk_...            # or the env var you name in config.ini

# Run the evaluation
.venv/bin/python -m evaluator.runner data/sample_testcases.csv --limit 3

# Or serve the API
.venv/bin/flask --app app run --port 8080
curl -X POST localhost:8080/run -H 'Content-Type: application/json' -d '{"limit": 3}'
curl localhost:8080/results
```

**Judge and candidates are independently configurable** in `config.ini`:
any OpenAI-compatible endpoint (Groq, Gemini, Ollama) or AWS Bedrock via the
Converse API. List multiple candidate sections to benchmark models
side-by-side; the results CSV carries a `model` column for comparison.

## Design notes

- **Inverse thresholds for harm metrics** — bias/toxicity/hallucination
  scores measure the *presence* of the problem, so they pass low.
- **Per-metric error isolation** — a judge parse failure records an `Error`
  verdict for that metric and the run continues.
- **No secrets in config** — `config.ini` names an env var per provider and
  is gitignored; the example file carries no credentials.

## Roadmap

- Angular dashboard over the Flask API (results by category, model diff view)
- Latency + token-cost columns alongside quality scores
- CI gate: fail a PR when pass-rate drops below a threshold
