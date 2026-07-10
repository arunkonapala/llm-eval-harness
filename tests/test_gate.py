from evaluator.gate import evaluate_gate


def _summary(**metrics):
    return {"by_metric": {name: counts for name, counts in metrics.items()}}


def test_all_pass_meets_threshold():
    result = evaluate_gate(_summary(
        answer_relevancy={"Pass": 4}, correctness={"Pass": 4},
    ), threshold=0.9)
    assert result.passed
    assert result.pass_rate == 1.0
    assert result.failures == 0


def test_below_threshold_fails():
    result = evaluate_gate(_summary(
        answer_relevancy={"Pass": 2, "Fail": 2},  # 50%
        correctness={"Pass": 4},
    ), threshold=0.9)
    assert not result.passed
    assert result.pass_rate == 0.75


def test_threshold_boundary_is_inclusive():
    result = evaluate_gate(_summary(a={"Pass": 9, "Fail": 1}), threshold=0.9)
    assert result.passed


def test_strict_safety_blocks_despite_high_pass_rate():
    result = evaluate_gate(_summary(
        answer_relevancy={"Pass": 10},
        correctness={"Pass": 10},
        toxicity={"Pass": 9, "Fail": 1},   # 96.7% overall, but a tox failure
    ), threshold=0.9, strict_safety=True)
    assert not result.passed
    assert result.safety_violations == {"toxicity": 1}


def test_non_safety_failures_dont_trip_safety_gate():
    result = evaluate_gate(_summary(
        correctness={"Pass": 9, "Fail": 1},
        toxicity={"Pass": 10},
    ), threshold=0.9, strict_safety=True)
    assert result.passed
    assert result.safety_violations == {}


def test_empty_summary_fails_closed():
    result = evaluate_gate({"by_metric": {}}, threshold=0.5)
    assert not result.passed
    assert result.total == 0
