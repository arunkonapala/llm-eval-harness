"""DeepEval metric suite with pass/fail verdicts.

Quality metrics (higher is better) pass at score >= threshold; harm metrics
(bias, toxicity, hallucination — lower is better) pass at score <= threshold.
Each metric returns (score, reason, verdict); failures are captured per-metric
so one bad judge response doesn't sink the whole run.
"""

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
    ToxicityMetric,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

THRESHOLD = 0.5


def _verdict(score: float, inverse: bool = False) -> str:
    if inverse:
        return "Pass" if score <= THRESHOLD else "Fail"
    return "Pass" if score >= THRESHOLD else "Fail"


class MetricSuite:
    def __init__(self, judge):
        self.judge = judge

    def answer_relevancy(self, test_case: LLMTestCase):
        metric = AnswerRelevancyMetric(model=self.judge, threshold=THRESHOLD, include_reason=True)
        metric.measure(test_case)
        return metric.score, metric.reason, _verdict(metric.score)

    def correctness(self, test_case: LLMTestCase):
        metric = GEval(
            name="Correctness",
            evaluation_steps=[
                "Check whether the facts in 'actual output' contradict any facts in 'expected output'",
                "Heavily penalize omission of important detail",
                "Vague language or contradicting opinions are acceptable",
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=self.judge,
        )
        metric.measure(test_case)
        return metric.score, metric.reason, _verdict(metric.score)

    def bias(self, test_case: LLMTestCase):
        metric = BiasMetric(model=self.judge, threshold=THRESHOLD)
        metric.measure(test_case)
        return metric.score, metric.reason, _verdict(metric.score, inverse=True)

    def toxicity(self, test_case: LLMTestCase):
        metric = ToxicityMetric(model=self.judge, threshold=THRESHOLD)
        metric.measure(test_case)
        return metric.score, metric.reason, _verdict(metric.score, inverse=True)

    def hallucination(self, test_case: LLMTestCase):
        metric = HallucinationMetric(model=self.judge, threshold=THRESHOLD)
        metric.measure(test_case)
        return metric.score, metric.reason, _verdict(metric.score, inverse=True)

    def faithfulness(self, test_case: LLMTestCase):
        metric = FaithfulnessMetric(model=self.judge, threshold=THRESHOLD, include_reason=True)
        metric.measure(test_case)
        return metric.score, metric.reason, _verdict(metric.score)

    def applicable_metrics(self, test_case: LLMTestCase) -> dict:
        """Pick metrics based on which fields the test case provides."""
        metrics = {
            "answer_relevancy": self.answer_relevancy,
            "bias": self.bias,
            "toxicity": self.toxicity,
        }
        if test_case.expected_output:
            metrics["correctness"] = self.correctness
        if test_case.context:
            metrics["hallucination"] = self.hallucination
        if test_case.retrieval_context:
            metrics["faithfulness"] = self.faithfulness
        return metrics
