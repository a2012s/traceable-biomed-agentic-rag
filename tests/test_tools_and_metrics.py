"""工具层（强参数校验/错误二分）与评估指标测试。"""
from __future__ import annotations

import asyncio
import unittest

from pydantic import ValidationError

from src.eval.golden_set import GoldenCase
from src.eval.metrics import generation_metrics, retrieval_metrics
from src.tools.biomedical import FatalToolError, TrialQuery, dispatch


class TestToolValidation(unittest.TestCase):
    def test_valid_params(self):
        result = asyncio.run(dispatch("clinicaltrials_lookup", {"indication": "diabetes"}))
        self.assertIsInstance(result, list)

    def test_invalid_params_raise_fatal(self):
        """非法参数属不可恢复错误，必须立即失败而非重试。"""
        with self.assertRaises(FatalToolError):
            asyncio.run(dispatch("clinicaltrials_lookup", {"indication": "x"}))  # 太短

    def test_unknown_tool_raise_fatal(self):
        with self.assertRaises(FatalToolError):
            asyncio.run(dispatch("no_such_tool", {}))

    def test_phase_pattern(self):
        self.assertEqual(TrialQuery(indication="lung", phase="phase-3").phase, "phase-3")
        with self.assertRaises(ValidationError):
            TrialQuery(indication="lung", phase="phase-9")


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.case = GoldenCase(
            question="q",
            relevant=frozenset({"A", "B"}),
            distractors=frozenset({"X"}),
        )

    def test_retrieval_metrics(self):
        m = retrieval_metrics(["X", "B", "A"], self.case, k=3)
        self.assertAlmostEqual(m.recall, 1.0)
        self.assertAlmostEqual(m.mrr, 1.0 / 2)  # 第一条相关 B 在 rank2
        self.assertAlmostEqual(m.precision, 2 / 3)

    def test_generation_metrics_detects_hallucinated_citation(self):
        m = generation_metrics(
            citation_ids=["A", "FAKE"],
            evidence_ids=["A", "X"],
            case=self.case,
            grounded=False,
        )
        self.assertAlmostEqual(m.citation_precision, 0.5)
        self.assertEqual(m.hallucinated_citations, ["FAKE"])
        self.assertAlmostEqual(m.distractor_leakage, 1.0)  # 干扰文档 X 泄漏进证据

    def test_generation_metrics_clean(self):
        m = generation_metrics(["A", "B"], ["A", "B"], self.case, grounded=True)
        self.assertEqual(m.citation_precision, 1.0)
        self.assertEqual(m.distractor_leakage, 0.0)
        self.assertTrue(m.grounded)


if __name__ == "__main__":
    unittest.main()
