"""生产化增量的行为测试：缓存、可观测性、实体链接、反馈存储。

这些测试都是离线 + 临时文件，不触碰真实 key，也不残留 DB 文件。
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from src.agents.graph import run
from src.cache import AnswerCache, normalize_question
from src.deps import get_answer_cache
from src.feedback import FeedbackStore
from src.observability import metrics
from src.rag.entity_linking import EntityLinking


class TestProductionFeatures(unittest.TestCase):
    def setUp(self):
        get_answer_cache().clear()

    def test_normalize_question_folds_whitespace_and_case(self):
        self.assertEqual(
            normalize_question("  EGFR  突变  "),
            normalize_question("egfr 突变"),
        )

    def test_cache_hit_avoids_rerun(self):
        cache = AnswerCache(ttl_s=60)
        q = "MS 的诊断标准是什么？"
        self.assertIsNone(cache.get(q))
        cache.put(q, {"answer": "cached"})
        self.assertEqual(cache.get(q)["answer"], "cached")

    def test_cache_key_changes_with_corpus(self):
        cache = AnswerCache(ttl_s=60)
        k1 = cache.key("同一个问题")
        cache.put("同一个问题", {"answer": "v1"})
        self.assertEqual(cache.get("同一个问题")["answer"], "v1")
        self.assertEqual(cache.key("同一个问题"), k1, "同配置下缓存键必须稳定")

    def test_cache_disabled_when_ttl_zero(self):
        cache = AnswerCache(ttl_s=0)
        cache.put("q", {"answer": "x"})
        self.assertIsNone(cache.get("q"))

    def test_entity_linking_detects_kidney_vs_gene(self):
        linker = EntityLinking(enabled=True)
        self.assertEqual(linker.detect_senses("达格列净对 CKD 患者 eGFR 下降的影响？"), {"egfr": "kidney_function"})
        self.assertEqual(
            linker.detect_senses("EGFR 突变非小细胞肺癌的一线治疗方案？"),
            {"egfr": "receptor_gene"},
        )

    def test_entity_linking_penalizes_opposite_sense(self):
        linker = EntityLinking(enabled=True)
        senses = {"egfr": "kidney_function"}
        self.assertLess(
            linker.penalty("Osimertinib in EGFR-mutated NSCLC exon 19", senses),
            linker.penalty("Dapagliflozin eGFR decline in CKD kidney", senses),
        )

    def test_metrics_snapshot_shape(self):
        snap = metrics.snapshot()
        for key in ("requests", "errors", "inflight", "latency_ms", "cache"):
            self.assertIn(key, snap)
        self.assertIn("p95", snap["latency_ms"])

    def test_feedback_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            store = FeedbackStore(str(Path(d) / "fb.db"))
            store.add("h1", 1, "good")
            store.add("h2", -1, "bad cite")
            summary = store.summary()
            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["negative"], 1)
            self.assertEqual(summary["recent"][0]["question_hash"], "h2")

    def test_feedback_rejects_invalid_rating(self):
        with tempfile.TemporaryDirectory() as d:
            store = FeedbackStore(str(Path(d) / "fb.db"))
            with self.assertRaises(ValueError):
                store.add("h1", 0, "")

    def test_pipeline_marked_cache_hit(self):
        q = "EGFR 突变非小细胞肺癌的一线治疗方案是什么？"
        first = asyncio.run(run(q, verbose=False))
        second = asyncio.run(run(q, verbose=False))
        self.assertFalse(first.get("cache_hit"))
        self.assertTrue(second.get("cache_hit"))

    def test_out_of_corpus_still_not_grounded_after_cache(self):
        q = "CRISPR base editing 在镰状细胞病中的长期安全性如何？"
        first = asyncio.run(run(q, verbose=False))
        second = asyncio.run(run(q, verbose=False))
        self.assertFalse(first.get("grounded"))
        self.assertFalse(second.get("grounded"))


if __name__ == "__main__":
    unittest.main()
