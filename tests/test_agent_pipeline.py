"""端到端行为测试：证据去重、反思有界、语料外问题不得谎称有据。

这三条是项目最重要也最容易回归的行为：
- 反思回流若重置证据，Recall 会随轮次下降；
- 反思若无上限，会死循环烧 token；
- grounded 判定若用重写后的查询算命中，语料外问题会被判"证据充分"，
  这是最危险的一类幻觉（看起来有依据，实际一个证据都没有）。
"""
from __future__ import annotations

import unittest

from src.agents.graph import run
from src.config import get_settings


def ask(question: str) -> dict:
    import asyncio

    return asyncio.run(run(question, verbose=False))


class TestPipelineBehaviour(unittest.TestCase):
    def test_evidence_ids_are_unique(self):
        result = ask("MS 的诊断标准是什么？")
        ids = [e["doc_id"] for e in result.get("evidences", [])]
        self.assertEqual(len(ids), len(set(ids)), "并行 agent 重复召回未去重")

    def test_reflection_is_bounded(self):
        result = ask("CRISPR base editing 在镰状细胞病中的长期安全性如何？")
        self.assertLessEqual(result.get("reflection_count", 0), get_settings().max_reflections)

    def test_out_of_corpus_question_is_not_grounded(self):
        """语料里没有的主题必须判为证据不足，而不是硬凑答案。"""
        result = ask("CRISPR base editing 在镰状细胞病中的长期安全性如何？")
        self.assertFalse(result.get("grounded"), "语料外问题被判为证据充分")
        self.assertEqual(result.get("evidences"), [])

    def test_in_corpus_question_returns_citations(self):
        result = ask("EGFR 突变非小细胞肺癌的一线治疗方案是什么？")
        self.assertTrue(result.get("grounded"))
        self.assertTrue(result.get("citations"))
        evidence_ids = {e["doc_id"] for e in result.get("evidences", [])}
        self.assertTrue(set(result["citations"]) <= evidence_ids, "引用了不存在的证据")


if __name__ == "__main__":
    unittest.main()
