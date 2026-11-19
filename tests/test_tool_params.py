"""工具参数抽取测试。

这层最容易退化成"硬编码关键词表"：一旦写死，换语料就静默失效，
所以用测试锁定"参数来自查询 ∩ 语料词表"这个契约。
"""
from __future__ import annotations

import unittest

from src.agents import tool_params
from src.data.corpus import CLINICAL


class TestToolParams(unittest.TestCase):
    def test_pubmed_keywords_come_from_query(self):
        params = tool_params.pubmed_params("EGFR 突变非小细胞肺癌的一线治疗方案")
        self.assertTrue(params["keywords"])
        self.assertIn("egfr", params["keywords"])
        self.assertLessEqual(len(params["keywords"]), 5)

    def test_clinical_indication_exists_in_trial_corpus(self):
        """适应证必须是试验语料里出现过的词，否则工具必然返回空。"""
        params = tool_params.clinical_params("SGLT2 inhibitors for type 2 diabetes")
        indication = params["indication"]
        corpus = " ".join(f"{d.title} {d.text}" for d in CLINICAL).lower()
        self.assertIn(indication, corpus)

    def test_umls_prefers_abbreviation(self):
        self.assertEqual(tool_params.umls_params("MS 的诊断标准是什么")["concept"], "ms")
        self.assertEqual(tool_params.umls_params("EGFR 突变肺癌")["concept"], "egfr")

    def test_abbreviation_needs_word_boundary(self):
        """"systems" 里的 "ms" 不是缩写，子串匹配会误判成多发性硬化。"""
        self.assertNotEqual(tool_params.umls_params("how do operating systems schedule threads")["concept"], "ms")

    def test_expanded_query_replaces_abbreviation_by_chosen_sense(self):
        """就地替换而不是追加：裸 "ms" 会命中 "LC-MS/MS"，追加等于没消歧。"""
        disease = tool_params.expanded_query("MS 的诊断标准是什么")
        self.assertIn("multiple sclerosis", disease)
        self.assertNotIn("mass spectrometry", disease)
        self.assertNotIn("MS", disease)

        method = tool_params.expanded_query("MS 定量检测的灵敏度与通量")
        self.assertIn("mass spectrometry", method)
        self.assertNotIn("multiple sclerosis", method)

    def test_expanded_query_without_cue_does_not_guess(self):
        """无线索时不猜方向：保持原样，交由检索与 Critic 处理。"""
        self.assertEqual(tool_params.expanded_query("MS"), "MS")

    def test_expanded_query_keeps_original_when_no_abbreviation(self):
        q = "SGLT2 inhibitors and kidney outcomes"
        self.assertEqual(tool_params.expanded_query(q), q)


if __name__ == "__main__":
    unittest.main()
