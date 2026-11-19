"""共享状态 reducer 测试：并行归并 + 反思去重累积。"""
from __future__ import annotations

import unittest

from src.state import merge_dict, merge_evidences


def ev(doc_id: str, score: float = 0.0) -> dict:
    return {"doc_id": doc_id, "source": "literature", "title": doc_id, "snippet": "",
            "url": "", "study_type": "", "evidence_level": "1", "score": score}


class TestReducers(unittest.TestCase):
    def test_merge_evidences_appends(self):
        result = merge_evidences([ev("A")], [ev("B")])
        self.assertEqual([e["doc_id"] for e in result], ["A", "B"])

    def test_merge_evidences_dedupes_by_doc_id(self):
        """并行 agent 会召回同一篇文档，重复项必须按 doc_id 合并。"""
        result = merge_evidences([ev("A"), ev("B")], [ev("B"), ev("C")])
        self.assertEqual([e["doc_id"] for e in result], ["A", "B", "C"])

    def test_merge_evidences_keeps_higher_score(self):
        result = merge_evidences([ev("A", 0.3)], [ev("A", 0.9)])
        self.assertEqual(result[0]["score"], 0.9)

    def test_merge_evidences_keeps_previous_round(self):
        """反思回流不得丢掉上一轮证据：累积优于重置（否则 Recall 随轮次下降）。"""
        result = merge_evidences([ev("A"), ev("B")], [ev("C")])
        self.assertEqual([e["doc_id"] for e in result], ["A", "B", "C"])

    def test_merge_dict_accumulates(self):
        self.assertEqual(merge_dict({"a": 2}, {"a": 3, "b": 1}), {"a": 5, "b": 1})

    def test_merge_dict_handles_none(self):
        self.assertEqual(merge_dict(None, {"x": 1}), {"x": 1})
        self.assertEqual(merge_dict({"x": 1}, None), {"x": 1})


if __name__ == "__main__":
    unittest.main()
