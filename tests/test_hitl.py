"""Human-in-the-loop 断点的回归测试。

覆盖的是「图能停、能续、三种决定各走各的出口」这几件事，而不是措辞：
- 停：Critic 判不足且机器本来还要再检索时才中断（已有结论的请求不该被打断）；
- 续：approve / refine / abort 从同一检查点续跑，不重头执行；
- 边界：HITL 关闭时行为与旧版完全一致；非法决定不白白多跑一轮。
"""
from __future__ import annotations

import asyncio
import unittest
import uuid
from unittest.mock import patch

from langgraph.types import Command

from src.agents import graph as graph_module
from src.agents.graph import build_graph

# 语料内有证据但等级不足：触发"机器想再检索一轮"的分支，正是 HITL 的入口
INSUFFICIENT = "mass spectrometry 质谱分析在血清蛋白定量中的应用"
GROUNDED = "MS 的诊断标准是什么？"
# 语料外：证据为 0，检索已枯竭 → 不该再打断人（直接拒答）
OUT_OF_CORPUS = "CRISPR base editing 在镰状细胞病中的长期安全性如何？"


def _tid(name: str) -> str:
    return f"test-hitl-{name}-{uuid.uuid4().hex[:8]}"


class HitlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.app = build_graph(verbose=False)

    def _ask(self, question: str, tid: str) -> dict:
        return asyncio.run(
            self.app.ainvoke({"question": question}, config={"configurable": {"thread_id": tid}})
        )

    def _resume(self, tid: str, decision: str, note: str = "") -> dict:
        return asyncio.run(
            self.app.ainvoke(
                Command(resume={"decision": decision, "note": note}),
                config={"configurable": {"thread_id": tid}},
            )
        )

    def _interrupt(self, question: str, name: str) -> tuple[str, dict]:
        tid = _tid(name)
        with patch.object(graph_module.settings, "hitl_enabled", 1):
            result = self._ask(question, tid)
        self.assertIn("__interrupt__", result, "证据不足时应停在人工确认节点")
        return tid, result["__interrupt__"][0].value

    def test_interrupt_carries_actionable_review_payload(self) -> None:
        _, review = self._interrupt(INSUFFICIENT, "payload")
        self.assertEqual(review["reason"], "critic_insufficient")
        self.assertEqual([o["decision"] for o in review["options"]], ["approve", "refine", "abort"])
        # 确认单必须让人能判断"值不值得继续"，所以要有证据规模、来源和 Critic 原话
        self.assertGreaterEqual(review["evidence_count"], 1)
        self.assertTrue(review["sources"])
        self.assertIn("判定不足", review["critique"])
        self.assertTrue(review["top_titles"])

    def test_approve_generates_without_extra_retrieval(self) -> None:
        tid, _ = self._interrupt(INSUFFICIENT, "approve")
        with patch.object(graph_module.settings, "hitl_enabled", 1):
            out = self._resume(tid, "approve")
        self.assertEqual(out["review_decision"], "approve")
        self.assertFalse(out["grounded"], "证据没变，不能因为人工点了继续就变成充分")
        self.assertTrue(out["answer"])
        self.assertTrue(out["citations"], "已有证据仍应被引用")
        agents = [t["agent"] for t in out["traces"]]
        self.assertIn("human_review", agents)
        self.assertIn("generate", agents)
        self.assertEqual(agents.count("query_rewriter"), 1, "approve 不该再检索一轮")

    def test_refine_runs_one_more_retrieval_round(self) -> None:
        tid, _ = self._interrupt(INSUFFICIENT, "refine")
        with patch.object(graph_module.settings, "hitl_enabled", 1):
            out = self._resume(tid, "refine", note="只看血清蛋白定量")
        self.assertEqual(out["review_decision"], "refine")
        self.assertEqual(out["reflection_count"], 1, "人工要求再检索也要计入反思轮次")
        self.assertEqual(out["review_count"], 1)
        agents = [t["agent"] for t in out["traces"]]
        self.assertEqual(agents.count("query_rewriter"), 2, "refine 必须真的再检索一轮")
        self.assertNotIn("__interrupt__", out, "HITL_MAX_REVIEWS=1 时不应二次打断")
        # 补充说明要留在决策链里，便于事后解释"这一轮为什么这么查"
        review_line = next(t["detail"] for t in out["traces"] if t["agent"] == "human_review")
        self.assertIn("只看血清蛋白定量", review_line)

    def test_refine_writes_note_into_query_focus(self) -> None:
        """补充说明必须拼进 query_focus，否则重写器看不到它，refine 等于空转。"""
        from src.agents import nodes

        state = {"question": INSUFFICIENT, "critique": "证据不足", "evidences": []}
        with patch("src.agents.nodes.interrupt", return_value={"decision": "refine", "note": "排除方法学综述"}):
            update = asyncio.run(nodes.human_review(state))
        self.assertTrue(update["query_focus"].startswith("证据不足"))
        self.assertIn("人工补充：排除方法学综述", update["query_focus"])
        self.assertEqual(update["reflection_count"], 1)

    def test_review_node_does_not_resume_past_budget(self) -> None:
        """已到中断上限就直接放行：否则 resume 会再次触发 interrupt，请求永远回不来。"""
        from src.agents import nodes

        state = {"question": INSUFFICIENT, "review_count": 1, "evidences": []}
        with patch.object(graph_module.settings, "hitl_max_reviews", 1):
            update = asyncio.run(nodes.human_review(state))
        self.assertEqual(update, {"review_decision": "approve"})

    def test_abort_refuses_without_calling_generator(self) -> None:
        tid, _ = self._interrupt(INSUFFICIENT, "abort")
        with patch.object(graph_module.settings, "hitl_enabled", 1):
            out = self._resume(tid, "abort")
        self.assertEqual(out["review_decision"], "abort")
        self.assertEqual(out["citations"], [], "中止后不应再引用任何证据")
        self.assertIn("中止", out["answer"])
        self.assertEqual(out["traces"][-1]["agent"], "refuse")
        self.assertNotIn("generate", [t["agent"] for t in out["traces"]])

    def test_invalid_decision_falls_back_to_approve(self) -> None:
        tid, _ = self._interrupt(INSUFFICIENT, "invalid")
        with patch.object(graph_module.settings, "hitl_enabled", 1):
            out = self._resume(tid, "looks-good-to-me")
        self.assertEqual(out["review_decision"], "approve")
        self.assertTrue(out["answer"])

    def test_grounded_question_never_interrupts(self) -> None:
        with patch.object(graph_module.settings, "hitl_enabled", 1):
            result = self._ask(GROUNDED, _tid("grounded"))
        self.assertNotIn("__interrupt__", result)
        self.assertTrue(result["grounded"])

    def test_exhausted_retrieval_does_not_interrupt(self) -> None:
        """证据为 0 时再检索也是 0，此时弹确认只是噪音：应当直接拒答。"""
        with patch.object(graph_module.settings, "hitl_enabled", 1):
            result = self._ask(OUT_OF_CORPUS, _tid("empty"))
        self.assertNotIn("__interrupt__", result)
        self.assertFalse(result["grounded"])

    def test_disabled_hitl_keeps_legacy_behavior(self) -> None:
        with patch.object(graph_module.settings, "hitl_enabled", 0):
            result = self._ask(INSUFFICIENT, _tid("disabled"))
        self.assertNotIn("__interrupt__", result)
        self.assertTrue(result["answer"])
        self.assertEqual(result.get("review_count", 0), 0)

    def test_resume_without_pending_checkpoint_raises(self) -> None:
        """已完成的 thread 再调 resume 要立刻报错，而不是悄悄重跑一遍（重复计费）。"""
        tid = _tid("done")
        with patch.object(graph_module.settings, "hitl_enabled", 0):
            self._ask(GROUNDED, tid)
        with self.assertRaises(ValueError):
            asyncio.run(graph_module.resume(tid, "approve"))


if __name__ == "__main__":
    unittest.main()
