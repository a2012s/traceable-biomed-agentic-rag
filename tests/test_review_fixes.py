"""针对代码审查 P0/P1 缺陷的回归测试。

每条测试对应一个真实踩过的坑，命名直接说明"修的是什么"，
避免后人重构时把这几处改回旧行为。
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents import nodes
from src.cache import AnswerCache
from src.data.corpus import Document
from src.rag.embedding import HashEmbedding
from src.rag.reranker import CrossEncoderReranker
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import NumpyVectorStore

ROOT = Path(__file__).resolve().parent.parent


def doc(doc_id: str, source: str = "literature", text: str = "text") -> Document:
    return Document(doc_id=doc_id, source=source, title=doc_id, text=text, url="")


def evidence(doc_id: str, score: float, source: str = "literature") -> dict:
    return {
        "doc_id": doc_id, "source": source, "title": doc_id, "snippet": "s",
        "url": "", "study_type": "", "evidence_level": "1", "score": score,
    }


class FakeRetriever(HybridRetriever):
    """固定三路召回结果，只测 RRF 合成逻辑。"""

    def __init__(self, docs: list[Document], lanes: list[list[int]]):
        self.docs = docs
        self.rrf_k = 60
        self._id_to_idx = {d.doc_id: i for i, d in enumerate(docs)}
        self._lanes = lanes

    def _bm25_search(self, query: str, top_k: int) -> list[int]:
        return self._lanes[0]

    def _vector_search(self, query: str, top_k: int) -> list[int]:
        return self._lanes[1]

    def _kg_search(self, query: str, top_k: int) -> list[int]:
        return self._lanes[2]


class TestHashEmbeddingCrossProcess(unittest.TestCase):
    def test_hash_embedding_is_reproducible_across_processes(self):
        """P0-1：hash(tok) 受 PYTHONHASHSEED 随机化，跨进程会投影到不同维度。

        用两个独立子进程（默认随机 hash seed）比较向量摘要，
        旧实现必然不相等，crc32 实现必须相等。
        """
        code = (
            "import hashlib,numpy as np;"
            "from src.rag.embedding import HashEmbedding;"
            "v=HashEmbedding(dim=64).encode(['multiple sclerosis diagnosis'])[0];"
            "print(hashlib.sha256(np.array(v,dtype=np.float32).tobytes()).hexdigest())"
        )
        env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONHASHSEED": "random"}
        digests = []
        for _ in range(2):
            out = subprocess.run(
                [sys.executable, "-c", code], cwd=ROOT, env=env,
                capture_output=True, text=True, timeout=60, check=True,
            )
            digests.append(out.stdout.strip())
        self.assertEqual(digests[0], digests[1])

    def test_hash_embedding_same_process_stable(self):
        emb = HashEmbedding(dim=32)
        self.assertEqual(emb.encode(["a b c"]), emb.encode(["a b c"]))
        self.assertNotEqual(emb.encode(["a b c"]), emb.encode(["x y z"]))


class TestKGSRecall(unittest.TestCase):
    def test_stopword_only_query_returns_nothing(self):
        """P0-5：用 tokenize 时 the/of 必然命中，KG 退化为全量返回。"""
        docs = [doc("K1", "kg", "multiple sclerosis concept"), doc("K2", "kg", "diabetes concept")]
        r = types.SimpleNamespace(docs=docs)
        self.assertEqual(HybridRetriever._kg_search(r, "what is the of", 10), [])

    def test_kg_results_sorted_by_overlap(self):
        docs = [
            doc("K1", "kg", "diabetes only"),
            doc("K2", "kg", "diabetes kidney cardiovascular"),
            doc("K3", "kg", "diabetes kidney"),
        ]
        r = types.SimpleNamespace(docs=docs)
        hits = HybridRetriever._kg_search(r, "diabetes kidney cardiovascular", 2)
        self.assertEqual([docs[i].doc_id for i in hits], ["K2", "K3"])


class TestRRFRanking(unittest.TestCase):
    def test_filter_happens_before_rank(self):
        """P0-4：被域过滤掉的文档不应占用名次。

        3 篇文献排在前面时，唯一一篇临床文档若排第 4 位，旧实现得 1/(60+4)；
        先过滤后它应排第 1 位，得 1/(60+1)。
        """
        docs = [doc("L1"), doc("L2"), doc("L3"), doc("C1", "clinical")]
        r = FakeRetriever(docs, [[0, 1, 2, 3], [], []])
        scores = r._rrf_scores([[0, 1, 2, 3], [], []], {"clinical"})
        self.assertIn(3, scores)
        self.assertAlmostEqual(scores[3], 1.0 / 61)
        self.assertNotIn(0, scores)


class TestFuseMinKeep(unittest.TestCase):
    def test_outlier_best_does_not_collapse_to_one(self):
        """P0-3：最佳项是离群值时，纯相对截断只剩 1 条 → 必然触发拒答。"""
        reranker = CrossEncoderReranker(top_k=5)
        items = [evidence(f"E{i}", s) for i, s in enumerate([10.0, 3.0, 2.9, 2.8, 2.7])]
        fused = reranker.fuse(items, top_k=8, margin=0.5, min_keep=3)
        self.assertGreaterEqual(len(fused), 3)
        self.assertEqual(fused[0]["doc_id"], "E0")

    def test_min_keep_never_exceeds_available(self):
        reranker = CrossEncoderReranker(top_k=5)
        fused = reranker.fuse([evidence("A", 1.0)], top_k=8, margin=0.5, min_keep=3)
        self.assertEqual(len(fused), 1)

    def test_below_floor_still_truncated_when_enough_above(self):
        reranker = CrossEncoderReranker(top_k=5)
        items = [evidence(f"E{i}", s) for i, s in enumerate([5.0, 4.9, 4.8, 0.1])]
        fused = reranker.fuse(items, top_k=8, margin=0.5, min_keep=3)
        self.assertEqual([e["doc_id"] for e in fused], ["E0", "E1", "E2"])

    def test_single_source_fallback_pulls_in_another_source(self):
        """保底 2 条若同域，Critic 的「独立数据源 >=2」仍不成立 → 证据充足也拒答。"""
        reranker = CrossEncoderReranker(top_k=5)
        items = [
            evidence("L-outlier", 10.0, "literature"),
            evidence("L2", 3.0, "literature"),
            evidence("L3", 2.9, "literature"),
            evidence("C1", 2.7, "clinical"),
        ]
        fused = reranker.fuse(items, top_k=8, margin=0.5, min_keep=2)
        self.assertEqual(len(fused), 2)                      # 条数不变，只换末位
        self.assertEqual(fused[0]["doc_id"], "L-outlier")     # 最佳项不能被替换掉
        self.assertGreaterEqual(len({e["source"] for e in fused}), 2)

    def test_min_keep_three_alone_does_not_fix_single_source(self):
        """只把 fuse_min_keep 提到 3 解决不了同源问题：3 条可能全在一个数据域。

        锁住「必须显式跨源补偿」这个结论——删掉 fuse() 里的跨源逻辑这条就会失败。
        """
        reranker = CrossEncoderReranker(top_k=5)
        items = [evidence(f"L{i}", s, "literature") for i, s in enumerate([10.0, 3.0, 2.9, 2.8])]
        items.append(evidence("C1", 1.0, "clinical"))
        fused = reranker.fuse(items, top_k=8, margin=0.5, min_keep=3)
        self.assertGreaterEqual(len({e["source"] for e in fused}), 2)


class TestCacheVersion(unittest.TestCase):
    def test_content_change_invalidates_cache_key(self):
        """P0-7：ID 不变、内容变（勘误/指南改版）时缓存必须失效。"""
        cache = AnswerCache(ttl_s=60)
        before = [doc("PMID:1", text="original text")]
        after = [doc("PMID:1", text="corrected text")]
        with patch("src.data.corpus.build_index", return_value=before):
            key_before = cache.key("same question")
        with patch("src.data.corpus.build_index", return_value=after):
            key_after = cache.key("same question")
        self.assertNotEqual(key_before, key_after)

    def test_prompt_version_in_cache_key(self):
        cache = AnswerCache(ttl_s=60)
        docs = [doc("PMID:1")]
        with patch("src.data.corpus.build_index", return_value=docs):
            k1 = cache.key("q")
            with patch("src.agents.prompts.PROMPT_VERSION", "v9-for-test"):
                k2 = cache.key("q")
        self.assertNotEqual(k1, k2)


class TestNumpyUpsert(unittest.TestCase):
    def test_upsert_preserves_unmentioned_items(self):
        """P1-6：旧实现整体替换，增量写入会静默丢掉旧文档。"""
        store = NumpyVectorStore(dim=3)
        store.upsert(["a", "b"], [[1, 0, 0], [0, 1, 0]], [{}, {}])
        store.upsert(["c"], [[0, 0, 1]], [{}])
        self.assertEqual(store.count(), 3)
        self.assertEqual(store.search([1, 0, 0], top_k=1)[0][0], "a")

    def test_upsert_updates_existing_id(self):
        store = NumpyVectorStore(dim=3)
        store.upsert(["a"], [[1, 0, 0]], [{}])
        store.upsert(["a"], [[0, 1, 0]], [{}])
        self.assertEqual(store.count(), 1)
        self.assertEqual(store.search([0, 1, 0], top_k=1)[0][0], "a")


class TestCitationWhitelist(unittest.TestCase):
    def test_citation_outside_prompt_set_is_dropped(self):
        """P0-6：白名单必须等于实际送进 prompt 的集合，第 9 条猜中也不能通过。"""
        evidences = [evidence(f"E{i}", 1.0 - i * 0.01) for i in range(10)]
        state = {"question": "q", "evidences": evidences, "grounded": True}

        class FakeLLM:
            is_mock = False

            def json_invoke(self, *args, **kwargs):
                return {"answer": "a", "citations": ["E9", "E0"]}, types.SimpleNamespace(tokens=1)

        with patch.object(nodes, "get_llm", return_value=FakeLLM()), \
                patch.object(nodes, "get_settings") as gs:
            gs.return_value.evidence_top_k = 8
            result = asyncio.run(nodes.generate(state))
        self.assertIn("E0", result["citations"])
        self.assertNotIn("E9", result["citations"])


class TestEmbeddingBatch(unittest.TestCase):
    def test_api_embedding_batches_and_validates_dim(self):
        """P1-5：必须分批发送，且维度以真实返回为准。"""
        from src.rag.embedding import APIEmbedding

        calls: list[int] = []

        def fake_post(texts, attempt=0):
            calls.append(len(texts))
            return [[0.1, 0.2, 0.3] for _ in texts]

        emb = APIEmbedding(model="m", base_url="http://x", api_key="k", batch_size=2)
        emb._post = fake_post  # type: ignore[method-assign]
        out = emb.encode(["a", "b", "c", "d", "e"])
        self.assertEqual(calls, [2, 2, 1])
        self.assertEqual(len(out), 5)
        self.assertEqual(emb.dim, 3)


class TestToolUseAndMultiTurn(unittest.TestCase):
    def test_tool_catalog_from_registry(self):
        """工具清单必须从注册表生成：新增工具时提示词自动跟上，避免两处维护。"""
        from src.agents.nodes import _tool_catalog
        from src.tools.biomedical import TOOL_SPECS

        catalog = _tool_catalog()
        for spec in TOOL_SPECS:
            self.assertIn(spec.name, catalog)

    def test_tool_result_becomes_evidence(self):
        """工具调用必须真正进 candidates，否则只是 trace 里的装饰。"""
        from src.agents.nodes import _normalize_tool_hits

        self.assertEqual(
            [d["doc_id"] for d in _normalize_tool_hits([{"doc_id": "PMID:1"}])], ["PMID:1"]
        )
        self.assertEqual(
            [d["doc_id"] for d in _normalize_tool_hits({"kg_docs": [{"doc_id": "UMLS:1"}]})],
            ["UMLS:1"],
        )

    def test_invalid_tool_params_do_not_kill_chain(self):
        """幻觉参数被 Pydantic 拒绝，但不该炸掉整条链路。"""
        from src.tools.biomedical import FatalToolError, dispatch

        with self.assertRaises(FatalToolError):
            asyncio.run(dispatch("pubmed_search", {"keywords": [], "top_k": 999}))

    def test_coreference_resolution_uses_history(self):
        """「它」应被上一轮的实体替换，否则多轮检索会丢掉上下文。"""
        from src.llm import _coreference_entities, _mock_rewrite

        history = "Q: EGFR 突变非小细胞肺癌的一线治疗?  A: 奥希替尼\n"
        entities = _coreference_entities(history)
        self.assertTrue(entities)
        self.assertTrue(any("lung" in e or "egfr" in e for e in entities))

        # 端到端验证 mock rewrite 真的做了替换，且历史块没有泄漏进检索查询
        rewritten = _mock_rewrite(
            "它的一线治疗?\nConversation history (for coreference resolution):\n" + history
        )
        first = rewritten["rewritten_queries"][0]
        self.assertNotIn("它", first)
        self.assertNotIn("Conversation history", first)

    def test_bootstrap_ci_is_deterministic_and_contains_point_estimate(self):
        """置信区间必须可复现；固定 seed 保证同输入同输出。"""
        from src.eval.metrics import bootstrap_ci

        values = [1.0, 0.8, 0.6, 0.4]
        self.assertEqual(bootstrap_ci(values), bootstrap_ci(values))
        lo, hi = bootstrap_ci(values)
        self.assertLessEqual(lo, hi)
        self.assertLessEqual(lo, 1.0)

    def test_golden_set_has_dev_test_split(self):
        """holdout 必须存在且固定，否则 test 会被调参污染。"""
        from src.eval.golden_set import GOLDEN_SET

        splits = {c.split for c in GOLDEN_SET}
        self.assertEqual(splits, {"dev", "test"})
        self.assertTrue(any(c.split == "test" for c in GOLDEN_SET))

    def test_cache_key_differs_with_history(self):
        """同一句「它」在不同上下文里必须走不同的缓存键。"""
        from src.cache import AnswerCache

        with patch("src.data.corpus.build_index", return_value=[doc("PMID:1")]):
            cache = AnswerCache(ttl_s=60)
            k1 = cache.key("它有什么副作用?")
            k2 = cache.key("它有什么副作用?||history:1:MS 的诊断标准")
        self.assertNotEqual(k1, k2)


if __name__ == "__main__":
    unittest.main()
