"""检索底座测试：向量后端一致性 + RRF 融合 + 金标集有效性。"""
from __future__ import annotations

import unittest

from src.data.corpus import ALL_DOCUMENTS
from src.eval.golden_set import GOLDEN_SET
from src.rag.embedding import HashEmbedding, TFIDFEmbedding
from src.rag.reranker import CrossEncoderReranker
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import NumpyVectorStore


class TestEmbeddingBackends(unittest.TestCase):
    def test_deterministic_shape(self):
        texts = [d.title for d in ALL_DOCUMENTS]
        for emb in (HashEmbedding(), TFIDFEmbedding().fit(texts)):
            with self.subTest(backend=emb.name):
                vecs = emb.encode(texts)
                self.assertEqual(len(vecs), len(texts))
                self.assertTrue(all(len(v) == emb.dim for v in vecs))
                self.assertEqual(vecs, emb.encode(texts))  # 可复现

    def test_tfidf_semantic_overlap(self):
        """TF-IDF 应让同主题文档比无关文档更接近。"""
        emb = TFIDFEmbedding().fit([d.title + " " + d.text for d in ALL_DOCUMENTS])
        q = emb.encode(["multiple sclerosis diagnosis McDonald criteria"])[0]
        texts = [d.title + " " + d.text for d in ALL_DOCUMENTS]
        vecs = emb.encode(texts)
        import numpy as np
        sims = [float(np.dot(q, v)) for v in vecs]
        ms_idx = next(i for i, d in enumerate(ALL_DOCUMENTS) if d.doc_id == "PMID:38112009")
        egfr_idx = next(i for i, d in enumerate(ALL_DOCUMENTS) if d.doc_id == "PMID:36100815")
        self.assertGreater(sims[ms_idx], sims[egfr_idx])


class TestVectorStore(unittest.TestCase):
    def test_upsert_search(self):
        store = NumpyVectorStore(dim=3)
        store.upsert(["a", "b"], [[1, 0, 0], [0, 1, 0]], [{}, {}])
        self.assertEqual(store.count(), 2)
        hits = store.search([1, 0, 0], top_k=2)
        self.assertEqual(hits[0][0], "a")
        self.assertAlmostEqual(hits[0][1], 1.0, places=5)
        # 正交文档得 0 分，应被过滤 —— 验证零相关截断
        self.assertEqual(len(hits), 1)


class TestRetrievalAndGoldenSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = HybridRetriever.from_corpus()
        cls.reranker = CrossEncoderReranker(top_k=5)

    @classmethod
    def tearDownClass(cls):
        close = getattr(cls.retriever.store, "close", None)
        if close:
            close()

    def test_golden_ids_exist_in_corpus(self):
        ids = {d.doc_id for d in ALL_DOCUMENTS}
        for case in GOLDEN_SET:
            with self.subTest(question=case.question):
                self.assertTrue(case.relevant <= ids, f"标注了语料中不存在的文档: {case.relevant - ids}")

    def test_retrieve_returns_ranked_documents(self):
        docs = self.retriever.retrieve(["EGFR mutation lung cancer"], top_k=5)
        self.assertTrue(docs)
        self.assertLessEqual(len(docs), 5)

    def test_rerank_filters_irrelevant(self):
        docs = self.retriever.retrieve(["SGLT2 inhibitors diabetes kidney"], top_k=20)
        ranked = self.reranker.rerank("SGLT2 inhibitors diabetes kidney", docs)
        self.assertTrue(ranked)
        self.assertTrue(all(e["score"] > 0 for e in ranked))


if __name__ == "__main__":
    unittest.main()
