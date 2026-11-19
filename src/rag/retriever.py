"""混合检索底座：BM25 + 向量库 + 知识图谱三路召回，RRF 融合。

三路互补：
- BM25 保基因名/药名/编号等精确匹配
- Vector 保语义相似（embedding 后端可插拔：hash/tfidf/api）
- KG   保概念上下位扩展（UMLS 风格）

RRF 用倒数排名融合，无需归一化不同检索器的分数，是工业界标准做法。
"""
from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from ..data.corpus import Document, build_index
from ..utils.text import tokenize
from .embedding import build_embedding
from .vector_store import build_store


@dataclass
class HybridRetriever:
    docs: list[Document]
    bm25: object = None
    store: object = None
    embedder: object = None
    rrf_k: int = 60

    @classmethod
    def from_corpus(cls) -> "HybridRetriever":
        docs = build_index()
        bm25 = BM25Okapi([tokenize(d.title + " " + d.text) for d in docs])

        # 真实向量库接入：embedding 与 store 后端均由配置决定
        embedder = build_embedding([d.title + " " + d.text for d in docs])
        vectors = embedder.encode([d.title + " " + d.text for d in docs])
        store = build_store(dim=embedder.dim)
        store.upsert(
            ids=[d.doc_id for d in docs],
            vectors=vectors,
            payloads=[d.as_dict() for d in docs],
        )
        return cls(docs=docs, bm25=bm25, store=store, embedder=embedder)

    def _bm25_search(self, query: str, top_k: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in ranked if scores[i] > 0][:top_k]

    def _vector_search(self, query: str, top_k: int) -> list[int]:
        vec = self.embedder.encode([query])[0]
        hits = self.store.search(vec, top_k)
        by_id = {d.doc_id: i for i, d in enumerate(self.docs)}
        return [by_id[doc_id] for doc_id, _ in hits if doc_id in by_id]

    def _kg_search(self, query: str, top_k: int) -> list[int]:
        """概念扩展：查询词命中 KG 概念时，把该概念的邻接文档一起召回。"""
        hits = []
        for i, doc in enumerate(self.docs):
            if doc.source != "kg":
                continue
            overlap = set(tokenize(query)) & set(tokenize(doc.title + " " + doc.text))
            if overlap:
                hits.append(i)
        return hits[:top_k]

    def retrieve(self, queries: list[str], top_k: int = 50, sources: set[str] | None = None) -> list[Document]:
        """多查询并行召回 + RRF 融合。top_k 控制送入重排前的候选数。"""
        rrf: dict[int, float] = {}
        for query in queries:
            lanes = [
                self._bm25_search(query, top_k),
                self._vector_search(query, top_k),
                self._kg_search(query, top_k),
            ]
            for ranked in lanes:
                for rank, idx in enumerate(ranked, start=1):
                    doc = self.docs[idx]
                    if sources and doc.source not in sources:
                        continue
                    rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
        ordered = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [self.docs[i] for i, _ in ordered]

    def backend_info(self) -> dict:
        """暴露当前 embedding / 向量库后端，评估与 CLI 展示用。"""
        return {
            "embedding": self.embedder.name,
            "embedding_dim": self.embedder.dim,
            "vector_store": self.store.name,
            "indexed": self.store.count(),
        }
