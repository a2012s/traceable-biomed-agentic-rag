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

from ..config import get_settings
from ..data.corpus import Document, build_index
from ..utils.text import content_tokens, tokenize
from .embedding import build_embedding
from .vector_store import build_store


@dataclass
class HybridRetriever:
    docs: list[Document]
    bm25: object = None
    store: object = None
    embedder: object = None
    rrf_k: int = 0          # 0 表示取 settings.rrf_k
    _id_to_idx: dict[str, int] | None = None

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
        retriever = cls(
            docs=docs, bm25=bm25, store=store, embedder=embedder,
            rrf_k=get_settings().rrf_k,
        )
        # doc_id -> 下标只随语料变化，放进构造期；每次查询重建是 O(N) 纯浪费
        retriever._id_to_idx = {d.doc_id: i for i, d in enumerate(docs)}
        return retriever

    def _bm25_search(self, query: str, top_k: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in ranked if scores[i] > 0][:top_k]

    def _vector_search(self, query: str, top_k: int) -> list[int]:
        by_id = self._id_to_idx or {d.doc_id: i for i, d in enumerate(self.docs)}
        vec = self.embedder.encode([query])[0]
        hits = self.store.search(vec, top_k)
        return [by_id[doc_id] for doc_id, _ in hits if doc_id in by_id]

    def _kg_search(self, query: str, top_k: int) -> list[int]:
        """概念扩展：查询词命中 KG 概念时，按重叠度降序召回邻接文档。

        两个细节决定这一路有没有用：
        1. 只用实词（content_tokens）。用 tokenize 会把 the/of/的 算进去，
           任何 KG 文档都能"命中"，这一路退化成全量返回。
        2. 必须排序。按语料顺序取前 k 等于"前 k 篇 KG 文档"，与查询无关；
           文档数超过 top_k 后，后面的永远召不到。
        """
        q = content_tokens(query)
        if not q:
            return []
        scored = []
        for i, doc in enumerate(self.docs):
            if doc.source != "kg":
                continue
            overlap = q & content_tokens(doc.title + " " + doc.text)
            if overlap:
                scored.append((len(overlap), i))
        scored.sort(key=lambda x: (-x[0], x[1]))  # 次序键保证确定性，避免并列时顺序漂移
        return [i for _, i in scored[:top_k]]

    def _rrf_scores(self, lanes: list[list[int]], sources: set[str] | None) -> dict[int, float]:
        """把多路召回结果按 RRF 合成 下标 -> 分数。

        先按域过滤，再编名次：被过滤掉的文档不应占用 rank。否则单域检索时
        其他域的文档仍占着 1..k，本域最相关的会掉到后面，RRF 贡献被摊薄，
        等于自废单域检索。
        """
        rrf: dict[int, float] = {}
        k = self.rrf_k or get_settings().rrf_k
        for ranked in lanes:
            filtered = [i for i in ranked if not sources or self.docs[i].source in sources]
            for rank, idx in enumerate(filtered, start=1):
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k + rank)
        return rrf

    def retrieve(self, queries: list[str], top_k: int = 50, sources: set[str] | None = None) -> list[Document]:
        """多查询并行召回 + RRF 融合。top_k 控制送入重排前的候选数。"""
        rrf: dict[int, float] = {}
        for query in queries:
            lanes = [
                self._bm25_search(query, top_k),
                self._vector_search(query, top_k),
                self._kg_search(query, top_k),
            ]
            for idx, score in self._rrf_scores(lanes, sources).items():
                rrf[idx] = rrf.get(idx, 0.0) + score
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
