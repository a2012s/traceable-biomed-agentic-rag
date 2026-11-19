"""重排 + 跨域融合 + 证据窗口。

真实项目用 bge-reranker-v2-m3 这类交叉编码器；本 demo 用**可解释的打分**模拟，
保留"重排"这一步的工程位置与接口：`score()` 是可替换点，换成真模型不改调用方。

打分构成（都在 0~1 之间，可解释）：
- 相关度  IDF 加权余弦（与检索共用同一 embedding，天然压掉 "receptor"、"mass" 这类高频词）
- 证据等级  系统评价/RCT > 队列 > 病例 > 专家意见/图谱
- 来源可信度 文献 > 试验 > 报告 > 图谱
- 时效性    近三年加权

为什么不用朴素词重叠：医学词汇高度复用（egfr 的 "receptor" 会撞 GLP-1 的 "receptor"），
单个通用词命中就能把跨领域文档拉进引用。IDF 加权后这类词的贡献天然变小。
"""
from __future__ import annotations

import json
import logging
import math
import re
import urllib.request
from typing import Any

from ..config import get_settings
from ..data.corpus import Document
from ..state import Evidence
from ..utils.text import content_tokens
from .entity_linking import EntityLinking

logger = logging.getLogger(__name__)

_LEVEL_WEIGHT = {"1": 1.0, "2": 0.7, "3": 0.45, "4": 0.25}
_SOURCE_WEIGHT = {"literature": 1.0, "clinical": 0.95, "report": 0.7, "kg": 0.55}

# 相关度权重压倒先验：先验只用于同等相关度时打破平手，不能把无关文档顶上来
_W_RELEVANCE, _W_LEVEL, _W_SOURCE, _W_RECENCY = 0.78, 0.12, 0.06, 0.04


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class CrossEncoderReranker:
    def __init__(self, top_k: int = 5, embedder: Any | None = None):
        self.top_k = top_k
        self.embedder = embedder
        self._doc_vectors: dict[str, list[float]] = {}
        self._query_cache: dict[str, list[float]] = {}
        self._external_cache: dict[tuple[str, str], float] = {}
        self._entity_linker = EntityLinking(enabled=bool(get_settings().entity_linking))

    def _external_score(self, query: str, text: str) -> float | None:
        """可选外部 cross-encoder：POST {url} {model?, query, text} -> {score}。

        只取 0~1 的相关度，先验与融合逻辑仍由本类处理 —— 换模型不动调用方。
        失败一律降级内置打分并告警，不让重排服务故障拖垮整条链路。
        """
        settings = get_settings()
        if not settings.reranker_api_url:
            return None
        key = (query, text)
        if key in self._external_cache:
            return self._external_cache[key]
        payload = {"query": query, "text": text}
        headers = {"Content-Type": "application/json"}
        if settings.reranker_api_key:
            headers["Authorization"] = f"Bearer {settings.reranker_api_key}"
        try:
            req = urllib.request.Request(
                settings.reranker_api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                score = float(json.loads(resp.read().decode("utf-8"))["score"])
            self._external_cache[key] = score
            return score
        except Exception as exc:  # noqa: BLE001 - 外部服务降级是刻意设计
            logger.warning("外部 reranker 不可用(%s)，降级内置打分", exc)
            settings.reranker_api_url = ""  # 熔断：本次进程内不再重试
            return None

    def _vector(self, text: str) -> list[float]:
        cached = self._query_cache.get(text)
        if cached is None:
            cached = self.embedder.encode([text])[0]
            self._query_cache[text] = cached
        return cached

    def _doc_vector(self, doc: Document) -> list[float]:
        vec = self._doc_vectors.get(doc.doc_id)
        if vec is None:
            vec = self.embedder.encode([f"{doc.title} {doc.text}"])[0]
            self._doc_vectors[doc.doc_id] = vec
        return vec

    def score(self, query: str, doc: Document) -> float:
        level = _LEVEL_WEIGHT.get(doc.evidence_level, 0.25)
        source = _SOURCE_WEIGHT.get(doc.source, 0.5)
        recency = min(1.0, max(0.0, (doc.year - 2020) / 6))

        if self.embedder is None:
            # 无 embedding 时的降级路径：只按先验排序（回调用例/测试）
            return 0.5 * level + 0.3 * source + 0.2 * recency

        external = self._external_score(query, f"{doc.title} {doc.text}")
        if external is not None:
            relevance = external
        else:
            relevance = _cosine(self._vector(query), self._doc_vector(doc))
        if relevance <= 0:
            return 0.0
        base = _W_RELEVANCE * relevance + _W_LEVEL * level + _W_SOURCE * source + _W_RECENCY * recency
        # 实体链接降权：同形异义的另一义项证据不该被当成强相关
        senses = self._entity_linker.detect_senses(query)
        penalty = self._entity_linker.penalty(f"{doc.title} {doc.text}", senses)
        return base * penalty

    def rerank(self, query: str, docs: list[Document]) -> list[Evidence]:
        # 语境判定每次 rerank 算一次即可，避免逐文档重复扫描查询词
        senses = self._entity_linker.detect_senses(query)
        scored = []
        for doc in docs:
            level = _LEVEL_WEIGHT.get(doc.evidence_level, 0.25)
            source = _SOURCE_WEIGHT.get(doc.source, 0.5)
            recency = min(1.0, max(0.0, (doc.year - 2020) / 6))
            external = self._external_score(query, f"{doc.title} {doc.text}")
            if external is not None:
                relevance = external
            elif self.embedder is not None:
                relevance = _cosine(self._vector(query), self._doc_vector(doc))
            else:
                relevance = 0.0
            if relevance <= 0 and self.embedder is not None:
                continue
            if self.embedder is None:
                score = 0.5 * level + 0.3 * source + 0.2 * recency
            else:
                score = _W_RELEVANCE * relevance + _W_LEVEL * level + _W_SOURCE * source + _W_RECENCY * recency
            score *= self._entity_linker.penalty(f"{doc.title} {doc.text}", senses)
            scored.append((score, doc))
        scored = [(s, d) for s, d in scored if s > 0]  # 零相关直接丢弃
        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[Evidence] = []
        for score, doc in scored[: self.top_k]:
            out.append(
                Evidence(
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    snippet=self.make_snippet(doc.text, query),
                    url=doc.url,
                    study_type=doc.study_type,
                    evidence_level=doc.evidence_level,
                    score=round(score, 4),
                )
            )
        return out

    def fuse(self, evidences: list[Evidence], top_k: int = 8, margin: float = 0.5) -> list[Evidence]:
        """跨域融合：把各域排好序的证据合并成一份全局序列。

        两件事必须做：
        1. 全局重排 —— 各域各自取 Top-k 后直接拼接，生成节点只会看到第一个数据域那几条；
        2. 相对截断 —— 低于最佳证据 `margin` 倍的一律丢弃。单个通用词偶然命中的文档
           分数通常只有最佳项的三四成，靠这条挡在引用之外（绝对阈值跨查询不可比）。
        """
        if not evidences:
            return []
        ordered = sorted(evidences, key=lambda e: float(e.get("score", 0.0)), reverse=True)
        floor = float(ordered[0].get("score", 0.0)) * margin
        return [e for e in ordered if float(e.get("score", 0.0)) >= floor][:top_k]

    @staticmethod
    def make_snippet(text: str, query: str, window: int = 220) -> str:
        """证据窗口：截取命中查询词最密集的片段，控制送入 LLM 的上下文长度。"""
        q = content_tokens(query)
        sentences = re.split(r"(?<=[.!?。])\s+", text)
        if not sentences:
            return text[:window]
        best = max(sentences, key=lambda s: len(q & content_tokens(s)))
        return best[:window]
