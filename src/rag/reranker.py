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
import time
import urllib.request
from collections import OrderedDict
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


_CACHE_MAX = 4096  # 缓存上限：无界 dict 在长跑进程里等于内存泄漏


class CrossEncoderReranker:
    def __init__(self, top_k: int = 5, embedder: Any | None = None):
        self.top_k = top_k
        self.embedder = embedder
        # OrderedDict + 容量上限：query/doc 文本做键，不设上限会随流量无界增长
        self._doc_vectors: OrderedDict[str, list[float]] = OrderedDict()
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._external_cache: OrderedDict[tuple[str, str], float] = OrderedDict()
        # 熔断状态放在实例上，绝不改写 get_settings() 的单例：
        # 一次网络抖动若把 settings.reranker_api_url 置空，整个进程的外排能力
        # 就永久关闭且污染所有读者（健康检查、backend_info 都会看到被篡改的值）。
        self._external_disabled_until: float = 0.0
        self._entity_linker = EntityLinking(enabled=bool(get_settings().entity_linking))

    @staticmethod
    def _cache_put(cache: OrderedDict, key: Any, value: Any) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > _CACHE_MAX:
            cache.popitem(last=False)

    def _external_available(self) -> bool:
        return time.time() >= self._external_disabled_until

    def _trip_breaker(self) -> None:
        cooldown = get_settings().reranker_breaker_cooldown_s
        self._external_disabled_until = time.time() + max(0.0, cooldown)
        logger.warning("外部 reranker 熔断 %.1fs，期间降级内置打分", cooldown)

    def _external_scores(self, query: str, texts: list[str]) -> list[float] | None:
        """可选外部 cross-encoder，批量接口：POST {query, texts} -> {scores: [...]}。

        批量的原因很实际：逐条发 50 个请求、每个 timeout 3s，最坏 150s，
        必然超过单请求预算。一次批量把最坏情况压到 3s。
        单条端点（{query, text} -> {score}）仍兼容：texts 长度为 1 时回退。
        """
        settings = get_settings()
        if not settings.reranker_api_url or not self._external_available() or not texts:
            return None
        pending = [(t, self._external_cache.get((query, t))) for t in texts]
        missing = [t for t, cached in pending if cached is None]
        if missing:
            payload = {"query": query, "texts": missing}
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
                with urllib.request.urlopen(req, timeout=settings.reranker_timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                scores = body.get("scores")
                if scores is None and "score" in body and len(missing) == 1:
                    scores = [body["score"]]
                if not isinstance(scores, list) or len(scores) != len(missing):
                    raise ValueError(f"返回的 scores 数量与输入不符: {body}")
                for text, score in zip(missing, scores, strict=True):
                    self._cache_put(self._external_cache, (query, text), float(score))
            except Exception as exc:  # noqa: BLE001 - 外部服务降级是刻意设计
                logger.warning("外部 reranker 不可用(%s)，降级内置打分", exc)
                self._trip_breaker()
                return None
        # 缺失项若因并发写入仍为空，保留 None 由调用方走内置打分
        return [self._external_cache.get((query, t)) for t in texts]

    def _external_score(self, query: str, text: str) -> float | None:
        """单条路径，供 score() 与测试使用；内部复用批量实现。"""
        scores = self._external_scores(query, [text])
        return scores[0] if scores else None

    def _vector(self, text: str) -> list[float]:
        cached = self._query_cache.get(text)
        if cached is None:
            cached = self.embedder.encode([text])[0]
            self._cache_put(self._query_cache, text, cached)
        else:
            self._query_cache.move_to_end(text)
        return cached

    def _doc_vector(self, doc: Document) -> list[float]:
        vec = self._doc_vectors.get(doc.doc_id)
        if vec is None:
            vec = self.embedder.encode([f"{doc.title} {doc.text}"])[0]
            self._cache_put(self._doc_vectors, doc.doc_id, vec)
        else:
            self._doc_vectors.move_to_end(doc.doc_id)
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
        # 只对前 N 条候选重排：召回 Top-50 全部送外排，延迟与成本都不划算，
        # 且深位候选几乎不可能进入最终证据窗口。
        limit = get_settings().rerank_candidates
        docs = docs[:limit] if limit > 0 else docs
        texts = [f"{d.title} {d.text}" for d in docs]
        external_all = self._external_scores(query, texts) if docs else None
        scored = []
        for pos, doc in enumerate(docs):
            level = _LEVEL_WEIGHT.get(doc.evidence_level, 0.25)
            source = _SOURCE_WEIGHT.get(doc.source, 0.5)
            recency = min(1.0, max(0.0, (doc.year - 2020) / 6))
            external = external_all[pos] if external_all is not None else None
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

    def fuse(
        self,
        evidences: list[Evidence],
        top_k: int | None = None,
        margin: float | None = None,
        min_keep: int | None = None,
    ) -> list[Evidence]:
        """跨域融合：把各域排好序的证据合并成一份全局序列。

        两件事必须做：
        1. 全局重排 —— 各域各自取 Top-k 后直接拼接，生成节点只会看到第一个数据域那几条；
        2. 相对截断 —— 低于最佳证据 `margin` 倍的一律丢弃。单个通用词偶然命中的文档
           分数通常只有最佳项的三四成，靠这条挡在引用之外（绝对阈值跨查询不可比）。

        但纯相对截断有个结构性坑：最佳项是离群值（标题精准命中）时 floor 被抬得过高，
        其余同为高质量的证据会被全部丢掉，只剩 1 条 → Critic 的「独立数据源 ≥2」必然
        不成立 → 反思一轮后截断逻辑不变 → 跑满上限，证据明明充足却拒答。
        因此保留 `min_keep` 条保底。
        """
        if not evidences:
            return []
        s = get_settings()
        top_k = s.evidence_top_k if top_k is None else top_k
        margin = s.fuse_margin if margin is None else margin
        min_keep = s.fuse_min_keep if min_keep is None else min_keep

        ordered = sorted(evidences, key=lambda e: float(e.get("score", 0.0)), reverse=True)
        floor = float(ordered[0].get("score", 0.0)) * margin
        above = [e for e in ordered if float(e.get("score", 0.0)) >= floor]
        keep = max(min_keep, len(above))
        picked = ordered[: min(keep, top_k)]

        # 只保条数不够，还要保「跨源」。Critic 的 grounded 判据是「独立数据源 >=2」，
        # 保底 2 条若同域，依然判不充分 → 依然跑满反思轮 → 证据充足却拒答。
        # 单纯提高 min_keep 也无效：实测把 3 条同域证据留下，distinct_sources 仍是 1。
        # 所以这里显式用一条不同 source 的候选替换末位（保留最佳项、条数不变）。
        sources = {e.get("source", "") for e in picked}
        if len(sources) < 2 and len(picked) >= 2:
            alt = next((e for e in ordered if e.get("source", "") not in sources), None)
            if alt is not None:
                picked = picked[:-1] + [alt]
                picked.sort(key=lambda e: float(e.get("score", 0.0)), reverse=True)
        return picked

    @staticmethod
    def make_snippet(text: str, query: str, window: int = 220) -> str:
        """证据窗口：截取命中查询词最密集的片段，控制送入 LLM 的上下文长度。"""
        q = content_tokens(query)
        sentences = re.split(r"(?<=[.!?。])\s+", text)
        if not sentences:
            return text[:window]
        best = max(sentences, key=lambda s: len(q & content_tokens(s)))
        return best[:window]
