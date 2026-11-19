"""答案缓存：相同问题短 TTL 命中，避免重复烧 token。

键必须包含影响答案的所有变量：规范化问题 + 语料版本 + 检索/融合配置 + 模型档位。
只按问题字符串做键是常见错误 —— 换模型或重建索引后会返回旧答案。
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from threading import Lock

from .config import get_settings


def _corpus_version() -> str:
    """语料版本标识：文档数 + 每条文档的 ID 与正文摘要。

    必须把正文纳入摘要。真实场景最常见的更新恰恰是「ID 不变、内容变」——
    同一 PMID 的勘误、指南年度改版沿用同一编号、同一 NCT 号的方案修订。
    只对 doc_id 做摘要时，缓存会继续命中旧答案，最长整整一个 TTL。
    """
    from .data.corpus import build_index

    docs = build_index()
    h = hashlib.sha256()
    for d in docs:
        h.update(d.doc_id.encode())
        h.update(b"\x00")
        h.update((d.title + "\n" + d.text).encode())
        h.update(b"\x01")
    return f"{len(docs)}-{h.hexdigest()[:12]}"


def normalize_question(question: str) -> str:
    text = question.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


@dataclass
class AnswerCache:
    ttl_s: int = 300
    _entries: dict[str, tuple[float, dict]] = field(default_factory=dict, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def key(self, question: str) -> str:
        from .agents.prompts import PROMPT_VERSION

        s = get_settings()
        parts = [
            normalize_question(question),
            _corpus_version(),
            f"prompt-{PROMPT_VERSION}",
            str(s.embedding_backend),
            str(s.retrieve_top_k),
            str(s.rerank_candidates),
            str(s.evidence_top_k),
            str(s.rerank_top_k),
            str(s.fuse_margin),
            str(s.fuse_min_keep),
            str(s.max_reflections),
            str(s.wide_recall),
            str(s.entity_linking),
            s.active_model if not s.is_mock else "mock",
        ]
        return hashlib.sha256("||".join(parts).encode()).hexdigest()

    def get(self, question: str) -> dict | None:
        if self.ttl_s <= 0:
            return None
        key = self.key(question)
        with self._lock:
            item = self._entries.get(key)
            if item is None:
                return None
            created, payload = item
            if time.time() - created > self.ttl_s:
                self._entries.pop(key, None)
                return None
            return dict(payload)

    def put(self, question: str, payload: dict) -> None:
        if self.ttl_s <= 0:
            return
        with self._lock:
            # 简易容量上限：超出后清掉最早插入的一半，避免无界增长
            if len(self._entries) >= 256:
                for k in list(self._entries)[:128]:
                    self._entries.pop(k, None)
            self._entries[self.key(question)] = (time.time(), dict(payload))

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
