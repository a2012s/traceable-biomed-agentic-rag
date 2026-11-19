"""可插拔 Embedding 后端。

三种实现，接口完全一致，生产切换只改配置不改调用方：
- hash  : 字符 n-gram 哈希投影，零依赖离线兜底（demo 默认，语义最弱）
- tfidf : 真 TF-IDF（numpy），离线可用、比哈希强，适合无 API 的冷启动
- api   : OpenAI 兼容 /embeddings 端点（百炼 text-embedding-v3 / OpenAI / DeepSeek）

设计说明：hash 不是"假向量"的遮羞布，而是明确标注的离线降级；
需要语义能力时把 EMBEDDING_BACKEND=api 即切到领域 embedding 模型。
"""
from __future__ import annotations

import json
import logging
import math
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..config import get_settings
from ..utils.text import tokenize

logger = logging.getLogger(__name__)


class EmbeddingBackend:
    dim: int
    name: str

    def encode(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


@dataclass
class HashEmbedding(EmbeddingBackend):
    """字符哈希投影：无依赖、可复现，作为离线最低档。"""

    dim: int = 256
    name: str = "hash"

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for tok in tokenize(text):
                # 用 crc32 而不是内置 hash()：str 的 hash 受 PYTHONHASHSEED 随机化，
                # 同一文本在不同进程会被投影到不同维度。索引一旦落盘，跨进程检索
                # 就会静默退化为随机结果——这是最难排查的一类 bug。
                vec[zlib.crc32(tok.encode("utf-8")) % self.dim] += 1.0
            norm = float(np.linalg.norm(vec)) or 1.0
            out.append((vec / norm).tolist())
        return out


@dataclass
class TFIDFEmbedding(EmbeddingBackend):
    """真实 TF-IDF：在语料上拟合 idf，查询与文档共用同一词表。

    离线可用且语义远强于哈希投影，是"无 API key 时也能讲清向量检索"的最佳档位。
    """

    name: str = "tfidf"
    dim: int = 0
    _vocab: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _idf: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, docs: list[str]) -> "TFIDFEmbedding":
        n = len(docs)
        df: Counter = Counter()
        for text in docs:
            df.update(set(tokenize(text)))
        self._vocab = {tok: i for i, tok in enumerate(sorted(df))}
        self.dim = len(self._vocab)
        self._idf = np.array(
            [math.log((n + 1) / (df[tok] + 1)) + 1 for tok in self._vocab],
            dtype=np.float32,
        )
        return self

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._idf is None:
            raise RuntimeError("TFIDFEmbedding.fit() must be called before encode()")
        out = []
        for text in texts:
            counts = Counter(tokenize(text))
            vec = np.zeros(self.dim, dtype=np.float32)
            for tok, count in counts.items():
                idx = self._vocab.get(tok)
                if idx is not None:
                    vec[idx] = (1.0 + math.log(count)) * self._idf[idx]
            norm = float(np.linalg.norm(vec)) or 1.0
            out.append((vec / norm).tolist())
        return out


@dataclass
class APIEmbedding(EmbeddingBackend):
    """OpenAI 兼容 /embeddings：百炼 text-embedding-v3 等。

    走 urllib 而非厂商 SDK，与其他模块保持一致，避免绑定单一供应商。
    """

    name: str = "api"
    dim: int = 1024
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    batch_size: int = 64
    timeout_s: float = 45.0
    _dim_observed: bool = False

    def _post(self, texts: list[str], attempt: int = 0) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")[:200]
            # 429/5xx 是瞬时过载，退避重试；4xx（除 429）是参数或授权错误，重试无意义
            if (exc.code == 429 or exc.code >= 500) and attempt < 2:
                time.sleep(0.5 * (2**attempt))
                return self._post(texts, attempt + 1)
            raise RuntimeError(f"embedding API HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))
                return self._post(texts, attempt + 1)
            raise RuntimeError(f"embedding API 不可达: {exc}") from exc
        return [item["embedding"] for item in sorted(body["data"], key=lambda x: x["index"])]

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        size = max(1, self.batch_size)
        for start in range(0, len(texts), size):
            out.extend(self._post(texts[start : start + size]))
        dim = len(out[0])
        if not self._dim_observed:
            # 首次以真实返回为准：模型维度与默认值不同不算错，否则换模型必炸
            self.dim = dim
            self._dim_observed = True
        elif dim != self.dim:
            # 同一进程内维度漂移说明后端被切换或代理返回了异构结果，必须立即失败
            raise ValueError(f"embedding 维度不一致：已观测 {self.dim}，本次 {dim}")
        return out


def build_embedding(corpus_texts: list[str]) -> EmbeddingBackend:
    """按配置构造后端；api 失败时默认降级 tfidf，严格模式下直接失败。"""
    s = get_settings()
    backend = s.embedding_backend.lower()
    if backend == "hash":
        return HashEmbedding()
    if backend == "api" and s.active_api_key:
        try:
            emb = APIEmbedding(
                model=s.embedding_model,
                base_url=s.active_base_url,
                api_key=s.active_api_key,
                batch_size=s.embedding_batch_size,
            )
            emb.encode(["connectivity probe"])  # 提前探测并确定真实维度
            return emb
        except Exception as exc:  # noqa: BLE001 - 降级是刻意设计
            # 生产开 EMBEDDING_STRICT 时不要静默换后端：检索质量劣化不会体现在
            # 任何指标上，只会表现为"最近回答变差了"。
            if s.embedding_strict:
                raise RuntimeError(f"embedding API 不可用且已开启严格模式: {exc}") from exc
            logger.warning("embedding API 不可用(%s)，自动降级 tfidf", exc)
    return TFIDFEmbedding().fit(corpus_texts)
