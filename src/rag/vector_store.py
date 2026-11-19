"""向量库：Qdrant 本地模式（真实持久化 + HNSW）+ numpy 暴力检索兜底。

两者接口一致（upsert / search）。生产把 QdrantLocal 换成 QdrantClient(url=...)
即切远端集群，检索代码不变 —— 这是 JD 里"熟悉向量数据库"的落地形态。
"""
from __future__ import annotations

import atexit
import contextlib
import logging
import uuid
from pathlib import Path

import numpy as np

from ..config import ROOT, get_settings

logger = logging.getLogger(__name__)


class VectorStore:
    dim: int
    name: str

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        raise NotImplementedError

    def search(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


class NumpyVectorStore(VectorStore):
    """零依赖兜底：内存矩阵 + 余弦相似度。语料量大时 O(N) 是瓶颈，但接口正确。"""

    def __init__(self, dim: int):
        self.dim = dim
        self.name = "numpy"
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        self._ids = list(ids)
        self._matrix = np.asarray(vectors, dtype=np.float32)
        # 余弦：预归一化，检索时点积即可
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        self._matrix = self._matrix / np.where(norms == 0, 1.0, norms)

    def search(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        if self._matrix is None:
            return []
        q = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(q)) or 1.0
        scores = self._matrix @ (q / norm)
        order = np.argsort(-scores)[:top_k]
        return [(self._ids[i], float(scores[i])) for i in order if scores[i] > 0]

    def count(self) -> int:
        return len(self._ids)


class QdrantLocal(VectorStore):
    """Qdrant 嵌入式模式：HNSW 索引 + 磁盘持久化，无需起服务。"""

    def __init__(self, collection: str = "biomedical", dim: int = 0, path: str | None = None):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        s = get_settings()
        location = path or s.qdrant_path
        if location and location != ":memory:":
            p = Path(location)
            if not p.is_absolute():
                p = ROOT / p
            p.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(p))
        else:
            self.client = QdrantClient(location=":memory:")
        self.collection = collection
        self.name = "qdrant-local"
        self.dim = dim
        # 本地模式底层是 sqlite 文件句柄，不显式关闭会在解释器退出时报
        # ResourceWarning / sys.meta_path is None，污染 CLI 与 CI 输出
        atexit.register(self.close)
        if dim and self.client.collection_exists(collection):
            existing = self.client.get_collection(collection_name=collection)
            existing_size = existing.config.params.vectors.size
            if existing_size != dim:
                # embedding 后端切换导致维度变化：重建集合，避免检索时静默降级
                self.client.delete_collection(collection_name=collection)
        if dim and not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        from qdrant_client.models import PointStruct

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={"": {"size": len(vectors[0]), "distance": "Cosine"}},
            )
            self.dim = len(vectors[0])
        # Qdrant point id 只接受 UUID/整数；医学 doc_id(PMID:xxx) 映射到确定性 UUID，
        # 原始 doc_id 存入 payload，检索时回读 —— 避免外部 ID 格式约束侵入检索层。
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"doc::{i}")),
                vector=v,
                payload={**p, "_doc_id": i},
            )
            for i, v, p in zip(ids, vectors, payloads, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        hits = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        ).points
        return [
            (str(hit.payload.get("_doc_id", hit.id)), float(hit.score))
            for hit in hits
            if hit.score > 0
        ]

    def count(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count

    def close(self) -> None:
        """显式释放本地锁文件，避免解释器退出时的 ResourceWarning。"""
        with contextlib.suppress(Exception):  # 关闭失败不影响业务
            self.client.close()


def build_store(dim: int, collection: str = "biomedical") -> VectorStore:
    """按配置选向量库；Qdrant 未安装或出错时降级 numpy，保证开箱即跑。"""
    s = get_settings()
    if s.vector_store.lower() == "qdrant":
        try:
            return QdrantLocal(collection=collection, dim=dim)
        except ImportError as exc:
            logger.warning("Qdrant 未安装(%s)，降级 numpy", exc)
        except Exception as exc:  # noqa: BLE001 - 锁冲突等运行时问题用内存模式兜底
            # 本地模式用文件锁，残留锁文件会导致冲突；改用内存 Qdrant 仍是真 HNSW 索引
            logger.warning("Qdrant 本地锁冲突(%s)，改用内存 Qdrant", exc)
            try:
                return QdrantLocal(collection=collection, dim=dim, path=":memory:")
            except Exception:  # noqa: BLE001
                logger.warning("Qdrant 内存模式也失败，降级 numpy")
    return NumpyVectorStore(dim=dim)
