"""运行时依赖容器：懒加载单例。

为什么不在模块顶层直接实例化：
1. `import src.agents.nodes` 会立刻构建向量索引——CLI 冷启动慢，
   单纯想复用某个函数也要付全部代价；
2. 单元测试无法把检索器换成替身，只能跑真实索引；
3. 配置在 import 时被固化，测试改环境变量不生效。

改成首次调用时构建，既保留"全局单例"的复用优势，又消除导入期副作用。
"""
from __future__ import annotations

from functools import lru_cache

from .config import get_settings
from .llm import LLMClient
from .rag.reranker import CrossEncoderReranker
from .rag.retriever import HybridRetriever


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    return LLMClient()


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever.from_corpus()


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    # 与检索共用同一个 embedding：词表与 idf 一致，重排分数才有可比性
    return CrossEncoderReranker(top_k=get_settings().rerank_top_k, embedder=get_retriever().embedder)


@lru_cache(maxsize=1)
def get_answer_cache():
    from .cache import AnswerCache

    return AnswerCache(ttl_s=get_settings().answer_cache_ttl_s)


def reset_runtime() -> None:
    """清空单例，供测试在切换配置后重建依赖。"""
    get_llm.cache_clear()
    get_retriever.cache_clear()
    get_reranker.cache_clear()
    get_answer_cache.cache_clear()
