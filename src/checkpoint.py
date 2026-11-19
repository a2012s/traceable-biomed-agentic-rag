"""检查点后端：可持久化，可选依赖缺失时优雅降级。

为什么不用单一 MemorySaver：
- 生产重启后要能续跑同一 thread_id；
- Web 多 worker 时内存检查点不共享。
默认仍是 memory，保持离线 Demo 零依赖；设 checkpointer=sqlite 且安装
langgraph-checkpoint-sqlite 后切到磁盘持久化，调用方无需改动。
"""
from __future__ import annotations

import logging
from functools import lru_cache

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from .config import ROOT, get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_checkpointer() -> BaseCheckpointSaver:
    settings = get_settings()
    backend = settings.checkpointer.lower()
    if backend == "memory":
        return InMemorySaver()
    if backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError:
            logger.warning(
                "checkpointer=sqlite 但未安装 langgraph-checkpoint-sqlite，降级 memory；"
                "安装：uv pip install langgraph-checkpoint-sqlite"
            )
            return InMemorySaver()
        db_path = ROOT / settings.checkpoint_db
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：FastAPI 线程池与主线程会交替访问同一个连接
        import sqlite3

        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        return SqliteSaver(conn)
    logger.warning("未知 checkpointer=%s，降级 memory", backend)
    return InMemorySaver()


def reset_checkpointer() -> None:
    get_checkpointer.cache_clear()
