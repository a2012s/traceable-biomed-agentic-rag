"""反馈闭环：把「这条回答有没有用」落到本地 SQLite。

为什么不用 JSON 文件：并发写入会互相覆盖，SQLite 有事务。
多实例部署时把此类换成共享数据库（同一张表结构即可），API 不变。
默认关闭（feedback_enabled=0），开启后创建表；不依赖外部服务。
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT, get_settings

_CREATE = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    request_id TEXT,
    question_hash TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
    reason TEXT NOT NULL DEFAULT '',
    mode TEXT NOT NULL DEFAULT ''
)
"""


class FeedbackStore:
    def __init__(self, db_path: str):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with closing(self._connect()) as conn, conn:
            conn.execute(_CREATE)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def add(self, question_hash: str, rating: int, reason: str, request_id: str = "", mode: str = "") -> None:
        if rating not in (-1, 1):
            raise ValueError("rating must be -1 or 1")
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO feedback (created_at, request_id, question_hash, rating, reason, mode) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), request_id, question_hash, rating, reason[:200], mode),
            )

    def summary(self) -> dict:
        with self._lock, closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT rating, COUNT(*) FROM feedback GROUP BY rating"
            ).fetchall()
            recent = conn.execute(
                "SELECT question_hash, rating, reason FROM feedback "
                "ORDER BY id DESC LIMIT 20"
            ).fetchall()
        counts = {str(rating): n for rating, n in rows}
        up, down = counts.get("1", 0), counts.get("-1", 0)
        return {
            "total": up + down,
            "positive": up,
            "negative": down,
            "negative_rate": round(down / (up + down), 3) if up + down else 0.0,
            "recent": [
                {"question_hash": h, "rating": r, "reason": reason}
                for h, r, reason in recent
            ],
        }


_store: FeedbackStore | None = None
_lock = threading.Lock()


def get_store() -> FeedbackStore | None:
    """默认关闭；开启时延迟建库，避免只读部署也写文件。"""
    global _store
    settings = get_settings()
    if not settings.feedback_enabled:
        return None
    if _store is None:
        with _lock:
            if _store is None:
                _store = FeedbackStore(str(ROOT / settings.feedback_db))
    return _store
