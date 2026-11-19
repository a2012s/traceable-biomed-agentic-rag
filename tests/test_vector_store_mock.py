"""QdrantLocal 逻辑验证：用 mock 模块模拟 qdrant_client，无需真实安装。

目的：证明 Qdrant 分支的 upsert/search 调用序列正确；
真实安装与否只影响"用哪个后端"，不影响这段代码的可信度。
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock


class FakeQdrant:
    """最小 qdrant_client 替身。"""

    def __init__(self):
        self.QdrantClient = MagicMock()
        self.models = types.SimpleNamespace(
            Distance=types.SimpleNamespace(COSINE="Cosine"),
            VectorParams=MagicMock(),
            PointStruct=MagicMock(),
        )


class TestQdrantLocalMock(unittest.TestCase):
    def test_upsert_and_search_calls(self):
        fake = FakeQdrant()
        original = sys.modules.get("qdrant_client")
        original_models = sys.modules.get("qdrant_client.models")
        try:
            sys.modules["qdrant_client"] = fake
            sys.modules["qdrant_client.models"] = fake.models
            # 强制重新导入（模块级 import 在函数内进行）
            from src.rag.vector_store import QdrantLocal

            client = fake.QdrantClient.return_value
            client.collection_exists.return_value = False
            client.query_points.return_value = types.SimpleNamespace(
                points=[types.SimpleNamespace(id="uuid-a", score=0.9, payload={"_doc_id": "a"})]
            )

            store = QdrantLocal(collection="test", dim=3, path=":memory:")
            store.upsert(["a"], [[1, 0, 0]], [{"k": "v"}])
            client.upsert.assert_called_once()

            hits = store.search([1, 0, 0], top_k=2)
            self.assertEqual(hits[0][0], "a")
            self.assertAlmostEqual(hits[0][1], 0.9)
            client.query_points.assert_called_once()
        finally:
            for key, mod in (("qdrant_client", original), ("qdrant_client.models", original_models)):
                if mod is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = mod


if __name__ == "__main__":
    unittest.main()
