"""测试包初始化：强制离线配置，避免单元测试打到真实付费 API。

为什么放在包初始化里而不是各测试文件：unittest discover 会先导入包再导入用例模块，
在这里设环境变量能保证任何 src.* 模块被 import 之前就生效（settings 是 lru_cache 缓存的，
一旦某个模块先读到了 .env 里的真实 provider，后面再改环境变量已经来不及）。
"""
from __future__ import annotations

import os

os.environ["LLM_PROVIDER"] = "mock"
os.environ["QDRANT_PATH"] = ":memory:"
os.environ["VERBOSE"] = "0"
