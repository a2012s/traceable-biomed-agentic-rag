"""日志配置：库代码只打日志，CLI 决定是否展示。

之前用 print 输出决策链有两个问题：
- 作为库被 import 时无处关闭，污染调用方 stdout（`python -c "import src..."` 也会打印）；
- 无法分级，排障时没法只看 warning。

改为标准 logging：trace 走 `agentic_rag.trace` logger，降级告警走 `agentic_rag` logger。
"""
from __future__ import annotations

import logging

TRACE_LOGGER = "agentic_rag.trace"

_FORMAT = "%(levelname)-7s %(name)s | %(message)s"


def configure_logging(verbose: bool | int = False) -> None:
    """配置根 logger；verbose 时打开决策链的实时日志。

    CLI 本身会渲染一份决策链表格，默认不重复打印日志；
    VERBOSE=1 时才有实时日志（真实模型跑得慢，需要看进度时用）。
    """
    logging.basicConfig(level=logging.INFO, format=_FORMAT, force=True)
    trace_logger = logging.getLogger(TRACE_LOGGER)
    trace_logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    # 第三方库在调试时噪音过大
    for noisy in ("httpx", "httpcore", "urllib3", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
