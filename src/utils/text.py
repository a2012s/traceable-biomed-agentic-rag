"""共享分词器与停用词。

检索、重排、工具参数抽取必须用同一套切分，否则查询与文档落在不同词空间。
停用词在这里统一维护：若各模块各写一份，重排分数会随调用点漂移。
"""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")

# 停用词不参与相关性计算，否则 "is/the/的/什么" 会稀释真实实体信号
STOPWORDS = frozenset({
    "is", "are", "the", "a", "an", "of", "in", "for", "and", "or", "to", "with", "on",
    "what", "how", "does", "do", "did", "be", "by", "from", "as", "at", "which", "that",
    "的", "是", "什", "么", "什么", "怎么", "如何", "吗", "呢", "和", "与", "及",
    "有", "在", "中", "对", "用", "为", "了", "这", "那", "些", "无",
})


def tokenize(text: str) -> list[str]:
    """保留全部 token（含停用词），供需要完整词表的场景使用。"""
    return TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> set[str]:
    """只保留实词：去停用词、去单字符（中文单字歧义过大，不参与匹配）。"""
    return {t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1}
