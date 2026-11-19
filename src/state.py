"""LangGraph 共享状态。

关键设计：并行 agent 同时写 evidences / traces / token_usage，必须用 reducer 归并，
否则后写覆盖先写 —— 这是多智能体状态管理最常见的坑。
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

RESET_SENTINEL = "__reset__"  # 已废弃：反思回流改为「去重累积」，保留常量避免外部引用报错


def merge_dict(left: dict | None, right: dict | None) -> dict:
    out = dict(left or {})
    for key, value in (right or {}).items():
        out[key] = out.get(key, 0) + value
    return out


def merge_evidences(left: list | None, right: list | None) -> list:
    """按 doc_id 去重后追加，同 ID 保留分数更高的那条。

    两个场景必须去重而非覆盖：
    - 并行 agent 会召回同一篇文档（文献既进 BM25 也进向量召回）；
    - 反思回流会重新检索到上一轮已有的证据。
    早期实现用「哨兵清空」，代价是反思一轮就把上一轮有效证据全丢了，
    端到端 Recall 反而低于单轮检索（实测 0.868 vs 0.903）。改为累积+去重后修复。
    """
    merged = list(left or [])
    index = {e["doc_id"]: i for i, e in enumerate(merged)}
    for item in right or []:
        if not isinstance(item, dict) or "doc_id" not in item:
            continue
        pos = index.get(item["doc_id"])
        if pos is None:
            index[item["doc_id"]] = len(merged)
            merged.append(item)
        elif item.get("score", 0.0) > merged[pos].get("score", 0.0):
            merged[pos] = item
    return merged


class Evidence(TypedDict):
    doc_id: str
    source: str          # literature | clinical | report | kg
    title: str
    snippet: str
    url: str
    study_type: str
    evidence_level: str  # 1=系统评价/RCT, 2=队列, 3=病例, 4=专家意见
    score: float


class TraceEntry(TypedDict):
    agent: str
    action: str
    detail: str
    elapsed_ms: float
    tokens: int


class AgentState(TypedDict, total=False):
    question: str
    domain: str                       # 医学子领域，决定分派哪些专家
    dispatch: list[str]               # 编排器决定本轮参与的数据域 agent
    rewritten_queries: list[str]
    query_focus: str                  # 反思后传给下一轮的检索侧重
    plan: str
    # candidates: 并行 agent 的原始召回（去重累积，供融合节点消费）
    # evidences:  跨域融合后的最终证据（覆盖语义，只有融合节点写）
    candidates: Annotated[list[Evidence], merge_evidences]
    evidences: list[Evidence]
    traces: Annotated[list[TraceEntry], operator.add]
    token_usage: Annotated[dict[str, int], merge_dict]
    critique: str
    grounded: bool
    reflection_count: int
    answer: str
    citations: list[str]
    final: str
