"""LangGraph 共享状态。

关键设计：并行 agent 同时写 evidences / traces / token_usage，必须用 reducer 归并，
否则后写覆盖先写 —— 这是多智能体状态管理最常见的坑。
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


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
    # 工具返回的原始文档（由 tool_use 节点写入，fuse 节点统一转成 Evidence 参与融合）
    # 用 reducer 归并：并行节点都只追加，不互相覆盖
    tool_docs: Annotated[list[dict], operator.add]
    evidences: list[Evidence]
    # 会话历史：多轮对话时传入，query_rewriter 用它做指代消解
    history: list[dict]
    # 融合节点每轮覆盖写入：上一轮保留的 doc_id 与"本轮新增条数"。
    # 新增为 0 说明检索已枯竭，反思再跑一轮只是重复付费。
    fused_ids: list[str]
    new_evidence_last_round: int
    traces: Annotated[list[TraceEntry], operator.add]
    token_usage: Annotated[dict[str, int], merge_dict]
    critique: str
    grounded: bool
    reflection_count: int
    # ---- Human-in-the-loop ----
    # review_count: 已发生的人工确认次数。写在 resume 之后（中断时节点尚未返回），
    # 因此恢复后重跑同一节点时它能挡住第二次中断。
    review_count: int
    review_decision: str      # approve（带不确定性继续）| refine（再检索）| abort（拒答）
    review_note: str          # 人工补充说明，refine 时拼进下一轮 query_focus
    review_payload: dict      # 中断时给前端的确认单（恢复后回填，便于前端复现上下文）
    answer: str
    citations: list[str]
    final: str
