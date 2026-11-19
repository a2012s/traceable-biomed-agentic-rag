"""LangGraph 状态机：编排 → 重写 → 并行专家 → 反思 → 生成/回流。

反思回流用条件边实现：证据不足且未达最大迭代次数则回到 query_rewriter，
达到上限则强制生成并标注证据不足，避免死循环。
"""
from __future__ import annotations

import uuid
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from ..checkpoint import get_checkpointer
from ..config import get_settings
from ..logging_setup import configure_logging
from ..state import AgentState
from . import nodes

settings = get_settings()

AGENT_NAMES = list(nodes.AGENT_NAMES)
AGENT_FUNCS = {
    "literature": nodes.literature_agent,
    "clinical": nodes.clinical_agent,
    "report": nodes.report_agent,
    "kg": nodes.kg_agent,
}


def route_to_agents(state: AgentState) -> list[str]:
    """并行分派：orchestrator 决定哪些数据域专家参与本轮。

    LangGraph 条件边返回列表时会并行执行这些节点，并在汇聚节点等待全部完成。
    """
    dispatch = state.get("dispatch") or ["literature"]
    selected = [name for name in dispatch if name in AGENT_NAMES]
    return selected or ["literature"]


def after_critique(state: AgentState) -> str:
    if state.get("grounded"):
        return "generate"
    if state.get("reflection_count", 0) < settings.max_reflections:
        return "rewrite_again"
    return "generate"


def increment_reflection(state: AgentState) -> dict:
    """回流前把反思意见写进 query_focus。

    不清空 evidences：reducer 已按 doc_id 去重，保留上一轮证据才能「累积」
    而不是每轮推倒重来（清空的实现会让 Recall 随反思轮次下降）。
    """
    return {
        "reflection_count": state.get("reflection_count", 0) + 1,
        "query_focus": state.get("critique", ""),
        "traces": [{
            "agent": "orchestrator",
            "action": "reflect-loop",
            "detail": f"第 {state.get('reflection_count', 0) + 1} 轮反思回流，重新检索",
            "elapsed_ms": 0.0,
            "tokens": 0,
        }],
        "token_usage": {},
    }


def build_graph(verbose: bool | None = None):
    """编译状态机；verbose 控制决策链日志（不再直接改全局 settings）。"""
    configure_logging(settings.verbose if verbose is None else verbose)

    graph = StateGraph(AgentState)
    graph.add_node("orchestrator", nodes.orchestrator)
    graph.add_node("query_rewriter", nodes.query_rewriter)
    for name in AGENT_NAMES:
        graph.add_node(name, AGENT_FUNCS[name])
    graph.add_node("critic", nodes.critic)
    graph.add_node("fuse", nodes.fuse)
    graph.add_node("rewrite_again", increment_reflection)
    graph.add_node("generate", nodes.generate)

    graph.add_edge(START, "orchestrator")
    graph.add_edge("orchestrator", "query_rewriter")
    graph.add_conditional_edges("query_rewriter", route_to_agents, AGENT_NAMES)
    for name in AGENT_NAMES:
        # 并行 agent 先汇聚到融合节点（fan-in），再统一验收
        graph.add_edge(name, "fuse")
    graph.add_edge("fuse", "critic")
    graph.add_conditional_edges(
        "critic",
        after_critique,
        {"generate": "generate", "rewrite_again": "rewrite_again"},
    )
    graph.add_edge("rewrite_again", "query_rewriter")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=get_checkpointer())


@lru_cache(maxsize=1)
def _compiled_app():
    """复用编译后的状态机。

    每次请求重新 compile 会把 LangGraph 的图结构检查与 checkpointer 绑定重复做一遍；
    在并发下这是纯浪费，且会创建多个 SqliteSaver 连接。configure_logging 在 run() 里
    单独处理，不进缓存键 —— 日志级别不是图结构的一部分。
    """
    return build_graph(verbose=None)


async def run(question: str, verbose: bool | None = None, thread_id: str | None = None) -> dict:
    """跑一次完整问答。

    thread_id 必须每次唯一（除非刻意续跑同一会话）：checkpointer 按 thread_id 恢复状态，
    复用同一个 id 会让下一题的 evidences/reflection_count 接着上一题，
    批量评估时表现为指标忽高忽低（且极难定位）。
    """
    if verbose is not None:
        configure_logging(verbose)

    from ..deps import get_answer_cache
    from ..observability import metrics

    cache = get_answer_cache()
    hit = cache.get(question)
    if hit is not None:
        metrics.cache_hit()
        cached = dict(hit)
        cached["cache_hit"] = True
        cached.setdefault("traces", [])
        return cached
    metrics.cache_miss()

    app = _compiled_app()
    result = await app.ainvoke(
        {"question": question},
        config={"configurable": {"thread_id": thread_id or f"run-{uuid.uuid4().hex}"}},
    )
    result["cache_hit"] = False
    cache.put(question, result)
    return result
