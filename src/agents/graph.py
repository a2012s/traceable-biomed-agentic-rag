"""LangGraph 状态机：编排 → 重写 → 并行专家 → 反思 → 生成/回流。

反思回流用条件边实现：证据不足且未达最大迭代次数则回到 query_rewriter，
达到上限则强制生成并标注证据不足，避免死循环。

开启 HITL 时，机器"本来还要再检索一轮"的那一步改为在 human_review 节点中断，
由调用方用 Command(resume=...) 给出 approve/refine/abort 后再续跑。
"""
from __future__ import annotations

import logging
import uuid
from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from ..checkpoint import get_checkpointer
from ..config import get_settings
from ..logging_setup import configure_logging
from ..state import AgentState
from . import nodes

settings = get_settings()
logger = logging.getLogger(__name__)

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
    """反思是否继续：四道闸，缺一道都会让成本或延迟失控。

    1. 证据已充分 → 直接生成；
    2. 轮次上限 → 防止死循环（原有）；
    3. token 预算 → 每轮都是一次完整检索 + 生成，没有预算上限时成本不可控；
    4. 本轮零新增证据 → 检索已枯竭，再跑一轮结果不变、成本翻倍；
    5. 以上都通过（机器本来还要再检索一轮）→ 若开了 HITL，把决定权交给人工。
       放在最后是有意的：已经判定"该停"的请求不该再打断人。
    """
    if state.get("grounded"):
        return "generate"
    if state.get("reflection_count", 0) >= settings.max_reflections:
        return "generate"
    spent = sum(int(v) for v in (state.get("token_usage") or {}).values())
    if spent >= settings.max_tokens_per_request:
        logger.debug("token 预算耗尽（%s），停止反思", spent)
        return "generate"
    if state.get("new_evidence_last_round", 0) == 0:
        logger.debug("本轮无新增有效证据，停止反思")
        return "generate"
    if settings.hitl_enabled and state.get("review_count", 0) < settings.hitl_max_reviews:
        return "review"
    return "rewrite_again"


def route_after_review(state: AgentState) -> str:
    """人工决定的三个出口。

    refine 直接回 query_rewriter（不再经过 rewrite_again）：human_review 已经
    写好了 query_focus 并计入了反思轮次，再走一遍会重复计数。
    """
    decision = state.get("review_decision", "approve")
    if decision == "refine":
        return "query_rewriter"
    if decision == "abort":
        return "refuse"
    return "generate"


async def tool_gate(state: AgentState) -> dict:
    """fan-in 空节点：本身不改状态，只用于等待 tool_use 与 query_rewriter 两路完成。"""
    return {}


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
    graph.add_node("tool_use", nodes.tool_use)
    graph.add_node("tool_gate", tool_gate)
    for name in AGENT_NAMES:
        graph.add_node(name, AGENT_FUNCS[name])
    graph.add_node("critic", nodes.critic)
    graph.add_node("fuse", nodes.fuse)
    graph.add_node("rewrite_again", increment_reflection)
    graph.add_node("human_review", nodes.human_review)
    graph.add_node("refuse", nodes.refuse)
    graph.add_node("generate", nodes.generate)

    graph.add_edge(START, "orchestrator")
    # 工具选择与查询重写无数据依赖，并行分叉；tool_gate 作为 fan-in 汇聚点，
    # 等两路都写完再 fan-out 到域 agent。不能让两个源各自连条件边——那会重复触发 agent。
    graph.add_edge("orchestrator", "tool_use")
    graph.add_edge("orchestrator", "query_rewriter")
    graph.add_edge("tool_use", "tool_gate")
    graph.add_edge("query_rewriter", "tool_gate")
    graph.add_conditional_edges("tool_gate", route_to_agents, AGENT_NAMES)
    for name in AGENT_NAMES:
        # 并行 agent 先汇聚到融合节点（fan-in），再统一验收
        graph.add_edge(name, "fuse")
    graph.add_edge("fuse", "critic")
    graph.add_conditional_edges(
        "critic",
        after_critique,
        {"generate": "generate", "rewrite_again": "rewrite_again", "review": "human_review"},
    )
    # 人工确认是图上的真实节点：interrupt 在这里暂停，Command(resume=...) 从这里续跑
    graph.add_conditional_edges(
        "human_review",
        route_after_review,
        {"generate": "generate", "query_rewriter": "query_rewriter", "refuse": "refuse"},
    )
    graph.add_edge("rewrite_again", "query_rewriter")
    graph.add_edge("refuse", END)
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


async def run(
    question: str,
    verbose: bool | None = None,
    thread_id: str | None = None,
    history: list[dict] | None = None,
) -> dict:
    """跑一次完整问答。

    thread_id 每次唯一（除非刻意续跑同一会话）：checkpointer 按 thread_id 恢复状态，
    复用同一个 id 会让下一题的 evidences/reflection_count 接着上一题。

    history 是多轮对话的上下文（[{"question", "answer"}, ...]）。它与 thread_id 续跑
    是两条独立路径：history 只把**问题文本**喂给 query_rewriter 做指代消解，
    不恢复上一轮的内部状态——后者会让旧证据/反思计数泄漏到新问题，
    是「多轮越聊越离谱」的典型根因。

    返回约定：`interrupted=True` 表示图停在人工确认点，`review` 是确认单内容，
    需要把 `thread_id` 与决定一起交给 resume() 才能继续；此时不读也不写答案缓存。
    """
    if verbose is not None:
        configure_logging(verbose)

    from ..deps import get_answer_cache
    from ..observability import metrics

    # 缓存键必须区分历史：同一句「它有什么副作用？」在不同上下文里答案不同
    cache_key = question if not history else f"{question}||history:{len(history)}:" + "|".join(
        str(h.get("question", ""))[:60] for h in history[-3:]
    )
    cache = get_answer_cache()
    # HITL 开启时不读答案缓存：命中缓存会整段跳过图，人工确认环节就永远不触发，
    # 表现为「同一个问题有时弹确认、有时直接出答案」——比慢一点更难排查。
    hitl_on = bool(settings.hitl_enabled)
    if not hitl_on:
        hit = cache.get(cache_key)
        if hit is not None:
            metrics.cache_hit()
            cached = dict(hit)
            cached["cache_hit"] = True
            cached.setdefault("traces", [])
            return cached
        metrics.cache_miss()

    app = _compiled_app()
    tid = thread_id or f"run-{uuid.uuid4().hex}"
    result = await app.ainvoke(
        {"question": question, "history": history or []},
        config={"configurable": {"thread_id": tid}},
    )
    result["thread_id"] = tid
    result["cache_hit"] = False
    interrupted = result.pop("__interrupt__", None)
    if interrupted:
        # 中断态不写缓存：续跑结果取决于人工决定，缓存某一支会让下一次直接返回旧结论
        result["interrupted"] = True
        result["review"] = interrupted[0].value
        return result
    result["interrupted"] = False
    result["review"] = None
    if not hitl_on:
        cache.put(cache_key, result)
    return result


async def resume(thread_id: str, decision: str, note: str = "") -> dict:
    """按人工决定恢复被中断的图。

    thread_id 必须与中断时一致：LangGraph 用它找回该线程的检查点与待恢复的中断。
    恢复后仍可能再次中断（`HITL_MAX_REVIEWS>1` 时），所以这里与 run() 用同样的返回约定。
    """
    app = _compiled_app()
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await app.aget_state(config)
    if not snapshot.next:
        raise ValueError("该 thread_id 没有待人工确认的请求（可能已完成、已恢复或已过期）")

    result = await app.ainvoke(
        Command(resume={"decision": decision, "note": note}),
        config=config,
    )
    result["thread_id"] = thread_id
    result["cache_hit"] = False
    interrupted = result.pop("__interrupt__", None)
    if interrupted:
        result["interrupted"] = True
        result["review"] = interrupted[0].value
        return result
    result["interrupted"] = False
    result["review"] = None
    return result
