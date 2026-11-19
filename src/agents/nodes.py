"""LangGraph 节点实现：编排、查询重写、四个数据域专家、反思验收、生成。

依赖通过 `deps` 懒加载获取，不在导入期构建索引——详见 src/deps.py。
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from ..config import get_settings
from ..deps import get_llm, get_reranker, get_retriever
from ..llm import parse_json_loose
from ..logging_setup import TRACE_LOGGER
from ..state import AgentState, Evidence, TraceEntry
from ..tools.biomedical import dispatch
from ..utils.text import content_tokens
from . import tool_params
from .prompts import ANSWER_SYSTEM, ORCHESTRATE_SYSTEM, REWRITE_SYSTEM

settings = get_settings()
trace_logger = logging.getLogger(TRACE_LOGGER)

# 单一来源：编排器白名单与图节点注册共用，避免两处各写一份名单
AGENT_NAMES: tuple[str, ...] = ("literature", "clinical", "report", "kg")


def _trace(agent: str, action: str, detail: str, start: float, tokens: int = 0) -> TraceEntry:
    return TraceEntry(
        agent=agent,
        action=action,
        detail=detail,
        elapsed_ms=round((time.perf_counter() - start) * 1000, 1),
        tokens=tokens,
    )


def _verbose(entry: TraceEntry) -> None:
    trace_logger.debug(
        "[%s] %s: %s (%sms, %s tok)",
        entry["agent"], entry["action"], entry["detail"],
        entry["elapsed_ms"], entry["tokens"],
    )


async def orchestrator(state: AgentState) -> dict:
    start = time.perf_counter()
    # to_thread：invoke 内部是同步 HTTP，直接调用会阻塞整个事件循环，
    # Web 层的 Semaphore 限并发随之退化为 1（多请求排队等一个同步调用）
    raw = await asyncio.to_thread(
        get_llm().invoke,
        ORCHESTRATE_SYSTEM, state["question"], mock_key=f"orchestrate|{state['question']}",
    )
    result = parse_json_loose(raw.content)
    agents = [a for a in result.get("agents", []) if a in AGENT_NAMES] or ["literature", "kg"]
    plan = result.get("plan", "")[:200]
    entry = _trace("orchestrator", "plan", plan, start, raw.tokens)
    _verbose(entry)
    return {
        "domain": str(result.get("domain", "general"))[:80],
        "dispatch": agents,
        "plan": plan,
        "evidences": [],   # 用 reducer 初始化，占位不影响归并
        "traces": [entry],
        "token_usage": {"orchestrator": raw.tokens},
        "reflection_count": 0,
        "query_focus": "",
    }


async def query_rewriter(state: AgentState) -> dict:
    start = time.perf_counter()
    focus_hint = f"\n上一轮反思意见：{state.get('query_focus', '')}" if state.get("query_focus") else ""
    # 重写属于抽取类任务，路由到便宜模型；未配置 cheap_model 时自动复用主模型
    result, raw = await asyncio.to_thread(
        get_llm().json_invoke,
        REWRITE_SYSTEM,
        state["question"] + focus_hint,
        mock_key=f"rewrite|{state['question']}{focus_hint}",
        tier="cheap",
        return_meta=True,
    )
    queries = (result.get("rewritten_queries") or [state["question"]])[:5]
    abbrs = result.get("abbreviations_resolved", [])[:5]
    focus = str(result.get("query_focus", ""))[:200]
    entry = _trace(
        "query_rewriter", "rewrite",
        f"生成 {len(queries)} 条查询；消歧 {abbrs[:3]}",
        start, raw.tokens,
    )
    _verbose(entry)
    return {
        "rewritten_queries": queries,
        "query_focus": focus,
        "traces": [entry],
        "token_usage": {"query_rewriter": raw.tokens},
    }


def make_domain_agent(
    name: str,
    sources: set[str],
    tool_name: str,
    params_fn: Callable[[str, list[str] | None], dict],
):
    """工厂：四个数据域 agent 共享同一套「检索 → 重排 → 工具调用」逻辑。

    差异只在三处：可访问的数据源、要调的专属工具、工具入参抽取函数。
    """

    async def agent(state: AgentState) -> dict:
        start = time.perf_counter()
        queries = state.get("rewritten_queries") or [state["question"]]

        # 检索与工具调用并行：两者无数据依赖，串行只是浪费延迟
        candidates, tool_result = await asyncio.gather(
            asyncio.to_thread(get_retriever().retrieve, queries, settings.retrieve_top_k, sources),
            dispatch(tool_name, params_fn(state["question"], queries)),
            return_exceptions=True,
        )

        if isinstance(candidates, BaseException):
            candidates = await asyncio.to_thread(
                get_retriever().retrieve, queries, settings.retrieve_top_k, sources
            )
        if isinstance(tool_result, BaseException):
            # 不可恢复错误（FatalToolError）与可恢复错误耗尽重试，都只降级本工具，不中断整条流程
            tool_result = None

        # 检索用重写后的多条查询（放宽召回），打分只用「原问题 + 缩写展开」（收紧精度）。
        # 注意：expanded_query 只做语境义项替换，绝不能用重写器输出打分——
        # 那里含 "systematic review" 这类通用词，会让任何文档都"相关"（假 grounded）。
        # rerank 内有同步计算与可能的外部 HTTP，同样不能占用事件循环。
        ranked: list[Evidence] = await asyncio.to_thread(
            get_reranker().rerank, tool_params.expanded_query(state["question"]), candidates
        )
        # 专家节点只做检索与工具调用，不调用 LLM —— token 消耗为 0
        entry = _trace(
            name, "retrieve+rerank",
            f"召回 {len(candidates)} → 重排 {len(ranked)} 条；工具 {'ok' if tool_result else 'skipped'}",
            start, 0,
        )
        _verbose(entry)
        return {
            "candidates": ranked,   # 原始召回交融合节点，不直接当最终证据
            "traces": [entry],
            "token_usage": {name: 0},
        }

    agent.__name__ = name
    return agent


literature_agent = make_domain_agent("literature", {"literature"}, "pubmed_search", tool_params.pubmed_params)
clinical_agent = make_domain_agent("clinical", {"clinical"}, "clinicaltrials_lookup", tool_params.clinical_params)
report_agent = make_domain_agent("report", {"report"}, "report_lookup", tool_params.report_params)
kg_agent = make_domain_agent("kg", {"kg"}, "umls_expand", tool_params.umls_params)


async def fuse(state: AgentState) -> dict:
    """跨域融合节点：并行 agent 的逐域排序 → 一份全局证据序列。

    缺了这一步，生成节点拿到的永远是「第一个数据域的前 k 条」，
    多源召回等于白做。
    """
    start = time.perf_counter()
    candidates: list[Evidence] = state.get("candidates", [])
    fused = get_reranker().fuse(candidates, top_k=get_settings().evidence_top_k)
    # 记录本轮新增了多少条上一轮没有的证据：一条都没有说明这条查询已经检索枯竭，
    # 再反思一轮只是把同样的检索和生成成本再付一遍。
    prev_ids = set(state.get("fused_ids") or [])
    cur_ids = [e["doc_id"] for e in fused]
    new_count = len(set(cur_ids) - prev_ids)
    entry = _trace(
        "fuse", "cross-domain-fuse",
        f"候选 {len(candidates)} 条（{len({e['source'] for e in candidates})} 个数据域）→ "
        f"全局保留 {len(fused)} 条（新增 {new_count}）",
        start, 0,
    )
    _verbose(entry)
    return {
        "evidences": fused,
        "fused_ids": cur_ids,
        "new_evidence_last_round": new_count,
        "traces": [entry],
        "token_usage": {"fuse": 0},
    }


async def critic(state: AgentState) -> dict:
    start = time.perf_counter()
    evidences: list[Evidence] = state.get("evidences", [])
    high_quality = [e for e in evidences if e["evidence_level"] in ("1", "2")]
    distinct_sources = {e["source"] for e in evidences}

    # 反思判据（可解释、零成本，替代 LLM judge）：
    #   ① 至少 1 条高等级证据（RCT/系统评价/指南，等级 1-2）
    #   ② 至少 2 个独立数据源（单一来源等于单一视角，谈不上可审计）
    #   ③ 证据片段确实命中查询实体（防"检索到了但答非所问"）
    # 早期要求"≥2 条高等级"，在 15 篇文档的玩具语料上把 7/12 问题判为不足——
    # 阈值应与语料规模匹配，这里只保留"最低可信门槛"，宁可在生成侧标注不确定性。
    query_entities = content_tokens(tool_params.expanded_query(state["question"]))
    covered = sum(
        1 for e in evidences if query_entities & content_tokens(f"{e['title']} {e['snippet']}")
    )
    grounded = (
        len(high_quality) >= 1
        and len(distinct_sources) >= 2
        and covered >= 2
    )
    critique = (
        f"证据 {len(evidences)} 条，等级1-2共 {len(high_quality)} 条，"
        f"数据源 {sorted(distinct_sources)}，命中查询实体 {covered} 条；"
        + ("判定充分" if grounded else "判定不足，需补充高等级或更相关的证据")
    )
    # Critic 是基于证据字段的规则判定，不调用 LLM，因此不消耗 token
    entry = _trace("critic", "reflect", critique, start, 0)
    _verbose(entry)
    return {
        "critique": critique,
        "grounded": grounded,
        "traces": [entry],
        "token_usage": {"critic": 0},
    }


async def generate(state: AgentState) -> dict:
    start = time.perf_counter()
    llm = get_llm()
    evidences: list[Evidence] = state.get("evidences", [])
    # 白名单必须严格等于「实际送进 prompt 的集合」。取全部 evidences 会让模型
    # 猜中第 9 条 doc_id 时也能通过校验——证据标签格式很规整，容易被模式化生成。
    top_n = max(1, get_settings().evidence_top_k)
    prompt_evidences = evidences[:top_n]
    blocks = "\n".join(
        f"[{e['doc_id']}] ({e['source']}, level {e['evidence_level']}) {e['title']}: {e['snippet']}"
        for e in prompt_evidences
    )
    caveat = "" if state.get("grounded") else "\n注意：当前证据不足，以下结论仅供参考。"
    user = f"问题：{state['question']}\n\n证据：\n{blocks}{caveat}"
    valid_ids = {e["doc_id"] for e in prompt_evidences}
    citation_limit = min(4, len(prompt_evidences))
    if llm.is_mock:
        # Mock 模式：直接用可追溯的拼装答案，保证引用真实存在
        answer = _fallback_answer(state, prompt_evidences)
        citations = [e["doc_id"] for e in prompt_evidences[:citation_limit]]
        answer_tokens = 0  # 本地拼装，不调用 LLM
    else:
        result, raw = await asyncio.to_thread(
            llm.json_invoke, ANSWER_SYSTEM, user, tier="strong", return_meta=True
        )
        # 真实 LLM 返回引用后校验合法性，防幻觉引用
        citations = [c for c in result.get("citations", []) if c in valid_ids][:citation_limit]
        answer = result.get("answer") or _fallback_answer(state, prompt_evidences)
        answer_tokens = raw.tokens
    final = f"{answer}\n\n证据来源：{', '.join(citations) if citations else '未找到可引用证据'}"
    entry = _trace("generate", "answer", f"生成 {len(answer)} 字，有效引用 {len(citations)} 条", start, answer_tokens)
    _verbose(entry)
    return {
        "answer": answer,
        "citations": citations,
        "final": final,
        "traces": [entry],
        "token_usage": {"generate": answer_tokens},
    }


def _fallback_answer(state: AgentState, evidences: list[Evidence]) -> str:
    """Mock 模式下拼装可追溯回答，行为与真实模型一致：只引用检索到的证据。"""
    top = evidences[:4]
    if not top:
        return "未检索到相关证据，无法给出有依据的回答。"
    lines = [f"针对「{state['question']}」，检索到以下关键证据："]
    for i, e in enumerate(top, 1):
        lines.append(f"{i}. [{e['doc_id']}] {e['title']}：{e['snippet']}")
    return "\n".join(lines)
