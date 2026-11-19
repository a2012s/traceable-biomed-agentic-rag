"""生物医学工具封装：Pydantic 强类型入参 + 幂等 + 异步 + 错误二分。

设计要点（对应 JD 第 3 条）：
- 单一注册表：schema、handler、描述放在同一个 ToolSpec 里，
  避免"加了工具忘了改映射表"这种两处维护导致的线上 KeyError；
- 入参用 Pydantic 校验，杜绝模型幻觉参数直接打到下游；
- 可恢复错误（超时/限流/无命中）带指数退避重试，不可恢复错误（参数非法/未知工具）立即失败；
- async 接口支持 asyncio.gather 并发调度；
- `to_mcp_tools()` 直接产出 MCP 工具描述，专有算法即插即用。
"""
from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, field_validator
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..data.corpus import ABBREVIATIONS, CLINICAL, KG_TRIPLES, LITERATURE, REPORTS

logger = logging.getLogger(__name__)


class RetryableToolError(Exception):
    """超时、限流等可恢复错误。"""


class FatalToolError(Exception):
    """参数非法、授权不足等不可恢复错误。"""


class PubMedQuery(BaseModel):
    keywords: list[str] = Field(min_length=1, max_length=5)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("keywords")
    @classmethod
    def strip_keywords(cls, v: list[str]) -> list[str]:
        cleaned = [x.strip().lower() for x in v if x.strip()]
        if not cleaned:
            raise ValueError("keywords cannot be empty")
        return cleaned


class TrialQuery(BaseModel):
    indication: str = Field(min_length=2)
    phase: str | None = Field(default=None, pattern=r"^(phase-[1234]|1|2|3|4)$")


class UMLSQuery(BaseModel):
    concept: str = Field(min_length=2)


class ReportQuery(BaseModel):
    keywords: list[str] = Field(min_length=1, max_length=5)
    top_k: int = Field(default=5, ge=1, le=20)


@retry(
    retry=retry_if_exception_type(RetryableToolError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.2, max=2),
    reraise=True,
)
async def pubmed_search(params: PubMedQuery) -> list[dict[str, Any]]:
    """从文献库检索；真实实现替换为 NCBI E-utilities / 内部向量库。"""
    await asyncio.sleep(random.uniform(0.01, 0.05))  # 模拟网络抖动
    hits = []
    for doc in LITERATURE:
        text = f"{doc.title} {doc.text}".lower()
        if any(kw in text for kw in params.keywords):
            hits.append(doc.as_dict())
    if not hits:
        raise RetryableToolError("no hit, retry with rewritten keywords")
    return hits[: params.top_k]


@retry(
    retry=retry_if_exception_type(RetryableToolError),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.2, max=1),
    reraise=True,
)
async def clinicaltrials_lookup(params: TrialQuery) -> list[dict[str, Any]]:
    """查询临床试验；phase 过滤模拟外部 API 的严格参数校验。"""
    await asyncio.sleep(random.uniform(0.01, 0.04))
    out = []
    for doc in CLINICAL:
        phase = doc.study_type.replace("-trial", "")
        if params.indication.lower() in f"{doc.title} {doc.text}".lower():
            if params.phase and params.phase not in phase:
                continue
            out.append(doc.as_dict())
    return out


async def umls_expand(params: UMLSQuery) -> dict[str, Any]:
    """概念扩展 + 缩写消歧。纯本地词典，不抛可恢复错误。"""
    q = params.concept.lower()
    matches = [d.as_dict() for d in KG_TRIPLES if q in f"{d.title} {d.text}".lower()]
    expansions = ABBREVIATIONS.get(q, {})
    return {"concept": params.concept, "expansions": expansions, "kg_docs": matches}


async def report_lookup(params: ReportQuery) -> list[dict[str, Any]]:
    """检索非结构化报告（病理/影像）；真实实现接 PACS / LIS / 内部报告库。

    与 pubmed_search 分开的理由：报告的检索键是部位、检查类型、分子标志，
    和文献的关键词体系不同，混在一个工具里会导致参数语义漂移。
    """
    hits = []
    for doc in REPORTS:
        text = f"{doc.title} {doc.text}".lower()
        if any(kw in text for kw in params.keywords):
            hits.append(doc.as_dict())
    return hits[: params.top_k]


@dataclass(frozen=True)
class ToolSpec:
    """一个工具的全部元信息：单一来源，注册与调度不再各维护一份。"""

    name: str
    schema: type[BaseModel]
    handler: Callable[[Any], Awaitable[Any]]
    description: str


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="pubmed_search",
        schema=PubMedQuery,
        handler=pubmed_search,
        description="在生物医学文献库中按关键词检索，返回带证据等级的文献片段。",
    ),
    ToolSpec(
        name="clinicaltrials_lookup",
        schema=TrialQuery,
        handler=clinicaltrials_lookup,
        description="按适应证（可选分期）查询临床试验记录。",
    ),
    ToolSpec(
        name="umls_expand",
        schema=UMLSQuery,
        handler=umls_expand,
        description="UMLS 风格的概念扩展与缩写消歧（如 MS = 多发性硬化 / 质谱分析）。",
    ),
    ToolSpec(
        name="report_lookup",
        schema=ReportQuery,
        handler=report_lookup,
        description="检索病理/影像等非结构化报告的原文片段（部位、检查类型、分子标志）。",
    ),
)

TOOL_REGISTRY: dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}


async def dispatch(name: str, raw_params: dict[str, Any]) -> Any:
    """统一调度入口：按工具名查注册表 → Pydantic 校验 → 执行。

    校验失败属不可恢复错误，直接抛 FatalToolError，不做无意义重试。
    """
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        raise FatalToolError(f"unknown tool: {name}")
    try:
        params = spec.schema(**raw_params)
    except Exception as exc:
        raise FatalToolError(f"invalid params for {name}: {exc}") from exc
    logger.debug("tool call %s params=%s", name, params.model_dump())
    return await spec.handler(params)


def to_mcp_tools() -> list[dict[str, Any]]:
    """导出 MCP 工具描述（name / description / inputSchema），供 MCP Server 直接注册。"""
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "inputSchema": spec.schema.model_json_schema(),
        }
        for spec in TOOL_SPECS
    ]
