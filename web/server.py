"""FastAPI 演示后端：把 LangGraph 的 run() 暴露成可点击的单页。

安全约束：默认强制 Mock 模式，公网演示不会调用 .env 里的付费 API。
只有显式设置 WEB_ALLOW_REAL_LLM=1 才允许真实模型（本地调试用）。
必须在导入任何 src.* 之前完成这个设置 —— config.get_settings() 会缓存配置。

生产化增量（默认值保持离线 Demo 行为不变）：
- asyncio.Semaphore 限并发 + 单请求超时，避免多 worker 下雪崩；
- request_id 贯穿响应头与响应体，便于跨日志定位；
- /api/metrics 暴露请求数、错误数、延迟分位、缓存命中率；
- 答案缓存命中直接跳过图执行（键含语料版本与模型档位）。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time

if os.environ.get("WEB_ALLOW_REAL_LLM", "0") != "1":
    os.environ["LLM_PROVIDER"] = "mock"

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.agents.graph import run as run_graph
from src.cache import normalize_question
from src.config import get_settings
from src.feedback import get_store
from src.observability import metrics

WEB_DIR = Path(__file__).resolve().parent
INDEX_HTML = WEB_DIR / "index.html"
settings = get_settings()

app = FastAPI(
    title="Agentic RAG · 交互式演示",
    description="多智能体 RAG 决策链与引用溯源可视化",
    version="0.3.0",
)

EXAMPLES = [
    "EGFR 突变非小细胞肺癌的一线治疗方案是什么？",
    "MS 的诊断标准是什么？",
    "mass spectrometry 质谱分析在血清蛋白定量中的应用",
    "CRISPR base editing 在镰状细胞病中的长期安全性如何？",
]

AGENT_LABELS = {
    "orchestrator": "编排器",
    "query_rewriter": "查询重写",
    "literature": "文献 agent",
    "clinical": "临床试验 agent",
    "report": "影像报告 agent",
    "kg": "知识图谱 agent",
    "fuse": "跨域融合",
    "critic": "Critic 验收",
    "reflect-loop": "反思回流",
    "generate": "答案生成",
}


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    thread_id: str | None = Field(default=None, max_length=100)
    # 多轮上下文：只传问题和答案文本，不传内部状态（避免旧证据污染新一轮）
    history: list[dict] = Field(default_factory=list, max_length=10)


class FeedbackRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    rating: int = Field(ge=-1, le=1)
    reason: str = Field(default="", max_length=200)
    request_id: str = Field(default="", max_length=64)


def _mode() -> str:
    return "mock" if settings.is_mock else "real"


_semaphore = asyncio.Semaphore(max(1, settings.max_concurrency))

# 只接受字母数字与短横线：客户端传来的 id 会回显到响应头与日志，
# 不校验就等于开放日志注入/响应头注入（换行符、控制字符、超长串）。
_REQ_ID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")


def _request_id(request: Request) -> str:
    incoming = request.headers.get("x-request-id", "")
    return incoming if _REQ_ID_RE.match(incoming) else metrics.new_request_id()


async def _execute(question: str, thread_id: str | None, history: list[dict] | None = None) -> dict:
    """带并发闸门与超时的链路执行。"""
    async with _semaphore:
        return await asyncio.wait_for(
            run_graph(question.strip(), verbose=False, thread_id=thread_id, history=history),
            timeout=settings.request_timeout_s,
        )


def _payload(state: dict, request_id: str, latency_ms: float) -> dict:
    return {
        "request_id": request_id,
        "question": state.get("question", ""),
        "answer": state.get("answer", ""),
        "final": state.get("final", ""),
        "citations": state.get("citations", []),
        "evidences": state.get("evidences", []),
        "traces": state.get("traces", []),
        "plan": state.get("plan", ""),
        "dispatch": state.get("dispatch", []),
        "rewritten_queries": state.get("rewritten_queries", []),
        "query_focus": state.get("query_focus", ""),
        "critique": state.get("critique", ""),
        "grounded": bool(state.get("grounded", False)),
        "reflection_count": state.get("reflection_count", 0),
        "token_usage": state.get("token_usage", {}),
        "latency_ms": round(latency_ms, 1),
        "mode": _mode(),
        "cache_hit": bool(state.get("cache_hit", False)),
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "mode": _mode(),
        "engine": "LangGraph + 本地向量库 + TF-IDF",
        "checkpointer": settings.checkpointer,
        "max_concurrency": settings.max_concurrency,
        "cache_ttl_s": settings.answer_cache_ttl_s,
    }


@app.get("/api/examples")
async def examples() -> dict:
    return {"examples": EXAMPLES, "agent_labels": AGENT_LABELS}


@app.get("/api/metrics")
async def get_metrics() -> dict:
    """进程内指标快照；多实例部署时由网关/监控聚合。"""
    return metrics.snapshot()


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest) -> dict:
    """记录赞/踩。只存问题哈希，不存问题原文（隐私最小化）。"""
    import hashlib

    store = get_store()
    if store is None:
        raise HTTPException(status_code=404, detail="反馈功能未开启（FEEDBACK_ENABLED=1）")
    if req.rating not in (-1, 1):
        raise HTTPException(status_code=422, detail="rating 只能是 -1 或 1")
    question_hash = hashlib.sha256(normalize_question(req.question).encode()).hexdigest()[:16]
    store.add(question_hash, req.rating, req.reason, req.request_id, _mode())
    return {"ok": True}


@app.get("/api/feedback/summary")
async def feedback_summary() -> dict:
    store = get_store()
    if store is None:
        raise HTTPException(status_code=404, detail="反馈功能未开启（FEEDBACK_ENABLED=1）")
    return store.summary()


@app.post("/api/ask")
async def ask(req: AskRequest, request: Request) -> JSONResponse:
    """跑一次完整问答，返回决策链 + 证据 + 答案。"""
    request_id = _request_id(request)
    started = time.perf_counter()
    metrics.start()
    ok = False
    try:
        state = await _execute(req.question, req.thread_id, req.history)
        payload = _payload(state, request_id, (time.perf_counter() - started) * 1000)
        ok = True
        return JSONResponse(payload, headers={"X-Request-ID": request_id})
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail=f"链路超时（{settings.request_timeout_s}s）[{request_id}]") from exc
    except Exception as exc:  # noqa: BLE001 - 演示页需要把失败原因显示给用户
        raise HTTPException(status_code=500, detail=f"链路执行失败（{request_id}）：{exc}") from exc
    finally:
        metrics.finish(started, ok=ok)


@app.post("/api/ask/stream")
async def ask_stream(req: AskRequest, request: Request) -> StreamingResponse:
    """SSE 流式返回。

    当前图执行是整条链路终止语义（generate 节点一次性返回），因此推送粒度为
    「链路完成」这一帧；token 级流式需把 generate 节点改成 stream 客户端，
    属于后续改造，不在本次范围。保留 SSE 端点是为了前端接入方式与生产一致。
    """
    request_id = _request_id(request)
    started = time.perf_counter()
    metrics.start()

    async def events():
        ok = False
        try:
            state = await _execute(req.question, req.thread_id, req.history)
            payload = _payload(state, request_id, (time.perf_counter() - started) * 1000)
            yield f"event: result\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"
            ok = True
        except asyncio.TimeoutError:
            yield f"event: error\ndata: {json.dumps({'message': '链路超时', 'request_id': request_id}, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"event: error\ndata: {json.dumps({'message': str(exc), 'request_id': request_id}, ensure_ascii=False)}\n\n"
        finally:
            metrics.finish(started, ok=ok)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Request-ID": request_id},
    )
