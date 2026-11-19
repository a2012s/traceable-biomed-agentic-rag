"""模型客户端：真实 API / Mock 双模式，带重试、分层路由与 token 计量。

真实模式走 OpenAI 兼容协议（OpenAI / DeepSeek / 百炼通用），避免绑定单一厂商 SDK。
可恢复错误（超时/429/5xx）用指数退避重试；不可恢复错误（401/参数错误）立即失败。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import get_settings


class UnrecoverableLLMError(Exception):
    """授权失败、参数错误等重试无意义的错误。"""


@dataclass
class LLMResponse:
    content: str
    model: str
    tokens: int = 0
    elapsed_ms: float = 0.0


@dataclass
class LLMClient:
    """按任务复杂度路由到强/弱模型；Mock 模式下所有调用本地规则生成。"""

    settings: object = field(default_factory=get_settings)
    _call_count: int = field(default=0, init=False)

    @property
    def is_mock(self) -> bool:
        return self.settings.is_mock

    def model_for(self, tier: str = "strong") -> str:
        s = self.settings
        if s.is_mock:
            return "mock-rule-engine"
        if tier == "cheap":
            return s.cheap_model or s.active_model
        return s.strong_model or s.active_model

    def estimate_tokens(self, text: str) -> int:
        # 中英混排近似：CJK 一字一 token，英文约 4 字符一 token
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        return cjk + max(1, (len(text) - cjk) // 4)

    def invoke(self, system: str, user: str, tier: str = "strong", *, mock_key: str = "") -> LLMResponse:
        start = time.perf_counter()
        self._call_count += 1
        model = self.model_for(tier)
        if self.is_mock:
            content = self._mock_response(mock_key or user, system)
            tokens = self.estimate_tokens(system + user) + self.estimate_tokens(content)
        else:
            content, usage = self._call_api(system, user, model)
            # 用 API 返回的真实 token 数，推理模型的 reasoning_tokens 计入成本
            tokens = usage.get("total_tokens") or usage.get("completion_tokens", 0)
        elapsed = (time.perf_counter() - start) * 1000
        return LLMResponse(
            content=content,
            model=model,
            tokens=tokens,
            elapsed_ms=elapsed,
        )

    def json_invoke(
        self,
        system: str,
        user: str,
        *,
        mock_key: str = "",
        tier: str = "strong",
        return_meta: bool = False,
    ) -> dict | tuple[dict, LLMResponse]:
        """要求返回 JSON，并做容错解析（模型常在 JSON 外包 markdown 围栏）。

        return_meta=True 时一并返回原始响应，供节点记录真实 token 用量。
        """
        raw = self.invoke(system, user, tier=tier, mock_key=mock_key)
        result = parse_json_loose(raw.content)
        return (result, raw) if return_meta else result

    @retry(
        retry=retry_if_exception_type((urllib.error.URLError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    def _call_api(self, system: str, user: str, model: str) -> tuple[str, dict]:
        s = self.settings
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            f"{s.active_base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {s.active_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")[:300]
            if exc.code in (401, 403, 400, 404):
                raise UnrecoverableLLMError(f"HTTP {exc.code}: {detail}") from exc
            raise  # 429 / 5xx 交给 tenacity 重试
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        return content, usage

    def _mock_response(self, key: str, system: str) -> str:
        """规则引擎模拟 LLM 结构化输出，保证无 key 也能演示完整闭环。

        key 约定为 "kind|payload"：kind 决定模拟哪种节点，payload 是真正的输入。
        早前把二者拼成一个句子，导致模拟出的查询里带上 "rewrite " 前缀，
        既污染检索又让 trace 难以阅读。
        """
        kind, _, payload = key.partition("|")
        probe = (kind or key).lower()
        body = payload or key

        if probe.startswith("orchestrate") or "编排" in system:
            return json.dumps(_mock_route(body), ensure_ascii=False)

        if probe.startswith("rewrite") or "重写" in system:
            return json.dumps(_mock_rewrite(body), ensure_ascii=False)

        if probe.startswith("critique") or "验收" in system:
            # 实际判定由 critic 节点基于证据计算，这里只提供兜底结构
            return json.dumps({"grounded": True, "critique": "证据充分"}, ensure_ascii=False)

        if probe.startswith("ground") or "依据" in system:
            return json.dumps(_mock_ground(body), ensure_ascii=False)

        return json.dumps({"answer": ""}, ensure_ascii=False)


def parse_json_loose(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        return {}


def _mock_route(question: str) -> dict:
    from .config import get_settings
    from .data.corpus import DOMAIN_HINTS

    q = question.lower()
    domains = [name for name, hints in DOMAIN_HINTS.items() if any(h in q for h in hints)]
    if not domains:
        domains = ["general"]

    # 召回优先：规则路由无法枚举所有医学表述（"brain MRI demyelination" 就匹配不到任何
    # 领域关键词），窄召回会直接把报告域漏掉。默认全量并行，专家 agent 不调 LLM，成本可控。
    if get_settings().wide_recall:
        agents = ["literature", "clinical", "report", "kg"]
        plan = f"领域 {domains}；召回优先，并行分派四个数据域专家，再由 Critic 判充分性"
    else:
        agents = ["literature", "kg"]
        if any(d in ("diabetes", "ms") for d in domains):
            agents.append("clinical")
        if "ms" in domains or "lung" in domains:
            agents.append("report")
        plan = f"领域 {domains}；窄召回，按领域分派 {agents}"
    return {
        "domain": ",".join(domains),
        "agents": agents,
        "plan": plan,
    }


def _mock_rewrite(question: str) -> dict:
    from .data.corpus import ABBREVIATIONS, resolve_senses

    # nodes.py 会把上一轮反思意见拼到 question 后面，这里据此生成针对性查询
    if "上一轮反思意见" in question:
        base, _, focus = question.partition("\n上一轮反思意见：")
    else:
        base, focus = question, ""

    # 只展开语境相符的义项：问"诊断标准"时不把 "mass spectrometry" 也塞进查询，
    # 否则质谱方法学论文会被召回并当成证据（MS 消歧做错方向比不做更糟）。
    senses = resolve_senses(base)
    terms: list[str] = []
    for abbr, expansions in ABBREVIATIONS.items():
        if abbr in base.lower().split() or abbr in base.lower():
            terms.extend(senses.get(abbr) or list(expansions.keys()))
    expanded = " ".join([base] + terms[:2]) if terms else base

    # 反思轮次：追加高等级证据导向词，模拟 Self-RAG 的针对性再检索
    focus_queries = []
    if focus:
        focus_queries = [
            f"{base} randomized controlled trial systematic review",
            f"{base} guideline recommendation evidence",
        ]
    return {
        "rewritten_queries": list(dict.fromkeys([base, expanded] + focus_queries)),
        "query_focus": "优先证据等级 1 的系统评价/RCT，并补充阴性结果与不良事件",
        "abbreviations_resolved": terms,
    }


def _mock_ground(question: str) -> dict:
    return {"answer": "基于检索到的证据生成回答（Mock 模式）。"}
