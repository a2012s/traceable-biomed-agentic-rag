#!/usr/bin/env python3
"""CLI 入口：打印完整 agent 决策链、耗时、token 与最终带引用回答。

用法：
    python run.py                       # 默认问题
    python run.py "EGFR 突变肺癌一线治疗方案?"  # 指定问题
"""
from __future__ import annotations

import asyncio
import sys
import time

from src.agents.graph import run
from src.config import get_settings
from src.logging_setup import configure_logging

DEFAULT_QUESTIONS = [
    "SGLT2 inhibitors for type 2 diabetes and kidney outcomes?",
    "EGFR 突变非小细胞肺癌的一线治疗方案是什么？",
    "MS 的诊断标准是什么？",
]


def banner(settings) -> None:
    mode = "Mock 规则引擎" if settings.is_mock else f"真实模型 {settings.active_model}"
    print("=" * 68, flush=True)
    print(f" Traceable Biomed Agentic RAG · 生物医学多智能体演示   模型模式: {mode}", flush=True)
    print("=" * 68, flush=True)


def show_trace(result: dict) -> None:
    print("\n--- 决策链（按执行顺序）---")
    total_ms = 0.0
    total_tokens = 0
    for i, t in enumerate(result.get("traces", []), 1):
        total_ms += t["elapsed_ms"]
        total_tokens += t["tokens"]
        print(
            f"{i:>2}. [{t['agent']:<14}] {t['action']:<16} "
            f"{t['elapsed_ms']:>7.1f}ms  {t['tokens']:>4} tok"
        )
        if t["detail"]:
            print(f"     └ {t['detail']}")
    print(f"\n反思轮次: {result.get('reflection_count', 0)}")
    print(f"合计: {total_ms:.1f}ms / {total_tokens} tokens")


async def main() -> None:
    settings = get_settings()
    configure_logging(bool(settings.verbose))
    banner(settings)
    questions = sys.argv[1:] or [DEFAULT_QUESTIONS[0]]

    for q in questions:
        print(f"\n>>> 问题: {q}", flush=True)
        start = time.perf_counter()
        result = await run(q)
        wall = (time.perf_counter() - start) * 1000
        show_trace(result)
        print("\n--- 最终回答 ---")
        print(result.get("final", "(无输出)"))
        print(f"\n[端到端 {wall:.1f}ms]")


if __name__ == "__main__":
    asyncio.run(main())
