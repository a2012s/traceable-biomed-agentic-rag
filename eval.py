#!/usr/bin/env python3
"""RAG 自动化评估入口。"""
from __future__ import annotations

import argparse
import asyncio

from src.eval.runner import evaluate, print_report
from src.logging_setup import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG 评估")
    parser.add_argument("--full", action="store_true", help="跑端到端流程（含 LLM 与反思）")
    args = parser.parse_args()
    configure_logging(verbose=False)
    report = asyncio.run(evaluate(full=args.full))
    print_report(report)


if __name__ == "__main__":
    main()
