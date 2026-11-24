# 贡献指南

这个项目的定位是**可读的 Agentic RAG 参考实现**：能跑、能改、能讲清为什么这么设计。
欢迎提 issue 与 PR，尤其是"某个设计有更好的取舍"这类讨论。

## 本地开发

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[real-vector]" ruff

python -m unittest discover -s tests -t .    # 76 项测试，强制离线（tests/__init__.py 里设 LLM_PROVIDER=mock）
ruff check .
python eval.py                               # 检索评估，秒级
LLM_PROVIDER=mock python eval.py --full      # 端到端评估，免 key
```

改动检索或提示词后，请**同时贴出 `eval.py` 前后的指标**。没有指标的检索改动无法判断是不是退化。

## 提交前必做

```bash
cp scripts/pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
```

钩子会扫描暂存区，发现 `.env` 或 `sk-` 形态的密钥直接拒绝提交。
另请注意：**不要用 `git add .` / `git add -A`**，显式列出文件，避免把 `.env` 一起带进去。

## 代码约定

| 方面 | 约定 |
|---|---|
| 类型 | 公共函数带类型标注；`from __future__ import annotations` 统一加 |
| 风格 | `ruff check` 必须通过（配置在 `pyproject.toml`） |
| 注释 | 注释解释**为什么**，不复述代码；关键取舍写清"不这么做会怎样" |
| 异常 | 区分可恢复（重试）与不可恢复（立即失败）两类，禁止裸 `except` |
| 日志 | 库代码用 `logging`，不用 `print`（`run.py` 的 CLI 输出除外） |
| 领域绑定 | 新增领域能力放进 `corpus/` `prompts/` `tools/`，不要散落到编排层 |

## 新增一个数据域 agent

1. `src/data/corpus.py` 加数据源与文档；
2. `src/tools/biomedical.py` 加一个 `ToolSpec`（schema + handler + 描述）；
3. `src/agents/tool_params.py` 加参数抽取函数（词表驱动，禁止硬编码关键词表）；
4. `src/agents/nodes.py` 用 `make_domain_agent(...)` 注册；
5. `src/agents/nodes.py::AGENT_NAMES` 与 `graph.py` 的节点注册会自动带上它；
6. `src/eval/golden_set.py` 补 2~3 条金标样本，跑 `eval.py` 确认未退化。

## 新增一个图节点

新增节点除了改 `graph.py`，还有三处必须同步，否则线上会"跑了但看不见"：

1. `web/server.py::AGENT_LABELS` 与 `web/index.html::LABELS` 各加一条中文名；
2. 若节点在流水线里是必经步骤，`web/index.html::PIPELINE` 也要加（HITL 的 `human_review` 因为是
   条件节点，采用"出现才插入"的方式，可参考现有实现）；
3. `tests/` 补一个直接跑该节点的测试，不要只靠 `eval.py` 端到端覆盖——端到端看不出停在哪一步。

## 提交信息

Conventional Commits 风格，一句话说清动机：

```
feat(retrieval): add cross-domain fusion before critic
fix(state): dedupe evidences instead of resetting on reflection
docs(readme): record ablation numbers for wide vs narrow recall
```

## PR 检查清单

- [ ] `ruff check .` 通过
- [ ] `python -m unittest discover -s tests -t .` 全绿
- [ ] `python eval.py` 指标未退化（或说明了退化原因与收益）
- [ ] 没有提交任何密钥、`.env`、`.qdrant_data/`
- [ ] 新行为有对应测试；新取舍在 README「设计决策」里有依据
