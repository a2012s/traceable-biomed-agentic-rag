# 数据与用途声明

## 语料为虚构教学数据

本仓库 `src/data/corpus.py` 中的全部文献、临床试验、病理报告和知识图谱条目**均为虚构的教学样例**。

其中出现的 `PMID:xxxxxxx`、`NCTxxxxxxxx`、`UMLS:Cxxxxxxx` 等编号：

- 采用真实编号的**格式**，但**不指向** PubMed / ClinicalTrials.gov / UMLS 中该编号对应的真实记录
- 标题、摘要、结论、统计数据均由本 demo 编写，用于演示检索、重排、引用锚点与反思闭环
- 经 NCBI E-utilities 与 ClinicalTrials.gov API 抽样核验，编号与内容**不一致**

任何将本语料当作真实医学证据、或依据其输出做出临床/科研/投资/法律决策的行为，均属误用。

## 用途限制

本项目是**工程架构演示**，用于展示 Agentic RAG 的：

- 多智能体编排与状态机（LangGraph）
- 混合检索、查询重写、重排与向量库接入
- 引用合法性校验与反思回流
- 自动化评估体系

它**不是**医疗决策支持系统，不提供任何诊断或治疗建议。

## 生产化必须替换的组件

| 组件 | 当前（demo） | 生产环境 |
|---|---|---|
| 语料 | 虚构内置数据 | PubMed/内部数仓真实数据，保留原始 ID 与原文 |
| Embedding | TF-IDF / 哈希 | 领域模型（BioBERT / BGE-M3 / text-embedding-v3） |
| Reranker | 可解释特征打分 | Cross-encoder（bge-reranker-v2-m3） |
| Critic | 规则判定 | LLM-as-judge + 医学专家复核 |
| 评估金标 | 自建小样本 | 领域专家标注 + RAGAS |

## 许可证

代码采用 MIT License（见 `LICENSE`）。本声明与语料虚构性质不因许可证而改变。
