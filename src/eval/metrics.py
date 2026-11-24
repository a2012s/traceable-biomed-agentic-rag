"""RAG 评估指标：检索侧 + 生成侧，全部确定性可复现（不靠 LLM 打分）。

检索指标（IR 标准）：
- Recall@k     : 相关文档被召回的比例 —— 衡量"漏没漏"
- MRR          : 第一条相关文档排名的倒数 —— 衡量"排得靠不靠前"
- Precision@k  : 前 k 条中相关的比例 —— 衡量"噪不噪"

生成指标（防幻觉，来自 RAGAS 的忠实度思想但用规则实现）：
- 引用准确率   : 生成引用中存在于检索证据的比例 —— 防"编造引用"
- 干扰命中率   : 干扰文档进入证据的比例 —— 越低越好，检验主题隔离
- 证据充分率   : Critic 判定 grounded 的比例 —— 衡量反思闭环是否生效
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .golden_set import GoldenCase


def bootstrap_ci(
    values: list[float], n: int = 2000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    """有放回重采样给出均值的 (1-alpha) 置信区间。

    固定 seed 保证同一批结果每次得到相同区间——评估报告必须可复现，
    否则「区间变窄了」可能只是随机数换了一次。

    注意 n=12 时区间会非常宽，这是诚实的：小样本上 0.910 vs 0.972 的差异
    完全可能是噪声。不要拿它当显著性检验，只用它标注点估计的不确定性。
    """
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    k = len(values)
    means = []
    for _ in range(n):
        sample = [values[rng.randrange(k)] for _ in range(k)]
        means.append(sum(sample) / k)
    means.sort()
    lo = means[int(len(means) * alpha / 2)]
    hi = means[min(len(means) - 1, int(len(means) * (1 - alpha / 2)))]
    return round(lo, 3), round(hi, 3)


@dataclass
class RetrievalMetrics:
    recall: float
    mrr: float
    precision: float
    relevant_found: list[str]

    def as_dict(self) -> dict:
        return {
            "recall": round(self.recall, 3),
            "mrr": round(self.mrr, 3),
            "precision": round(self.precision, 3),
            "relevant_found": self.relevant_found,
        }


def retrieval_metrics(retrieved_ids: list[str], case: GoldenCase, k: int = 5) -> RetrievalMetrics:
    top = retrieved_ids[:k]
    relevant = case.relevant
    hits = [doc_id in relevant for doc_id in top]

    recall = len(set(top) & relevant) / len(relevant) if relevant else 0.0
    mrr = 0.0
    for rank, doc_id in enumerate(top, start=1):
        if doc_id in relevant:
            mrr = 1.0 / rank
            break
    precision = sum(hits) / len(top) if top else 0.0
    return RetrievalMetrics(
        recall=recall,
        mrr=mrr,
        precision=precision,
        relevant_found=[d for d in top if d in relevant],
    )


@dataclass
class GenerationMetrics:
    citation_precision: float
    distractor_leakage: float
    grounded: bool
    hallucinated_citations: list[str]

    def as_dict(self) -> dict:
        return {
            "citation_precision": round(self.citation_precision, 3),
            "distractor_leakage": round(self.distractor_leakage, 3),
            "grounded": self.grounded,
            "hallucinated_citations": self.hallucinated_citations,
        }


def generation_metrics(
    citation_ids: list[str],
    evidence_ids: list[str],
    case: GoldenCase,
    grounded: bool,
) -> GenerationMetrics:
    """两件事分开衡量：
    - hallucinated_citations: 引用了证据里不存在的 ID（生成侧自检，防编造）
    - citation_precision:     引用中真正属于金标相关文档的比例（对齐答案与标准答案）

    早期版本把 citation_precision 定义成"引用是否在证据中"，而生成节点已经过滤过
    非法引用，指标恒为 1.0 —— 等于没测。改为对齐金标集才有区分度。
    """
    valid = set(evidence_ids)
    hallucinated = [c for c in citation_ids if c not in valid]
    hits = sum(1 for c in citation_ids if c in case.relevant)
    precision = hits / len(citation_ids) if citation_ids else 0.0
    leakage = len(set(evidence_ids) & case.distractors) / len(case.distractors) if case.distractors else 0.0
    return GenerationMetrics(
        citation_precision=precision,
        distractor_leakage=leakage,
        grounded=grounded,
        hallucinated_citations=hallucinated,
    )
