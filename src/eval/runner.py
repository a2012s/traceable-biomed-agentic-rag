"""评估运行器：本地检索评估（无需 LLM）+ 端到端流程评估（可选）。

用法：
    python eval.py                 # 检索评估，秒级，无需 key
    python eval.py --full          # 端到端，跑真实 LLM 与反思闭环
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from ..agents.graph import run as run_graph
from ..config import get_settings
from ..deps import get_reranker, get_retriever
from .golden_set import GOLDEN_SET
from .metrics import GenerationMetrics, RetrievalMetrics, generation_metrics, retrieval_metrics


@dataclass
class EvalReport:
    mode: str
    backend: dict
    retrieval: list[tuple[str, RetrievalMetrics]] = field(default_factory=list)
    generation: list[tuple[str, GenerationMetrics]] = field(default_factory=list)

    def summary(self) -> dict:
        out = {
            "mode": self.mode,
            "backend": self.backend,
            "n_cases": len(self.retrieval),
        }
        if self.retrieval:
            out.update({
                "mean_recall": round(statistics.mean(m.recall for _, m in self.retrieval), 3),
                "mean_mrr": round(statistics.mean(m.mrr for _, m in self.retrieval), 3),
                "mean_precision": round(statistics.mean(m.precision for _, m in self.retrieval), 3),
            })
        if self.generation:
            out.update({
                "mean_citation_precision": round(
                    statistics.mean(m.citation_precision for _, m in self.generation), 3
                ),
                "mean_distractor_leakage": round(
                    statistics.mean(m.distractor_leakage for _, m in self.generation), 3
                ),
                "grounded_rate": round(
                    statistics.mean(1.0 if m.grounded else 0.0 for _, m in self.generation), 3
                ),
            })
        return out


async def evaluate(full: bool = False) -> EvalReport:
    settings = get_settings()
    # 复用运行时单例：与线上走同一套检索/重排实现，评估结果才有代表性
    retriever = get_retriever()
    reranker = get_reranker()
    report = EvalReport(
        mode="end-to-end" if full else "retrieval-only",
        backend=retriever.backend_info(),
    )

    for case in GOLDEN_SET:
        if full:
            result = await run_graph(case.question, verbose=False)
            evidence_ids = [e["doc_id"] for e in result.get("evidences", [])]
            citations = result.get("citations", [])
            report.retrieval.append((case.question, retrieval_metrics(evidence_ids, case)))
            report.generation.append((
                case.question,
                generation_metrics(citations, evidence_ids, case, bool(result.get("grounded"))),
            ))
        else:
            candidates = retriever.retrieve([case.question], settings.retrieve_top_k)
            ranked = reranker.rerank(case.question, candidates)
            ids = [e["doc_id"] for e in ranked]
            report.retrieval.append((case.question, retrieval_metrics(ids, case)))

    return report


def print_report(report: EvalReport) -> None:
    print("=" * 72)
    print(f" RAG 评估报告 · {report.mode}")
    print("=" * 72)
    print(f" 后端: {report.backend}")
    print()
    for question, metrics in report.retrieval:
        d = metrics.as_dict()
        print(f" [{question}]")
        print(f"   Recall@5={d['recall']:.3f}  MRR={d['mrr']:.3f}  Precision@5={d['precision']:.3f}")
        print(f"   命中: {d['relevant_found']}")
        if report.generation:
            gen = next((m for q, m in report.generation if q == question), None)
            if gen:
                g = gen.as_dict()
                print(
                    f"   引用准确率={g['citation_precision']:.3f}  "
                    f"干扰泄漏={g['distractor_leakage']:.3f}  grounded={g['grounded']}"
                )
                if g["hallucinated_citations"]:
                    print(f"   ⚠ 幻觉引用: {g['hallucinated_citations']}")
        print()
    print("--- 汇总 ---")
    for key, value in report.summary().items():
        print(f" {key}: {value}")
