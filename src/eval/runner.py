"""评估运行器：本地检索评估（无需 LLM）+ 端到端流程评估（可选）。

用法：
    python eval.py                 # 检索评估，秒级，无需 key
    python eval.py --full          # 端到端，跑真实 LLM 与反思闭环
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..agents.graph import run as run_graph
from ..config import get_settings
from ..deps import get_reranker, get_retriever
from .golden_set import GOLDEN_SET
from .metrics import (
    GenerationMetrics,
    RetrievalMetrics,
    bootstrap_ci,
    generation_metrics,
    retrieval_metrics,
)


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
            recall = [m.recall for _, m in self.retrieval]
            mrr = [m.mrr for _, m in self.retrieval]
            precision = [m.precision for _, m in self.retrieval]
            out.update({
                "mean_recall": round(statistics.mean(recall), 3),
                "recall_ci95": bootstrap_ci(recall),
                "mean_mrr": round(statistics.mean(mrr), 3),
                "mrr_ci95": bootstrap_ci(mrr),
                "mean_precision": round(statistics.mean(precision), 3),
                "precision_ci95": bootstrap_ci(precision),
            })
        if self.generation:
            cp = [m.citation_precision for _, m in self.generation]
            out.update({
                "mean_citation_precision": round(statistics.mean(cp), 3),
                "citation_precision_ci95": bootstrap_ci(cp),
                "mean_distractor_leakage": round(
                    statistics.mean(m.distractor_leakage for _, m in self.generation), 3
                ),
                "grounded_rate": round(
                    statistics.mean(1.0 if m.grounded else 0.0 for _, m in self.generation), 3
                ),
            })
        return out

    def by_split(self) -> dict:
        """按 dev/test 分层报告：test 不参与调参，只看最终泛化。"""
        by_q = {q: i for i, (q, _) in enumerate(self.retrieval)}
        result = {}
        for split in ("dev", "test"):
            indices = [
                by_q[c.question] for c in GOLDEN_SET if c.split == split and c.question in by_q
            ]
            if not indices:
                continue
            result[split] = {
                "n": len(indices),
                "recall": round(statistics.mean(self.retrieval[i][1].recall for i in indices), 3),
                "mrr": round(statistics.mean(self.retrieval[i][1].mrr for i in indices), 3),
                "precision": round(statistics.mean(self.retrieval[i][1].precision for i in indices), 3),
            }
        return result

    def write_cases(self, out_dir: str | Path | None = None) -> Path:
        """落逐样本明细：指标掉了能定位是哪几条、哪一路召回出了问题。"""
        root = Path(out_dir) if out_dir else Path("eval_runs")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = root / stamp
        target.mkdir(parents=True, exist_ok=True)
        by_q_gen = dict(self.generation)
        by_q_case = {c.question: c for c in GOLDEN_SET}
        with (target / "cases.jsonl").open("w", encoding="utf-8") as fh:
            for question, metrics in self.retrieval:
                case = by_q_case.get(question)
                gen = by_q_gen.get(question)
                row = {
                    "question": question,
                    "split": case.split if case else "unknown",
                    "note": case.note if case else "",
                    "relevant": sorted(case.relevant) if case else [],
                    "distractors": sorted(case.distractors) if case else [],
                    "retrieval": metrics.as_dict(),
                    "generation": gen.as_dict() if gen else None,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        (target / "summary.json").write_text(
            json.dumps({"summary": self.summary(), "by_split": self.by_split()},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target


async def evaluate(full: bool = False, save_cases: bool = True) -> EvalReport:
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

    if save_cases:
        target = report.write_cases()
        print(f"[eval] 逐样本明细已写入 {target}")
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
    print("--- 分层（dev 调参 / test 仅报告）---")
    for split, values in report.by_split().items():
        print(f" {split}: {values}")
    print(" 注：n=12 的置信区间很宽，只用于标注不确定性，不构成显著性结论。")
