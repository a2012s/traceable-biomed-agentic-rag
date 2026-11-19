"""金标集：每条样本标注应召回文档与干扰文档。

设计原则（对应 JD 第 4 条"数据驱动的迭代"）：
- relevant: 回答问题必需的最小文档集合（下限，不是越多越好）
- distractors: 主题相近但无关，专门检验重排是否会被语义近似带偏
- 用 doc_id 而非文本做判定，避免评估本身依赖 LLM 造成循环论证
- 覆盖四种数据域、缩写消歧、跨域干扰三类难点，单条样本不足以代表整体

注意：样本量 12 条仍属小样本，指标用于回归对比（改检索前后是否退化），
不应解读为生产环境的绝对精度。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GoldenCase:
    question: str
    relevant: frozenset[str]
    distractors: frozenset[str] = field(default_factory=frozenset)
    note: str = ""


GOLDEN_SET: list[GoldenCase] = [
    GoldenCase(
        question="SGLT2 inhibitors for type 2 diabetes and kidney outcomes?",
        relevant=frozenset({"PMID:37423117", "NCT04925583", "UMLS:C0011860"}),
        distractors=frozenset({"PMID:39201884", "PMID:36100815"}),
        note="GLP-1 肾脏文献与奥希替尼肺癌是近似干扰，检验主题隔离",
    ),
    GoldenCase(
        question="EGFR 突变非小细胞肺癌的一线治疗方案是什么？",
        relevant=frozenset({"PMID:36100815", "NCT05123410", "UMLS:C0006825", "RPT-2024-0912"}),
        distractors=frozenset({"PMID:37423117", "UMLS:C0011860", "NCT04925583"}),
        note="需同时命中文献、试验、报告、KG 四个数据域；达格列净 CKD 试验是 eGFR/EGFR 词形干扰",
    ),
    GoldenCase(
        question="MS 的诊断标准是什么？",
        relevant=frozenset({"PMID:38112009", "UMLS:C0026769", "RPT-2024-1130"}),
        distractors=frozenset({"PMID:35581204", "UMLS:C0027506"}),
        note="MS 缩写消歧：质谱文献是刻意设置的歧义干扰",
    ),
    GoldenCase(
        question="mass spectrometry 质谱分析在血清蛋白定量中的应用",
        relevant=frozenset({"PMID:35581204", "UMLS:C0027506"}),
        distractors=frozenset({"PMID:38112009", "UMLS:C0026769"}),
        note="冷门问题，检验低证据时不硬编",
    ),
    GoldenCase(
        question="dapagliflozin 在慢性肾脏病的肾脏终点试验结果",
        relevant=frozenset({"NCT04925583", "PMID:37423117"}),
        distractors=frozenset({"PMID:36100815", "RPT-2024-0912"}),
        note="具体药物名 + 终点指标，检验试验库与文献库的精确匹配",
    ),
    GoldenCase(
        question="semaglutide 对肥胖相关肾病的白蛋白尿影响",
        relevant=frozenset({"NCT03877144", "PMID:39201884"}),
        distractors=frozenset({"PMID:36100815", "RPT-2024-1130"}),
        note="早期试验尚无结果，检验是否会拿其他药物证据顶替",
    ),
    GoldenCase(
        question="oligoclonal bands 在脑脊液中的诊断价值",
        relevant=frozenset({"PMID:38112009", "UMLS:C0026769", "RPT-2024-1130"}),
        distractors=frozenset({"PMID:35581204", "NCT05123410"}),
        note="同一指标横跨指南、图谱、影像报告三域",
    ),
    GoldenCase(
        question="lung adenocarcinoma EGFR exon 19 deletion 术后病理特征",
        relevant=frozenset({"RPT-2024-0912", "UMLS:C0006825"}),
        distractors=frozenset({"PMID:37423117", "NCT04925583"}),
        note="病理报告域必答，检验非文献数据源不被忽略",
    ),
    GoldenCase(
        question="brain MRI demyelination 的影像学表现",
        relevant=frozenset({"RPT-2024-1130"}),
        distractors=frozenset({"RPT-2024-0912", "UMLS:C0026769"}),
        note="标注只收真正描述影像表现的报告；KG 概念条文不含 MRI 描述，列干扰而非正例",
    ),
    GoldenCase(
        question="high-efficacy therapy 对早期 multiple sclerosis 残疾进展的影响",
        relevant=frozenset({"PMID:40218776"}),
        distractors=frozenset({"PMID:38112009", "NCT03877144"}),
        note="单文档必答，反映真实场景中最常见的一问一证",
    ),
    GoldenCase(
        question="GLP-1 receptor agonist 的肾脏获益证据",
        relevant=frozenset({"PMID:39201884", "UMLS:C0011860"}),
        distractors=frozenset({"PMID:37423117", "NCT04925583"}),
        note="与 SGLT2 问题构成近邻干扰对，最容易互相串味",
    ),
    GoldenCase(
        question="mass spectrometry 与 immunoassay 的检测通量对比",
        relevant=frozenset({"PMID:35581204", "UMLS:C0027506"}),
        distractors=frozenset({"PMID:38112009", "UMLS:C0026769"}),
        note="与方法学问题的对照样本，检验 MS 消歧方向是否反了过来",
    ),
]


def by_question(question: str) -> GoldenCase | None:
    q = question.strip().lower()
    for case in GOLDEN_SET:
        if case.question.lower() == q:
            return case
    return None
