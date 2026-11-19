"""内置生物医学样例语料库。

⚠️ 重要声明（开源诚信）：
本文件中所有 PMID / NCT 编号均为【教学演示用的占位符】，其标题与正文为虚构内容，
与 PubMed / ClinicalTrials.gov 上该编号对应的真实记录【无关】。
禁止将本语料及其生成的回答用于真实临床、科研或决策场景。
生产使用时请把 build_index() 替换为真实数据源（PubMed API / 内部数仓）加载。
详见仓库根目录 DISCLAIMER.md。

四类数据源对应四个专职 agent：
- literature: 期刊文献（PMC/PMID）
- clinical:   临床试验（ClinicalTrials.gov NCT 编号）
- report:     非结构化病理/影像报告
- kg:         知识图谱三元组（UMLS 风格概念关系）

真实项目把 build_index() 换成从 PubMed API / 内部数仓拉取即可，接口不变。
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass
class Document:
    doc_id: str
    source: str
    title: str
    text: str
    url: str
    study_type: str = "narrative"
    evidence_level: str = "4"
    year: int = 2023

    def as_dict(self) -> dict:
        return asdict(self)


LITERATURE: list[Document] = [
    Document(
        doc_id="PMID:37423117",
        source="literature",
        title="SGLT2 inhibitors and cardiovascular outcomes in type 2 diabetes: a meta-analysis",
        text=("SGLT2 inhibitors significantly reduced major adverse cardiovascular events (MACE) "
              "by 14% (HR 0.86, 95% CI 0.80-0.93) in patients with type 2 diabetes. "
              "Empagliflozin and dapagliflozin showed consistent benefit. "
              "The benefit was independent of glycemic control and was mediated partly by "
              "reduced heart failure hospitalization (HR 0.71)."),
        url="https://pubmed.ncbi.nlm.nih.gov/37423117/",
        study_type="meta-analysis", evidence_level="1", year=2024,
    ),
    Document(
        doc_id="PMID:36100815",
        source="literature",
        title="Osimertinib in EGFR-mutated non-small-cell lung cancer: 5-year follow-up",
        text=("First-line osimertinib achieved a median overall survival of 38.6 months in "
              "EGFR exon 19 deletion or L858R mutant advanced NSCLC. "
              "Brain metastasis incidence was reduced compared with first-generation EGFR-TKI. "
              "Common adverse events included rash and diarrhea; interstitial lung disease occurred in 3.6%."),
        url="https://pubmed.ncbi.nlm.nih.gov/36100815/",
        study_type="randomized-controlled-trial", evidence_level="1", year=2023,
    ),
    Document(
        doc_id="PMID:39201884",
        source="literature",
        title="GLP-1 receptor agonists and renal outcomes: systematic review",
        text=("Pooled analysis of 8 trials (n=62,104) showed GLP-1 receptor agonists reduced "
              "kidney composite outcomes by 18% (HR 0.82, 95% CI 0.75-0.90). "
              "Effect was driven mainly by reduction in albuminuria progression. "
              "Semaglutide showed the most consistent renal signal."),
        url="https://pubmed.ncbi.nlm.nih.gov/39201884/",
        study_type="systematic-review", evidence_level="1", year=2024,
    ),
    Document(
        doc_id="PMID:35581204",
        source="literature",
        title="Mass spectrometry versus immunoassay for serum protein quantification",
        text=("Liquid chromatography-mass spectrometry (LC-MS/MS) offers higher analytical "
              "specificity than immunoassay for low-abundance serum proteins, but requires "
              "specialized infrastructure and trained personnel. "
              "Immunoassay remains first-line in routine clinical laboratories due to throughput."),
        url="https://pubmed.ncbi.nlm.nih.gov/35581204/",
        study_type="review", evidence_level="4", year=2022,
    ),
    Document(
        doc_id="PMID:40218776",
        source="literature",
        title="High-efficacy disease-modifying therapy in early multiple sclerosis: a randomized trial",
        text=("A randomized controlled trial of 892 patients with early relapsing-remitting multiple sclerosis "
              "compared high-efficacy therapy against moderate-efficacy therapy. "
              "The primary endpoint was time to confirmed disability worsening. "
              "High-efficacy therapy reduced disability worsening by 31% (HR 0.69, 95% CI 0.55-0.87). "
              "Adverse events were mainly infusion-related reactions."),
        url="https://pubmed.ncbi.nlm.nih.gov/40218776/",
        study_type="randomized-controlled-trial", evidence_level="1", year=2025,
    ),
    Document(
        doc_id="PMID:38112009",
        source="literature",
        title="Multiple sclerosis diagnosis: 2024 revised McDonald criteria",
        text=("The revised McDonald criteria emphasize MRI evidence of dissemination in space "
              "and time. Cerebrospinal fluid oligoclonal bands can substitute for temporal "
              "dissemination. Early diagnosis is recommended to enable disease-modifying therapy."),
        url="https://pubmed.ncbi.nlm.nih.gov/38112009/",
        study_type="guideline", evidence_level="1", year=2024,
    ),
]

CLINICAL: list[Document] = [
    Document(
        doc_id="NCT04925583",
        source="clinical",
        title="Phase 3 trial of dapagliflozin in CKD with type 2 diabetes",
        text=("Enrollment 11,404. Intervention dapagliflozin 10mg daily versus placebo. "
              "Primary outcome: composite of sustained eGFR decline, ESKD, or renal/cardiovascular death. "
              "Status: completed. Result: primary endpoint met with 39% relative risk reduction. "
              "Sponsor: AstraZeneca."),
        url="https://clinicaltrials.gov/study/NCT04925583",
        study_type="phase-3-trial", evidence_level="1", year=2023,
    ),
    Document(
        doc_id="NCT05123410",
        source="clinical",
        title="Osimertinib plus chemotherapy in advanced EGFR-mutated NSCLC",
        text=("Phase 3, n=557. Combination improved progression-free survival to 18.9 months "
              "versus 10.2 months with osimertinib alone. Status: active, not recruiting. "
              "Grade 3+ adverse events 42% vs 19%."),
        url="https://clinicaltrials.gov/study/NCT05123410",
        study_type="phase-3-trial", evidence_level="1", year=2024,
    ),
    Document(
        doc_id="NCT03877144",
        source="clinical",
        title="Semaglutide 2.4mg in obesity-related kidney disease",
        text=("Phase 2 proof-of-concept, n=296. Exploratory endpoint of albuminuria reduction. "
              "Status: recruiting. Estimated completion 2026. No efficacy results posted yet."),
        url="https://clinicaltrials.gov/study/NCT03877144",
        study_type="phase-2-trial", evidence_level="2", year=2025,
    ),
]

REPORTS: list[Document] = [
    Document(
        doc_id="RPT-2024-0912",
        source="report",
        title="Pathology report: lung adenocarcinoma resection",
        text=("Specimen: left upper lobectomy. Histology: invasive adenocarcinoma, acinar predominant. "
              "Tumor size 3.2 cm. Margins negative. Lymph nodes: 0/14 positive. "
              "Molecular: EGFR exon 19 deletion detected by NGS (allele frequency 18.4%). "
              "PD-L1 TPS 15%."),
        url="internal://pathology/RPT-2024-0912",
        study_type="case-report", evidence_level="3", year=2024,
    ),
    Document(
        doc_id="RPT-2024-1130",
        source="report",
        title="Radiology report: brain MRI, demyelination workup",
        text=("Multiple periventricular and juxtacortical T2 hyperintensities, some enhancing "
              "after gadolinium. No mass effect. Findings are consistent with demyelinating "
              "disease; correlate with CSF oligoclonal bands and clinical history."),
        url="internal://radiology/RPT-2024-1130",
        study_type="case-report", evidence_level="3", year=2024,
    ),
]

KG_TRIPLES: list[Document] = [
    Document(
        doc_id="UMLS:C0011860",
        source="kg",
        title="Diabetes Mellitus, Type 2",
        text=("UMLS:C0011860 Type 2 diabetes mellitus --has_complication--> diabetic kidney disease; "
              "--treated_by--> SGLT2 inhibitors; --treated_by--> GLP-1 receptor agonists; "
              "--has_risk_factor--> obesity."),
        url="https://uts.nlm.nih.gov/", study_type="knowledge-graph", evidence_level="4", year=2025,
    ),
    Document(
        doc_id="UMLS:C0026769",
        source="kg",
        title="Multiple Sclerosis",
        text=("UMLS:C0026769 Multiple sclerosis --diagnosed_by--> McDonald criteria; "
              "--has_finding--> oligoclonal bands; --treated_by--> disease-modifying therapy."),
        url="https://uts.nlm.nih.gov/", study_type="knowledge-graph", evidence_level="4", year=2025,
    ),
    Document(
        doc_id="UMLS:C0006825",
        source="kg",
        title="EGFR gene",
        text=("UMLS:C0006825 EGFR --mutation--> exon 19 deletion; --mutation--> L858R; "
              "--targeted_by--> osimertinib; --associated_with--> non-small-cell lung carcinoma."),
        url="https://uts.nlm.nih.gov/", study_type="knowledge-graph", evidence_level="4", year=2025,
    ),
    Document(
        doc_id="UMLS:C0027506",
        source="kg",
        title="Mass Spectrometry",
        text=("UMLS:C0027506 Mass spectrometry --is_a--> analytical chemistry method; "
              "--used_for--> serum protein quantification; --compared_with--> immunoassay."),
        url="https://uts.nlm.nih.gov/", study_type="knowledge-graph", evidence_level="4", year=2025,
    ),
]

ALL_DOCUMENTS = LITERATURE + CLINICAL + REPORTS + KG_TRIPLES

# 缩写消歧词典：医学查询没做这一步，检索必然跑偏（JD 明确要求 Query Rewriting）
ABBREVIATIONS = {
    "ms": {"multiple sclerosis": "多发性硬化", "mass spectrometry": "质谱分析"},
    "egfr": {"epidermal growth factor receptor": "表皮生长因子受体"},
    "nsclc": {"non-small-cell lung cancer": "非小细胞肺癌"},
    "t2dm": {"type 2 diabetes mellitus": "2型糖尿病"},
    "mace": {"major adverse cardiovascular events": "主要不良心血管事件"},
    "ckd": {"chronic kidney disease": "慢性肾脏病"},
    "dmp": {"disease-modifying therapy": "疾病修饰治疗"},
}

# 多义缩写的语境线索：用于从候选义项里选出与提问语境相符的那一个。
# 只做"扩展全部义项"会把两个义项的文献一起召回 —— 问诊断标准却引用了质谱方法学论文。
ABBREVIATION_SENSE_CUES: dict[str, dict[str, tuple[str, ...]]] = {
    "ms": {
        "multiple sclerosis": (
            "诊断", "标准", "复发", "缓解", "残疾", "脱髓鞘", "影像", "治疗",
            "mcdonald", "oligoclonal", "demyelination", "relapsing", "disability", "mri",
        ),
        "mass spectrometry": (
            "质谱", "定量", "检测", "蛋白", "通量", "色谱",
            "spectrometry", "quantification", "analytical", "chromatography", "lc-ms", "immunoassay",
        ),
    },
}


def resolve_senses(question: str) -> dict[str, list[str]]:
    """按语境线索选出每个多义缩写应采用义项。

    无任何线索命中时返回全部义项（宁可多召回，也不猜错方向）。
    """
    lowered = question.lower()
    resolved: dict[str, list[str]] = {}
    for abbr, senses in ABBREVIATION_SENSE_CUES.items():
        if not re.search(rf"(?<![a-z0-9]){re.escape(abbr)}(?![a-z0-9])", lowered):
            continue
        scored = {
            sense: sum(1 for cue in cues if cue in lowered)
            for sense, cues in senses.items()
        }
        best = max(scored, key=lambda k: scored[k])
        resolved[abbr] = [best] if scored[best] > 0 else list(senses)
    return resolved

# 领域关键词 -> 应分派的专家，作为 Mock 编排器的路由规则
DOMAIN_HINTS = {
    "diabetes": ["diabetes", "血糖", "糖", "sglt2", "glp-1", "dapagliflozin", "semaglutide", "糖尿病"],
    "lung": ["egfr", "nsclc", "lung", "osimertinib", "肺", "腺癌", "奥希替尼"],
    "ms": ["multiple sclerosis", "脱髓鞘", "oligoclonal", "mcdonald", "多发性硬化"],
}


def build_index() -> list[Document]:
    """生产环境替换为真实数据源加载；保持返回 Document 列表即可。"""
    return list(ALL_DOCUMENTS)
