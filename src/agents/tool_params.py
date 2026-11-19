"""工具入参抽取：由「查询实词 ∩ 目标数据源词表」推导，不硬编码领域名词。

为什么不用固定关键词表（`["diabetes", "egfr", ...]`）：
新增数据源、换领域、加一种病，固定表都要改代码，且漏词即静默失效。
这里把抽取做成词表驱动——词表随语料/外部数据源生成，工具契约（Pydantic schema）不变。

对应 JD 第 3 条"工具封装与调度"：参数从查询里推，而不是从代码里写死。
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from ..data.corpus import (
    ABBREVIATIONS,
    ALL_DOCUMENTS,
    CLINICAL,
    KG_TRIPLES,
    LITERATURE,
    REPORTS,
    resolve_senses,
)
from ..utils.text import content_tokens


def _has_abbr(abbr: str, text: str) -> bool:
    """按词边界判断缩写是否出现。

    子串匹配会把 "ms" 命中 "systems"、把 "dmp" 命中 "damp"，
    导致工具参数被无关缩写带偏。
    """
    return re.search(rf"(?<![a-z0-9]){re.escape(abbr)}(?![a-z0-9])", text) is not None


def _vocabulary(docs: Iterable) -> Counter[str]:
    """数据源词表：词 -> 出现该词的文档数（用作区分度权重）。"""
    vocab: Counter[str] = Counter()
    for doc in docs:
        vocab.update(content_tokens(f"{doc.title} {doc.text}"))
    return vocab


CLINICAL_VOCAB = _vocabulary(CLINICAL)
LITERATURE_VOCAB = _vocabulary(LITERATURE)
KG_VOCAB = _vocabulary(KG_TRIPLES)
REPORT_VOCAB = _vocabulary(REPORTS)
ALL_VOCAB = _vocabulary(ALL_DOCUMENTS)



def query_terms(question: str) -> set[str]:
    """查询实词 + 缩写展开词。

    只取原词不够：问 "MS" 时文献写作 "multiple sclerosis"，
    不展开就永远匹配不到，工具会返回空并触发无意义重试。
    """
    terms = content_tokens(question)
    lowered = question.lower()
    for abbr, expansions in ABBREVIATIONS.items():
        if _has_abbr(abbr, lowered):
            for expansion in expansions:
                terms |= content_tokens(expansion)
    return terms


def _in_corpus(phrase: str) -> bool:
    """语料词表里是否出现过该短语的全部词。

    只在语料能表达这个义项时才展开：本语料里从没写过 "epidermal growth factor receptor"，
    强行展开会让 "receptor" 命中 "GLP-1 receptor agonists"，把糖尿病文献拉进肺癌问题的引用。
    真实系统换用大语料时这条通常恒为真，不会限制泛化。
    """
    return all(term in ALL_VOCAB for term in phrase.lower().split())


def expanded_query(question: str) -> str:
    """打分用的查询：把**可消歧**的缩写就地替换成语境相符的全称。

    三个关键决策，都是踩过坑之后定的：
    1. 不用重写器的输出打分。重写器为放宽召回会追加
       "systematic review / randomized controlled trial" 这类通用词，
       拿它算相关度，任何带方法学字样的文档都命中，语料外主题也会被"证据充分"。
    2. 替换而不是追加。"MS 的诊断标准" 追加成 "MS ... multiple sclerosis" 后，
       裸 "ms" 仍会命中 "LC-MS/MS"，把质谱论文拉进引用；替换成
       "multiple sclerosis ..." 才能彻底切断歧义。
    3. 只在语料能表达该义项时替换（`_in_corpus`）。语料里从没写过
       "epidermal growth factor receptor"，硬展开会让 "receptor" 撞上
       "GLP-1 receptor agonists"，糖尿病文献就混进肺癌问题的证据。
    """
    text = question
    for abbr, chosen in resolve_senses(question).items():
        if len(chosen) != 1 or not _in_corpus(chosen[0]):
            continue  # 消歧不了就别猜，保留原缩写
        text = re.sub(rf"(?<![a-z0-9]){re.escape(abbr)}(?![a-z0-9])", chosen[0], text, flags=re.IGNORECASE)
    return text


def _pick(question: str, vocab: Counter[str], fallback: str) -> str:
    """选一个能区分该数据源的查询词：先按词表覆盖度，再按词长（信息量）。"""
    hits = query_terms(question) & set(vocab)
    if not hits:
        return fallback
    return max(hits, key=lambda term: (vocab[term], len(term)))


def _fallback(question: str) -> str:
    english = sorted((t for t in content_tokens(question) if t.isascii()), key=len, reverse=True)
    return english[0] if english else "condition"


def pubmed_params(question: str, queries: list[str] | None = None) -> dict:
    """文献检索关键词：取与文献词表重合度最高的 5 个实词。"""
    text = " ".join([question, *(queries or [])])
    terms = query_terms(text) & set(LITERATURE_VOCAB)
    ranked = sorted(terms, key=lambda t: (LITERATURE_VOCAB[t], len(t)), reverse=True)[:5]
    return {"keywords": ranked or list(content_tokens(question))[:3] or ["evidence"], "top_k": 5}


def clinical_params(question: str, queries: list[str] | None = None) -> dict:
    """临床试验适应证：取试验语料里出现过的疾病词，避免传一个查不到的概念。"""
    text = " ".join([question, *(queries or [])])
    return {"indication": _pick(text, CLINICAL_VOCAB, _fallback(text))}


def umls_params(question: str, queries: list[str] | None = None) -> dict:
    """概念扩展：优先用查询里出现的缩写（演示消歧），否则取图谱词表命中的概念。"""
    text = " ".join([question, *(queries or [])])
    lowered = text.lower()
    for abbr in ABBREVIATIONS:
        if _has_abbr(abbr, lowered):
            return {"concept": abbr}
    return {"concept": _pick(text, KG_VOCAB, _fallback(text))}


def report_params(question: str, queries: list[str] | None = None) -> dict:
    """报告检索键：部位 / 检查类型 / 分子标志，取与报告语料重合的实词。"""
    text = " ".join([question, *(queries or [])])
    terms = query_terms(text) & set(REPORT_VOCAB)
    ranked = sorted(terms, key=lambda t: (REPORT_VOCAB[t], len(t)), reverse=True)[:5]
    return {"keywords": ranked or list(content_tokens(question))[:3] or ["report"], "top_k": 5}
