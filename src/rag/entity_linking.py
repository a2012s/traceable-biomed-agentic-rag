"""实体链接（规则版）：解决同形异义碰撞导致的跨域证据污染。

典型例子：eGFR（估算肾小球滤过率，肾功能指标）小写后与 EGFR（表皮生长因子受体，
肺癌靶点）同形。此前达格列净 CKD 试验会混进 EGFR 肺癌问题的证据集。

这里做两件事，都不依赖网络：
1. detect_senses(question)：从语境线索判断当前问题指的是哪个义项；
2. penalty(doc, senses)：文档属于"另一义项"时返回惩罚系数（0~1），供重排使用。

与硬过滤相比，降权保留了可解释性：若确有必要，低分证据仍能被 Critic 看到。
生产接入 UMLS / 领域实体链接服务时，只需替换 detect_senses 的实现。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# 每个碰撞组：规范化 token -> 各义项的定义线索。文档文本命中哪组线索即归到该义项。
COLLISIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "egfr": {
        "kidney_function": (
            "egfr", "gfr", "estimated glomerular", "glomerular filtration", "ckd",
            "kidney", "renal", "albuminuria", "肾病", "肾功能", "肾脏", "滤过率",
        ),
        "receptor_gene": (
            "egfr-mutated", "egfr mutation", "exon 19", "l858r", "osimertinib",
            "nsclc", "non-small-cell lung", "剃", "肺癌", "突变", "奥希替尼", "受体",
        ),
    },
}

# 查询侧语境线索：优先级高于裸 token，因为 "egfr" 本身在两个义项里都出现。
_QUERY_CUES: dict[str, dict[str, tuple[str, ...]]] = {
    "egfr": {
        "kidney_function": (
            "egfr decline", "egfr slope", "estimated gfr", "renal", "kidney", "ckd",
            "albuminuria", "dialysis", "肾功能", "肾脏", "肾病", "滤过率", "尿蛋白",
        ),
        "receptor_gene": (
            "mutation", "mutated", "exon 19", "l858r", "osimertinib", "nsclc",
            "lung cancer", "腺癌", "肺癌", "突变", "靶点", "奥希替尼", "一线治疗",
        ),
    },
}


@dataclass
class EntityLinking:
    enabled: bool = True

    def detect_senses(self, question: str) -> dict[str, str]:
        """返回 {token: 命中义项}；无线索时不返回该 token（= 不做降权）。"""
        if not self.enabled:
            return {}
        lowered = question.lower()
        result: dict[str, str] = {}
        for token, cues_by_sense in _QUERY_CUES.items():
            # 只在问题确实出现该 token 时才判断
            if not re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", lowered):
                continue
            scored = {
                sense: sum(1 for cue in cues if cue in lowered)
                for sense, cues in cues_by_sense.items()
            }
            best = max(scored, key=lambda k: scored[k])
            if scored[best] > 0:
                result[token] = best
        return result

    def penalty(self, doc_text: str, senses: dict[str, str]) -> float:
        """文档命中查询义项的对立义项时返回 <1 的系数；否则 1.0。"""
        if not self.enabled or not senses:
            return 1.0
        lowered = doc_text.lower()
        for token, wanted in senses.items():
            groups = COLLISIONS.get(token, {})
            if wanted not in groups:
                continue
            others = [s for s in groups if s != wanted]
            wanted_hits = sum(1 for cue in groups[wanted] if cue in lowered)
            other_hits = sum(1 for s in others for cue in groups[s] if cue in lowered)
            if other_hits > wanted_hits and other_hits >= 2:
                return 0.35  # 强线索判定为另一义项，重排分数打三折
        return 1.0
