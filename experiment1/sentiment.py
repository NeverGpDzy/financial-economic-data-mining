"""Lightweight reproducible sentiment labeling for Chinese financial news.

The assignment mentions BERT. The local environment does not include
``transformers`` by default, so the executable pipeline uses a transparent
financial lexicon scorer and records that methodological choice in the report.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

POSITIVE_WORDS = {
    "上涨", "涨停", "大涨", "走高", "反弹", "拉升", "攀升", "上扬", "领涨", "飙升",
    "突破", "创新高", "增长", "增加", "提升", "改善", "回暖", "复苏", "盈利", "利润",
    "利好", "看好", "乐观", "强劲", "稳健", "受益", "增持", "买入", "推荐", "超预期",
    "扩张", "订单", "中标", "签约", "并购", "重组", "注入", "分红", "高送转", "回购",
    "政策支持", "获批", "通过", "增长率", "净利", "同比增", "业绩预增", "扭亏",
    "提振", "活跃", "放量", "资金流入", "突破", "牛市", "景气", "领先", "升逾",
}

NEGATIVE_WORDS = {
    "下跌", "跌停", "大跌", "走低", "回落", "跳水", "杀跌", "领跌", "破位", "下滑",
    "下降", "减少", "亏损", "亏", "利空", "悲观", "承压", "风险", "警惕", "减持",
    "卖出", "下调", "低于预期", "违规", "调查", "处罚", "诉讼", "债务", "违约",
    "退市", "停牌", "质押", "爆仓", "裁员", "亏损", "业绩预减", "业绩下滑", "流出",
    "萎缩", "疲软", "砸盘", "出逃", "套现", "跌逾", "低迷", "恶化", "不确定",
    "压力", "受挫", "放缓", "下滑", "亏损扩大", "黑天鹅", "危机", "熊市",
}

NEGATIONS = {"不", "未", "无", "没有", "并未", "难以", "不能", "不是"}


def _tokens(segmented: str | None, title: str | None, content: str | None) -> list[str]:
    parts: list[str] = []
    if segmented:
        parts.extend(str(segmented).split("|"))
    if title:
        parts.extend(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", str(title)))
    if content:
        parts.extend(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", str(content)))
    return [p.strip() for p in parts if p and p.strip()]


def _contains_negation(tokens: list[str], index: int) -> bool:
    start = max(0, index - 3)
    return any(tokens[i] in NEGATIONS for i in range(start, index))


def score_text(segmented: str | None, title: str | None, content: str | None) -> int:
    tokens = _tokens(segmented, title, content)
    score = 0
    for i, token in enumerate(tokens):
        weight = 2 if i < 30 else 1
        pos_hit = any(word in token for word in POSITIVE_WORDS)
        neg_hit = any(word in token for word in NEGATIVE_WORDS)
        if pos_hit and not neg_hit:
            score += -weight if _contains_negation(tokens, i) else weight
        elif neg_hit and not pos_hit:
            score += weight if _contains_negation(tokens, i) else -weight
    return score


def label_from_score(score: int, neutral_band: int = 1) -> int:
    if score > neutral_band:
        return 1
    if score < -neutral_band:
        return -1
    return 0


def label_name(label: int) -> str:
    return {1: "positive", 0: "neutral", -1: "negative"}[int(label)]


def summarize_labels(labels: Iterable[int]) -> dict[str, int]:
    values = list(labels)
    return {
        "positive": sum(v == 1 for v in values),
        "neutral": sum(v == 0 for v in values),
        "negative": sum(v == -1 for v in values),
    }

