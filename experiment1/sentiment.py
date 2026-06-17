"""Sentiment labeling for Chinese financial news.

Default strategy: local FinBERT three-class inference via
``yiyanghkust/finbert-tone-chinese`` (Positive / Neutral / Negative).
If the model files are not found, falls back to a transparent financial
lexicon scorer so the pipeline always produces reproducible results.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# BERT-based labeling (direct model inference, FP16, dynamic padding)
# ---------------------------------------------------------------------------

_bert_ctx: dict | None = None

# Model id2label: {'0': 'Neutral', '1': 'Positive', '2': 'Negative'}
_MODEL_ID2LABEL = {0: 0, 1: 1, 2: -1}  # Neutral→0, Positive→1, Negative→-1


def _get_bert_ctx() -> dict | None:
    """Lazy-load tokenizer + model on GPU with FP16."""
    global _bert_ctx
    if _bert_ctx is not None:
        return _bert_ctx
    if not config.MODEL_DIR.exists():
        return None
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(config.MODEL_DIR), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(config.MODEL_DIR), local_files_only=True, torch_dtype=torch.float16
    ).to(device)
    model.eval()
    _bert_ctx = {"tokenizer": tokenizer, "model": model, "device": device}
    return _bert_ctx


def bert_label_batch(texts: list[str], batch_size: int = 256) -> list[int]:
    """Run BERT on texts using direct model inference (GPU FP16, fast)."""
    import torch

    ctx = _get_bert_ctx()
    if ctx is None:
        raise RuntimeError("BERT model not available")
    tok = ctx["tokenizer"]
    model = ctx["model"]
    device = ctx["device"]

    clean = [str(t)[:256] if pd.notna(t) else "" for t in texts]
    total = len(clean)
    labels: list[int] = []

    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch = clean[start : start + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            preds = logits.argmax(dim=-1).cpu().tolist()
            labels.extend(_MODEL_ID2LABEL[p] for p in preds)
            done = min(start + batch_size, total)
            if done % (batch_size * 4) == 0 or done == total:
                print(f"  BERT labeling: {done}/{total} ({100*done//total}%)", flush=True)
    return labels


def bert_label_text(text: str) -> int:
    """Run BERT on a single text and return {-1, 0, 1}."""
    return bert_label_batch([text], batch_size=1)[0]


# ---------------------------------------------------------------------------
# Lexicon-based fallback
# ---------------------------------------------------------------------------

POSITIVE_WORDS = {
    "上涨", "涨停", "大涨", "走高", "反弹", "拉升", "攀升", "上扬", "领涨", "飙升",
    "突破", "创新高", "增长", "增加", "提升", "改善", "回暖", "复苏", "盈利", "利润",
    "利好", "看好", "乐观", "强劲", "稳健", "受益", "增持", "买入", "推荐", "超预期",
    "扩张", "订单", "中标", "签约", "并购", "重组", "注入", "分红", "高送转", "回购",
    "政策支持", "获批", "通过", "增长率", "净利", "同比增", "业绩预增", "扭亏",
    "提振", "活跃", "放量", "资金流入", "牛市", "景气", "领先", "升逾",
}

NEGATIVE_WORDS = {
    "下跌", "跌停", "大跌", "走低", "回落", "跳水", "杀跌", "领跌", "破位", "下滑",
    "下降", "减少", "亏损", "亏", "利空", "悲观", "承压", "风险", "警惕", "减持",
    "卖出", "下调", "低于预期", "违规", "调查", "处罚", "诉讼", "债务", "违约",
    "退市", "停牌", "质押", "爆仓", "裁员", "业绩预减", "业绩下滑", "流出",
    "萎缩", "疲软", "砸盘", "出逃", "套现", "跌逾", "低迷", "恶化", "不确定",
    "压力", "受挫", "放缓", "亏损扩大", "黑天鹅", "危机", "熊市",
}

NEGATIONS = {"不", "未", "无", "没有", "并未", "难以", "不能", "不是"}


def _tokens(segmented: str | None, title: str | None, content: str | None) -> list[str]:
    parts: list[str] = []
    if segmented:
        parts.extend(str(segmented).split("|"))
    if title:
        parts.extend(re.findall(r"[一-鿿A-Za-z0-9]+", str(title)))
    if content:
        parts.extend(re.findall(r"[一-鿿A-Za-z0-9]+", str(content)))
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
