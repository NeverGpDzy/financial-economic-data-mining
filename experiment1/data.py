"""Data loading, cleaning, and persistence helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from . import config
from .sentiment import bert_label_batch, label_from_score, label_name, score_text


def load_news() -> pd.DataFrame:
    df = pd.read_excel(config.NEWS_FILE)
    df.columns = [str(c).strip() for c in df.columns]
    df["date_time"] = pd.to_datetime(df["Time"], errors="coerce")
    return df


def load_hs300_daily() -> pd.DataFrame:
    raw = pd.read_csv(config.HS300_FILE, sep="\t", encoding="gbk", skiprows=2)
    raw.columns = [str(c).strip() for c in raw.columns]
    keep = raw.rename(
        columns={"时间": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
    )
    keep = keep[["date", "open", "high", "low", "close", "volume"]].copy()
    keep["date"] = pd.to_datetime(keep["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        keep[col] = pd.to_numeric(keep[col], errors="coerce")
    keep = keep.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return keep


def make_trading_calendar(hs300: pd.DataFrame) -> pd.DataFrame:
    cal = hs300[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
    cal["trade_index"] = range(len(cal))
    cal["week_id"] = cal["trade_index"] // config.TRADING_DAYS_PER_WEEK
    week_bounds = cal.groupby("week_id")["date"].agg(week_start="min", week_end="max").reset_index()
    return cal.merge(week_bounds, on="week_id", how="left")


def clean_and_label_news(news: pd.DataFrame, calendar: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    start = pd.Timestamp(config.START_DATE)
    end = pd.Timestamp(config.END_DATE) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    before_rows = len(news)
    df = news.dropna(subset=["date_time", "Content"]).copy()
    df = df[(df["date_time"] >= start) & (df["date_time"] <= end)].copy()
    df["trade_date"] = df["date_time"].dt.normalize()
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["Content"]).copy()
    after_dedup = len(df)
    cal = calendar[["date", "week_id", "week_start", "week_end"]].rename(columns={"date": "trade_date"})
    df = df.merge(cal, on="trade_date", how="inner")

    # Prefer BERT batch labeling; fall back to lexicon if model unavailable
    use_bert = config.MODEL_DIR.exists()
    labeling_method = "lexicon"
    if use_bert:
        try:
            texts = (df["Title"].fillna("") + " " + df["Content"].fillna("")).tolist()
            df["label"] = bert_label_batch(texts, batch_size=config.BERT_BATCH_SIZE)
            labeling_method = "bert"
        except Exception:
            use_bert = False

    if not use_bert:
        df["sentiment_score"] = [
            score_text(seg, title, content)
            for seg, title, content in zip(df.get("SegContent"), df.get("Title"), df.get("Content"))
        ]
        df["label"] = df["sentiment_score"].map(label_from_score)

    df["label_name"] = df["label"].map(label_name)
    df = df.sort_values(["date_time", "NewsID"]).reset_index(drop=True)
    audit = {
        "raw_news_rows": before_rows,
        "rows_after_date_and_non_null_filter": before_dedup,
        "duplicate_content_removed": before_dedup - after_dedup,
        "labeled_trading_day_rows": len(df),
        "non_trading_or_out_of_calendar_removed": after_dedup - len(df),
        "date_start": str(df["trade_date"].min().date()) if not df.empty else "",
        "date_end": str(df["trade_date"].max().date()) if not df.empty else "",
        "labeling_method": labeling_method,
    }
    return df, audit


def build_weekly_sentiment(labeled: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        labeled.groupby(["week_id", "week_start", "week_end"])["label"]
        .agg(
            WeekPositive=lambda x: int((x == 1).sum()),
            WeekNeutral=lambda x: int((x == 0).sum()),
            WeekNegative=lambda x: int((x == -1).sum()),
            NewsCount="count",
        )
        .reset_index()
    )
    denom = weekly["WeekPositive"] + weekly["WeekNegative"]
    weekly["P_t"] = (weekly["WeekPositive"] / denom).where(denom.ne(0), 0.5)
    weekly["week"] = weekly["week_end"]
    columns = ["week", "week_start", "week_end", "week_id", "WeekPositive", "WeekNeutral", "WeekNegative", "NewsCount", "P_t"]
    return weekly[columns].sort_values("week").reset_index(drop=True)


def build_hs300_weekly(hs300: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    df = hs300.merge(calendar[["date", "week_id", "week_start", "week_end"]], on="date", how="inner")
    weekly = (
        df.groupby(["week_id", "week_start", "week_end"])
        .agg(open=("open", "first"), close=("close", "last"), high=("high", "max"), low=("low", "min"), volume=("volume", "sum"))
        .reset_index()
    )
    weekly["return"] = weekly["close"].pct_change()
    weekly["week"] = weekly["week_end"]
    return weekly[["week", "week_start", "week_end", "week_id", "open", "high", "low", "close", "volume", "return"]]


def save_outputs(
    labeled: pd.DataFrame,
    weekly_sentiment: pd.DataFrame,
    weekly_herd: pd.DataFrame,
    hs300_weekly: pd.DataFrame,
    modeling: pd.DataFrame,
) -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_cols = [c for c in ["NewsID", "Title", "date_time", "trade_date", "week_id", "sentiment_score", "label", "label_name"] if c in labeled.columns]
    labeled_sample = labeled[sample_cols].head(1000)
    labeled_sample.to_csv(config.OUTPUT_DIR / "sentiment_labeled_sample.csv", index=False, encoding="utf-8-sig")
    weekly_sentiment.to_csv(config.OUTPUT_DIR / "weekly_sentiment.csv", index=False, encoding="utf-8-sig")
    weekly_herd.to_csv(config.OUTPUT_DIR / "weekly_herd_index.csv", index=False, encoding="utf-8-sig")
    hs300_weekly.to_csv(config.OUTPUT_DIR / "hs300_weekly_return.csv", index=False, encoding="utf-8-sig")
    modeling.to_csv(config.OUTPUT_DIR / "modeling_dataset.csv", index=False, encoding="utf-8-sig")
    with sqlite3.connect(config.DB_FILE) as conn:
        labeled_sample.to_sql("sentiment_labeled_sample", conn, if_exists="replace", index=False)
        weekly_sentiment.to_sql("weekly_sentiment", conn, if_exists="replace", index=False)
        weekly_herd.to_sql("weekly_herd_index", conn, if_exists="replace", index=False)
        hs300_weekly.to_sql("hs300_weekly_return", conn, if_exists="replace", index=False)
        modeling.to_sql("modeling_dataset", conn, if_exists="replace", index=False)


def write_data_description(audit: dict, path: Path | None = None) -> None:
    path = path or (config.DATA_DIR / "数据说明.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# 实验一配套数据说明

## 原始资料

- `original/金融数据挖掘实验指导书 20260528.docx`：老师发布的实验指导书原件。
- `original/实验配套数据.zip`：老师发布的实验配套数据压缩包。
- `raw/新闻数据.xls`：金融新闻文本数据，字段包括 `ID`、`NewsID`、`SegContent`、`Title`、`Content`、`Time`。
- `raw/沪深300日价格指数.xls`：沪深300日价格数据。该文件实际为 GBK 编码的制表符文本，扩展名为 `.xls`。

## 本次处理口径

- 新闻筛选区间：{config.START_DATE} 至 {config.END_DATE}。
- 周度统计单位：严格按沪深300交易日历每 {config.TRADING_DAYS_PER_WEEK} 个交易日分为一组，而不是简单日历周。
- 清洗步骤：删除时间或正文为空的数据，按正文去重，剔除非交易日新闻。
- 情绪标注：使用 `yiyanghkust/finbert-tone-chinese` 模型进行 BERT 三分类推理（正面/中性/负面）；若模型文件不可用则回退到金融情绪词典打分器。

## 清洗摘要

- 原始新闻行数：{audit.get('raw_news_rows')}
- 日期与非空过滤后行数：{audit.get('rows_after_date_and_non_null_filter')}
- 最终进入交易日样本行数：{audit.get('labeled_trading_day_rows')}
- 样本日期范围：{audit.get('date_start')} 至 {audit.get('date_end')}

## 主要输出

- `outputs/experiment1/weekly_sentiment.csv`：实验一周度情绪表。
- `outputs/experiment1/weekly_herd_index.csv`：实验二羊群效应指标表。
- `outputs/experiment1/modeling_dataset.csv`：实验三建模对齐数据。
- `outputs/experiment1/experiment1.db`：SQLite 数据库，包含周度情绪、羊群指标、沪深300周收益和建模数据表。
"""
    path.write_text(text, encoding="utf-8")
