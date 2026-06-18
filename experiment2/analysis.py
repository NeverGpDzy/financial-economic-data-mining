"""Herd-effect indicator construction for Experiment 2."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


REQUIRED_COLUMNS = {
    "week",
    "week_start",
    "week_end",
    "week_id",
    "WeekPositive",
    "WeekNeutral",
    "WeekNegative",
    "NewsCount",
}


def load_weekly_sentiment(path=config.INPUT_WEEKLY_SENTIMENT) -> pd.DataFrame:
    """Load Experiment 1 weekly sentiment output."""
    if not path.exists():
        raise FileNotFoundError(
            f"缺少实验一周度情绪表：{path}。请先运行 `python -m experiment1.main`。"
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"周度情绪表缺少字段：{sorted(missing)}")

    date_cols = ["week", "week_start", "week_end"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    numeric_cols = ["week_id", "WeekPositive", "WeekNeutral", "WeekNegative", "NewsCount"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=date_cols + numeric_cols).copy()
    df[numeric_cols] = df[numeric_cols].astype(int)
    return df.sort_values(["week", "week_id"]).reset_index(drop=True)


def _minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or np.isclose(hi, lo):
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def _sigma_outlier_mask(df: pd.DataFrame, columns: list[str], sigma: float) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for col in columns:
        s = df[col].astype(float)
        mean = s.mean()
        std = s.std(ddof=0)
        if pd.isna(std) or np.isclose(std, 0):
            continue
        mask |= (s - mean).abs() > sigma * std
    return mask


def build_herd_index(weekly_sentiment: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build H1, H2 and H3 according to the Experiment 2 requirement.

    The extracted Word formula is retained as H2t_formula_raw. The final H2t
    is implemented as a disagreement score:
    1 - abs((positive - negative) / (positive + negative)).
    This keeps the teacher's text direction consistent: lower H2 means stronger
    one-sided consensus. H3 therefore multiplies normalized abs(H1) by
    1 - normalized(H2).
    """
    df = weekly_sentiment.copy().sort_values(["week", "week_id"]).reset_index(drop=True)
    original_rows = len(df)

    sentiment_total = df["WeekPositive"] + df["WeekNegative"]
    df["PositiveNegativeTotal"] = sentiment_total
    df = df[sentiment_total > 0].copy()
    dropped_zero_denominator = original_rows - len(df)

    df["P_t"] = df["WeekPositive"] / df["PositiveNegativeTotal"]
    df["E_P_t"] = (
        df["P_t"]
        .shift(1)
        .rolling(config.ROLLING_BASELINE_WEEKS, min_periods=1)
        .mean()
    )
    df["H1t"] = df["P_t"] - df["E_P_t"]

    df["H2t_formula_raw"] = (
        (df["WeekPositive"] - df["WeekNegative"]) / df["PositiveNegativeTotal"]
    )
    df["H2t"] = 1 - df["H2t_formula_raw"].abs()
    df["H2t"] = df["H2t"].clip(lower=0, upper=1)

    no_baseline = df["E_P_t"].isna()
    working = df[~no_baseline].copy()
    outlier_mask = _sigma_outlier_mask(working, ["H1t", "H2t"], config.OUTLIER_SIGMA)
    working["is_outlier"] = outlier_mask
    cleaned = working[~outlier_mask].copy()
    if cleaned.empty:
        raise ValueError("剔除基准缺失与异常值后没有可用样本。")

    cleaned["H1t_abs"] = cleaned["H1t"].abs()
    cleaned["H1t_formula_norm"] = _minmax(cleaned["H1t"])
    cleaned["H2t_formula_norm"] = _minmax(cleaned["H2t_formula_raw"])
    cleaned["H3t_formula_raw"] = cleaned["H1t_formula_norm"] * cleaned["H2t_formula_norm"]
    cleaned["H1t_norm"] = _minmax(cleaned["H1t_abs"])
    cleaned["H2t_norm"] = _minmax(cleaned["H2t"])
    cleaned["H2t_strength"] = 1 - cleaned["H2t_norm"]
    cleaned["H3t"] = cleaned["H1t_norm"] * cleaned["H2t_strength"]
    cleaned["trade_date"] = cleaned["week_end"]

    output_cols = [
        "trade_date",
        "week",
        "week_start",
        "week_end",
        "week_id",
        "P_t",
        "E_P_t",
        "H1t",
        "H1t_abs",
        "H1t_formula_norm",
        "H1t_norm",
        "H2t_formula_raw",
        "H2t_formula_norm",
        "H2t",
        "H2t_norm",
        "H2t_strength",
        "H3t_formula_raw",
        "H3t",
        "WeekPositive",
        "WeekNeutral",
        "WeekNegative",
        "NewsCount",
        "PositiveNegativeTotal",
    ]
    herd = cleaned[output_cols].sort_values("trade_date").reset_index(drop=True)

    audit = {
        "input_rows": int(original_rows),
        "dropped_zero_denominator": int(dropped_zero_denominator),
        "dropped_no_history_baseline": int(no_baseline.sum()),
        "outlier_removed": int(outlier_mask.sum()),
        "output_rows": int(len(herd)),
        "date_start": str(herd["trade_date"].min().date()),
        "date_end": str(herd["trade_date"].max().date()),
        "h3_min": float(herd["H3t"].min()),
        "h3_max": float(herd["H3t"].max()),
        "h3_mean": float(herd["H3t"].mean()),
        "h3_std": float(herd["H3t"].std(ddof=0)),
    }
    return herd, audit


def build_quality_checks(herd: pd.DataFrame) -> pd.DataFrame:
    """Return machine-readable quality checks for the generated indicator."""
    checks = [
        {
            "check": "关键字段无缺失",
            "result": bool(not herd[["trade_date", "H1t", "H2t", "H3t"]].isna().any().any()),
            "detail": "trade_date、H1t、H2t、H3t 均应完整。",
        },
        {
            "check": "日期单调递增",
            "result": bool(pd.to_datetime(herd["trade_date"]).is_monotonic_increasing),
            "detail": "输出时序按交易周末日期升序排列。",
        },
        {
            "check": "P_t 范围合法",
            "result": bool(herd["P_t"].between(0, 1).all()),
            "detail": "P_t 为正面新闻占正负情绪新闻比例。",
        },
        {
            "check": "H2t 范围合法",
            "result": bool(herd["H2t"].between(0, 1).all()),
            "detail": "H2t 为分歧度，越接近 0 表示越一边倒。",
        },
        {
            "check": "任务书公式H2原文列范围合法",
            "result": bool(herd["H2t_formula_raw"].between(-1, 1).all()),
            "detail": "H2t_formula_raw 保留任务书公式4的直接转写结果。",
        },
        {
            "check": "H3t 范围合法",
            "result": bool(herd["H3t"].between(0, 1).all()),
            "detail": "H3t 由 Min-Max 后的 H1 强度和 H2 反向强度相乘。",
        },
    ]
    return pd.DataFrame(checks)


def build_summary_statistics(herd: pd.DataFrame) -> pd.DataFrame:
    """Summarize the main time-series indicators."""
    rows = []
    for col in ["P_t", "H1t", "H2t_formula_raw", "H2t", "H2t_strength", "H3t_formula_raw", "H3t"]:
        s = herd[col].astype(float)
        rows.append(
            {
                "indicator": col,
                "count": int(s.count()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=0)),
                "min": float(s.min()),
                "median": float(s.median()),
                "max": float(s.max()),
            }
        )
    return pd.DataFrame(rows)


def build_top_herd_weeks(herd: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Select weeks with the highest H3 herd-effect score."""
    cols = [
        "rank",
        "trade_date",
        "P_t",
        "E_P_t",
        "H1t",
        "H2t",
        "H2t_formula_raw",
        "H2t_strength",
        "H3t_formula_raw",
        "H3t",
        "WeekPositive",
        "WeekNegative",
        "NewsCount",
    ]
    top = herd.sort_values("H3t", ascending=False).head(n).copy()
    top["rank"] = range(1, len(top) + 1)
    return top[cols]
