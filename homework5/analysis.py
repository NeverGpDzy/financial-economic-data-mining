"""CAPM alpha and persistence analysis for Homework 5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config


@dataclass(frozen=True)
class PeriodSpec:
    label: str
    start: str
    end: str


TRAIN_PERIOD = PeriodSpec("2019-2021", config.TRAIN_START, config.TRAIN_END)
PRIMARY_TEST_PERIOD = PeriodSpec(
    config.PRIMARY_TEST_LABEL, config.PRIMARY_TEST_START, config.PRIMARY_TEST_END
)
ROBUST_TEST_PERIOD = PeriodSpec(
    config.ROBUST_TEST_LABEL, config.ROBUST_TEST_START, config.ROBUST_TEST_END
)


def _period_filter(df: pd.DataFrame, date_col: str, period: PeriodSpec) -> pd.DataFrame:
    start = pd.Timestamp(period.start)
    end = pd.Timestamp(period.end)
    return df[(df[date_col] >= start) & (df[date_col] <= end)].copy()


def capm_alpha_by_stock(
    stock_returns: pd.DataFrame,
    market_returns: pd.DataFrame,
    period: PeriodSpec,
    min_obs: int = config.MIN_OBS,
) -> pd.DataFrame:
    """Estimate CAPM alpha for each stock in one period."""
    stocks = _period_filter(stock_returns, "date", period)
    market = _period_filter(market_returns, "date", period)[["date", "market_return"]]
    merged = stocks.merge(market, on="date", how="inner")

    rows = []
    for code, sub in merged.groupby("code"):
        sub = sub.dropna(subset=["stock_return", "market_return"])
        if len(sub) < min_obs:
            continue

        y = sub["stock_return"].astype(float) - config.RISK_FREE_DAILY
        x = sub["market_return"].astype(float) - config.RISK_FREE_DAILY
        x = sm.add_constant(x, has_constant="add")
        model = sm.OLS(y, x).fit()

        alpha_daily = float(model.params["const"])
        beta = float(model.params["market_return"])
        rows.append(
            {
                "code": code,
                "period": period.label,
                "start": period.start,
                "end": period.end,
                "obs": int(model.nobs),
                "alpha_daily": alpha_daily,
                "alpha_annual": alpha_daily * config.TRADING_DAYS,
                "alpha_t": float(model.tvalues["const"]),
                "alpha_pvalue": float(model.pvalues["const"]),
                "beta": beta,
                "beta_t": float(model.tvalues["market_return"]),
                "beta_pvalue": float(model.pvalues["market_return"]),
                "r2": float(model.rsquared),
                "stock_total_return": float((1.0 + sub["stock_return"]).prod() - 1.0),
                "market_total_return": float((1.0 + sub["market_return"]).prod() - 1.0),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(f"{period.label} 没有足够样本用于CAPM回归")

    result = result.sort_values("alpha_annual", ascending=False).reset_index(drop=True)
    result["alpha_rank"] = np.arange(1, len(result) + 1)
    return result


def compare_alpha_periods(
    train_alpha: pd.DataFrame,
    test_alpha: pd.DataFrame,
    test_label: str,
    top_n: int = config.TOP_N,
) -> dict:
    """Compare historical and future alpha rankings."""
    train_cols = [
        "code",
        "obs",
        "alpha_daily",
        "alpha_annual",
        "alpha_t",
        "alpha_pvalue",
        "beta",
        "beta_pvalue",
        "r2",
        "stock_total_return",
        "market_total_return",
        "alpha_rank",
    ]
    comparison = train_alpha[train_cols].merge(
        test_alpha[train_cols],
        on="code",
        suffixes=("_train", "_test"),
        how="inner",
    )
    comparison = comparison.sort_values("alpha_annual_train", ascending=False).reset_index(drop=True)
    comparison["is_train_top20"] = comparison["alpha_rank_train"] <= top_n
    comparison["is_future_top20"] = comparison["alpha_rank_test"] <= top_n
    comparison["rank_change"] = comparison["alpha_rank_test"] - comparison["alpha_rank_train"]
    comparison["alpha_change"] = comparison["alpha_annual_test"] - comparison["alpha_annual_train"]

    train_top = comparison[comparison["is_train_top20"]].copy()
    future_top = comparison[comparison["is_future_top20"]].copy().sort_values("alpha_rank_test")
    overlap = comparison[comparison["is_train_top20"] & comparison["is_future_top20"]].copy()

    pearson = comparison["alpha_annual_train"].corr(comparison["alpha_annual_test"], method="pearson")
    spearman = comparison["alpha_annual_train"].corr(comparison["alpha_annual_test"], method="spearman")
    top_future_mean = train_top["alpha_annual_test"].mean()
    top_train_mean = train_top["alpha_annual_train"].mean()
    future_all_mean = comparison["alpha_annual_test"].mean()
    change = top_future_mean - top_train_mean
    if change > 0.01:
        trend = "上升"
    elif change < -0.01:
        trend = "下降"
    else:
        trend = "基本不变"

    group_df = build_group_persistence(comparison)
    summary = {
        "test_label": test_label,
        "stock_count": int(len(comparison)),
        "top_n": int(top_n),
        "overlap_count": int(len(overlap)),
        "overlap_ratio": float(len(overlap) / top_n),
        "train_top_alpha_mean": float(top_train_mean),
        "train_top_future_alpha_mean": float(top_future_mean),
        "future_all_alpha_mean": float(future_all_mean),
        "future_top_alpha_mean": float(future_top["alpha_annual_test"].mean()),
        "alpha_mean_change": float(change),
        "alpha_trend": trend,
        "pearson_corr": float(pearson),
        "spearman_corr": float(spearman),
        "top20_future_excess_vs_all": float(top_future_mean - future_all_mean),
        "top20_retention_codes": overlap["code"].tolist(),
    }

    return {
        "summary": summary,
        "comparison": comparison,
        "train_top": train_top.sort_values("alpha_rank_train"),
        "future_top": future_top,
        "overlap": overlap.sort_values("alpha_rank_train"),
        "group_persistence": group_df,
    }


def build_group_persistence(comparison: pd.DataFrame, groups: int = 5) -> pd.DataFrame:
    """Measure future alpha by historical alpha quintile."""
    ordered = comparison.sort_values("alpha_annual_train", ascending=False).copy()
    labels = [f"Q{i}({'高' if i == 1 else '低' if i == groups else '中'}Alpha)" for i in range(1, groups + 1)]
    ordered["history_group"] = pd.qcut(
        ordered["alpha_rank_train"],
        q=groups,
        labels=labels,
        duplicates="drop",
    )
    grouped = (
        ordered.groupby("history_group", observed=False)
        .agg(
            stock_count=("code", "count"),
            train_alpha_mean=("alpha_annual_train", "mean"),
            future_alpha_mean=("alpha_annual_test", "mean"),
            future_alpha_median=("alpha_annual_test", "median"),
            future_positive_ratio=("alpha_annual_test", lambda s: float((s > 0).mean())),
            avg_future_rank=("alpha_rank_test", "mean"),
        )
        .reset_index()
    )
    grouped["alpha_decay"] = grouped["future_alpha_mean"] - grouped["train_alpha_mean"]
    return grouped


def format_pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}%}"

