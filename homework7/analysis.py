"""Stationarity statistics for Homework 7."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from . import config


def adf_test(series: pd.Series) -> dict:
    """Run ADF test and return the values needed by the assignment."""
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.nunique() < 2 or len(clean) < 10:
        raise ValueError("ADF检验样本不足或序列无波动")

    stat, pvalue, used_lag, nobs, critical, _ = adfuller(clean, autolag="AIC")
    is_stationary = bool(pvalue < config.ADF_P_THRESHOLD and stat < critical["5%"])
    return {
        "adf_stat": float(stat),
        "p_value": float(pvalue),
        "used_lag": int(used_lag),
        "nobs": int(nobs),
        "critical_1pct": float(critical["1%"]),
        "critical_5pct": float(critical["5%"]),
        "critical_10pct": float(critical["10%"]),
        "is_stationary": is_stationary,
        "conclusion": "平稳" if is_stationary else "非平稳",
    }


def compute_log_return(df: pd.DataFrame) -> pd.DataFrame:
    """Add log returns by stock."""
    out = df.sort_values(["display_code", "date"]).copy()
    out["log_return"] = out.groupby("display_code")["close"].transform(
        lambda s: np.log(s).diff()
    )
    return out.replace([np.inf, -np.inf], np.nan)


def analyze_stationarity(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze close price and log-return stationarity for every stock."""
    panel = compute_log_return(prices)
    rows = []

    for (name, code), g in panel.groupby(["name", "display_code"], sort=False):
        close = g["close"].dropna()
        ret = g["log_return"].dropna()
        close_adf = adf_test(close)
        ret_adf = adf_test(ret)

        rows.append(
            {
                "stock_name": name,
                "code": code,
                "start_date": str(g["date"].min().date()),
                "end_date": str(g["date"].max().date()),
                "price_obs": int(close.shape[0]),
                "return_obs": int(ret.shape[0]),
                "price_mean": float(close.mean()),
                "price_variance": float(close.var(ddof=1)),
                "price_adf_stat": close_adf["adf_stat"],
                "price_p_value": close_adf["p_value"],
                "price_critical_5pct": close_adf["critical_5pct"],
                "price_stationarity": close_adf["conclusion"],
                "return_mean": float(ret.mean()),
                "return_variance": float(ret.var(ddof=1)),
                "return_adf_stat": ret_adf["adf_stat"],
                "return_p_value": ret_adf["p_value"],
                "return_critical_5pct": ret_adf["critical_5pct"],
                "return_stationarity": ret_adf["conclusion"],
            }
        )

    return panel, pd.DataFrame(rows)

