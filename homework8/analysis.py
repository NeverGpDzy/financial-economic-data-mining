"""Cointegration and pair-trading analysis for Homework 8."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from . import config


def adf_test(series: pd.Series) -> dict:
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.nunique() < 2 or len(clean) < 20:
        raise ValueError("ADF检验样本不足或序列无波动")
    stat, pvalue, used_lag, nobs, critical, _ = adfuller(clean, autolag="AIC")
    is_stationary = bool(pvalue < config.ADF_P_THRESHOLD and stat < critical["5%"])
    return {
        "adf_stat": float(stat),
        "p_value": float(pvalue),
        "used_lag": int(used_lag),
        "nobs": int(nobs),
        "critical_5pct": float(critical["5%"]),
        "is_stationary": is_stationary,
        "conclusion": "平稳" if is_stationary else "非平稳",
    }


def eg_cointegration_test(y: pd.Series, x: pd.Series) -> dict:
    """Engle-Granger two-step test using OLS residual ADF."""
    aligned = pd.concat([y, x], axis=1).dropna()
    aligned.columns = ["y", "x"]
    model = sm.OLS(aligned["y"], sm.add_constant(aligned["x"])).fit()
    residual = model.resid.rename("residual")
    adf = adf_test(residual)
    return {
        "alpha": float(model.params["const"]),
        "beta": float(model.params["x"]),
        "r_squared": float(model.rsquared),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std(ddof=1)),
        "adf_stat": adf["adf_stat"],
        "p_value": adf["p_value"],
        "critical_5pct": adf["critical_5pct"],
        "is_cointegrated": adf["is_stationary"],
        "conclusion": "协整显著" if adf["is_stationary"] else "未通过协整检验",
    }


def scan_all_pairs(close_matrix: pd.DataFrame) -> pd.DataFrame:
    """Run EG tests for all ordered-independent pairs."""
    rows = []
    for y_name, x_name in combinations(close_matrix.columns, 2):
        result = eg_cointegration_test(close_matrix[y_name], close_matrix[x_name])
        rows.append(
            {
                "y_asset": y_name,
                "x_asset": x_name,
                **result,
            }
        )
    out = pd.DataFrame(rows).sort_values(["p_value", "adf_stat"], ascending=[True, True])
    out["rank"] = range(1, len(out) + 1)
    return out[
        [
            "rank",
            "y_asset",
            "x_asset",
            "alpha",
            "beta",
            "r_squared",
            "residual_mean",
            "residual_std",
            "adf_stat",
            "p_value",
            "critical_5pct",
            "is_cointegrated",
            "conclusion",
        ]
    ]


def build_best_pair_detail(close_matrix: pd.DataFrame, pairs: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Build residual, z-score and signals for the best pair."""
    best = pairs.iloc[0].to_dict()
    y_name = best["y_asset"]
    x_name = best["x_asset"]
    aligned = close_matrix[[y_name, x_name]].dropna().copy()
    aligned.columns = ["y_price", "x_price"]

    model = sm.OLS(aligned["y_price"], sm.add_constant(aligned["x_price"])).fit()
    aligned["fitted_y"] = model.fittedvalues
    aligned["spread"] = model.resid
    aligned["z_score"] = (aligned["spread"] - aligned["spread"].mean()) / aligned["spread"].std(ddof=1)
    aligned["signal"] = "观望"
    aligned.loc[aligned["z_score"] > config.ZSCORE_ENTRY, "signal"] = "价差过大：做空Y、做多X"
    aligned.loc[aligned["z_score"] < -config.ZSCORE_ENTRY, "signal"] = "价差过小：做多Y、做空X"
    aligned.loc[aligned["z_score"].abs() <= 0.2, "signal"] = "接近均值：平仓观察"
    aligned = aligned.reset_index()

    detail = {
        **best,
        "ols_alpha": float(model.params["const"]),
        "ols_beta": float(model.params["x_price"]),
        "ols_r_squared": float(model.rsquared),
        "trade_signal_count_long_y": int((aligned["z_score"] < -config.ZSCORE_ENTRY).sum()),
        "trade_signal_count_short_y": int((aligned["z_score"] > config.ZSCORE_ENTRY).sum()),
        "exit_zone_count": int((aligned["z_score"].abs() <= 0.2).sum()),
    }
    return detail, aligned

