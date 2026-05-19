"""CAPM regression utilities."""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm


def fit_capm(stock_returns: pd.Series, market_returns: pd.Series, rf_daily: float):
    """Fit Ri - Rf = alpha + beta * (Rm - Rf)."""
    df = pd.DataFrame({"Ri": stock_returns, "Rm": market_returns}).dropna()
    df["Ri_excess"] = df["Ri"] - rf_daily
    df["Rm_excess"] = df["Rm"] - rf_daily

    x = sm.add_constant(df["Rm_excess"], has_constant="add")
    model = sm.OLS(df["Ri_excess"], x).fit()

    return {
        "alpha": float(model.params["const"]),
        "beta": float(model.params["Rm_excess"]),
        "alpha_pvalue": float(model.pvalues["const"]),
        "beta_pvalue": float(model.pvalues["Rm_excess"]),
        "r_squared": float(model.rsquared),
        "observations": int(model.nobs),
        "model": model,
    }


def build_capm_table(
    stock_data: dict[str, pd.DataFrame],
    market_returns: pd.Series,
    rf_daily: float,
    significance_level: float,
) -> pd.DataFrame:
    """Fit CAPM for each stock and return a sorted result table."""
    rows = []
    for name, df in stock_data.items():
        fit = fit_capm(df["return"], market_returns, rf_daily)
        rows.append(
            {
                "stock": name,
                "alpha_daily": fit["alpha"],
                "alpha_annualized": fit["alpha"] * 250,
                "beta": fit["beta"],
                "alpha_pvalue": fit["alpha_pvalue"],
                "beta_pvalue": fit["beta_pvalue"],
                "r_squared": fit["r_squared"],
                "observations": fit["observations"],
                "alpha_significant_10pct": fit["alpha_pvalue"] < significance_level,
            }
        )

    result = pd.DataFrame(rows)
    return result.sort_values(["alpha_significant_10pct", "alpha_daily"], ascending=[False, False])


def select_best_alpha(result: pd.DataFrame) -> pd.Series:
    """Pick the highest-alpha stock among significant alphas, fallback to highest alpha."""
    significant = result[result["alpha_significant_10pct"]]
    pool = significant if not significant.empty else result
    return pool.sort_values("alpha_daily", ascending=False).iloc[0]

