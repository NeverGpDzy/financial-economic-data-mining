"""CAPM and Two-Factor model fitting for Homework 3."""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm


def fit_capm(
    stock_ret: pd.Series, mkt_ret: pd.Series, rf_daily: float,
) -> dict:
    """Fit single-factor CAPM: Ri - Rf = α + β1*(Rm - Rf)."""
    df = pd.DataFrame({"Ri": stock_ret, "Rm": mkt_ret}).dropna()
    df["Ri_excess"] = df["Ri"] - rf_daily
    df["Rm_excess"] = df["Rm"] - rf_daily

    x = sm.add_constant(df["Rm_excess"])
    model = sm.OLS(df["Ri_excess"], x).fit()

    return {
        "alpha": float(model.params["const"]),
        "beta_mkt": float(model.params["Rm_excess"]),
        "alpha_pvalue": float(model.pvalues["const"]),
        "beta_mkt_pvalue": float(model.pvalues["Rm_excess"]),
        "r_squared": float(model.rsquared),
        "observations": int(model.nobs),
        "model": model,
    }


def fit_two_factor(
    stock_ret: pd.Series, mkt_ret: pd.Series, pe_norm: pd.Series, rf_daily: float,
) -> dict:
    """Fit two-factor model: Ri - Rf = α + β1*(Rm - Rf) + β2*PE_TTM_norm."""
    df = pd.DataFrame({"Ri": stock_ret, "Rm": mkt_ret, "PE": pe_norm}).dropna()
    df["Ri_excess"] = df["Ri"] - rf_daily
    df["Rm_excess"] = df["Rm"] - rf_daily

    x = sm.add_constant(df[["Rm_excess", "PE"]])
    model = sm.OLS(df["Ri_excess"], x).fit()

    return {
        "alpha": float(model.params["const"]),
        "beta_mkt": float(model.params["Rm_excess"]),
        "beta_pe": float(model.params["PE"]),
        "alpha_pvalue": float(model.pvalues["const"]),
        "beta_mkt_pvalue": float(model.pvalues["Rm_excess"]),
        "beta_pe_pvalue": float(model.pvalues["PE"]),
        "r_squared": float(model.rsquared),
        "observations": int(model.nobs),
        "model": model,
    }


def build_model_comparison(
    stock_data: dict[str, pd.DataFrame],
    mkt_col: str,
    pe_col: str,
    rf_daily: float,
) -> pd.DataFrame:
    """Build CAPM vs two-factor comparison table for all stocks."""
    rows = []
    for name, df in stock_data.items():
        capm = fit_capm(df["stock_return"], df[mkt_col], rf_daily)
        tf = fit_two_factor(df["stock_return"], df[mkt_col], df[pe_col], rf_daily)
        rows.append({
            "股票": name,
            # CAPM
            "CAPM_α": capm["alpha"],
            "CAPM_β(mkt)": capm["beta_mkt"],
            "CAPM_α_p值": capm["alpha_pvalue"],
            "CAPM_R²": capm["r_squared"],
            # Two-factor
            "二因子_α": tf["alpha"],
            "二因子_β(mkt)": tf["beta_mkt"],
            "二因子_β(pe)": tf["beta_pe"],
            "二因子_α_p值": tf["alpha_pvalue"],
            "二因子_β(pe)_p值": tf["beta_pe_pvalue"],
            "二因子_R²": tf["r_squared"],
            # Comparison
            "α变化": tf["alpha"] - capm["alpha"],
            "R²提升": tf["r_squared"] - capm["r_squared"],
            # For backtest use
            "_alpha_2f": tf["alpha"],
        })

    result = pd.DataFrame(rows)
    result["alpha_rank"] = result["_alpha_2f"].rank(ascending=False).astype(int)
    return result
