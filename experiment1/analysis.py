"""Herd-index construction and market-prediction analysis."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from . import config


def _minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    lo = s.min()
    hi = s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def winsorize_series(series: pd.Series, sigma: float = 3.0) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return series
    return series.clip(mean - sigma * std, mean + sigma * std)


def build_herd_index(weekly_sentiment: pd.DataFrame) -> pd.DataFrame:
    df = weekly_sentiment.copy().sort_values("week").reset_index(drop=True)
    df["E_P_t"] = df["P_t"].shift(1).rolling(config.ROLLING_BASELINE_WEEKS, min_periods=1).mean()
    df["E_P_t"] = df["E_P_t"].fillna(df["P_t"])
    df["H1t"] = df["P_t"] - df["E_P_t"]
    denom = df["WeekPositive"] + df["WeekNegative"]
    df["H2t"] = (1 - (df["WeekPositive"] - df["WeekNegative"]).abs() / denom).where(denom.ne(0), 1.0)
    df["H1t_winsor"] = winsorize_series(df["H1t"])
    df["H2t_winsor"] = winsorize_series(df["H2t"])
    df["H1t_norm"] = _minmax(df["H1t_winsor"].abs())
    df["H2t_norm"] = _minmax(df["H2t_winsor"])
    df["H3t"] = df["H1t_norm"] * (1 - df["H2t_norm"])
    df["H3t_formula_reference"] = _minmax(df["H1t_winsor"]) * df["H2t_norm"]
    columns = [
        "week", "week_start", "week_end", "week_id", "P_t", "E_P_t", "H1t", "H2t",
        "H1t_norm", "H2t_norm", "H3t", "H3t_formula_reference",
        "WeekPositive", "WeekNeutral", "WeekNegative", "NewsCount",
    ]
    return df[columns]


def build_modeling_dataset(weekly_herd: pd.DataFrame, hs300_weekly: pd.DataFrame) -> pd.DataFrame:
    df = weekly_herd.merge(
        hs300_weekly[["week_id", "week", "close", "return"]],
        on=["week_id", "week"],
        how="inner",
    )
    return df.dropna(subset=["H3t", "return"]).sort_values("week").reset_index(drop=True)


def correlation_analysis(modeling: pd.DataFrame) -> pd.DataFrame:
    return modeling[["H1t", "H2t", "H3t", "return"]].corr(method="pearson")


def granger_analysis(modeling: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    data = modeling[["return", "H3t"]].dropna()
    maxlag = min(config.MAX_GRANGER_LAG, max(1, len(data) // 5))
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
    for lag, tests in result.items():
        stat, pvalue, *_ = tests[0]["ssr_ftest"]
        rows.append({"lag": lag, "ssr_ftest_stat": stat, "p_value": pvalue})
    out = pd.DataFrame(rows)
    best_lag = int(out.sort_values(["p_value", "lag"]).iloc[0]["lag"])
    return out, best_lag


def lag_regression_and_adf(modeling: pd.DataFrame, best_lag: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = modeling.copy()
    df[f"H3t_lag{best_lag}"] = df["H3t"].shift(best_lag)
    reg_df = df.dropna(subset=[f"H3t_lag{best_lag}", "return"]).copy()
    x = sm.add_constant(reg_df[[f"H3t_lag{best_lag}"]])
    y = reg_df["return"]
    model = sm.OLS(y, x).fit()
    reg_df["fitted_return"] = model.predict(x)
    reg_df["residual"] = y - reg_df["fitted_return"]
    adf = adfuller(reg_df["residual"])
    regression = pd.DataFrame(
        [
            {
                "best_lag": best_lag,
                "intercept": model.params["const"],
                "beta": model.params[f"H3t_lag{best_lag}"],
                "r_squared": model.rsquared,
                "p_value_beta": model.pvalues[f"H3t_lag{best_lag}"],
                "n_obs": int(model.nobs),
                "equation": f"return_t = {model.params['const']:.6f} + {model.params[f'H3t_lag{best_lag}']:.6f} * H3t_lag{best_lag}",
            }
        ]
    )
    adf_df = pd.DataFrame(
        [
            {
                "adf_statistic": adf[0],
                "p_value": adf[1],
                "used_lag": adf[2],
                "n_obs": adf[3],
                "is_residual_stationary_5pct": bool(adf[1] < 0.05),
            }
        ]
    )
    return regression, adf_df, reg_df


def save_analysis_outputs(
    correlation: pd.DataFrame,
    granger: pd.DataFrame,
    regression: pd.DataFrame,
    adf: pd.DataFrame,
    reg_dataset: pd.DataFrame,
) -> None:
    correlation.to_csv(config.OUTPUT_DIR / "correlation_matrix.csv", encoding="utf-8-sig")
    granger.to_csv(config.OUTPUT_DIR / "granger_results.csv", index=False, encoding="utf-8-sig")
    regression.to_csv(config.OUTPUT_DIR / "regression_results.csv", index=False, encoding="utf-8-sig")
    adf.to_csv(config.OUTPUT_DIR / "adf_results.csv", index=False, encoding="utf-8-sig")
    reg_dataset.to_csv(config.OUTPUT_DIR / "regression_dataset.csv", index=False, encoding="utf-8-sig")

