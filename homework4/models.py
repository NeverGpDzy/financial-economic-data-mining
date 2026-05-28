"""Modules 2-4: CAPM screening, single-factor test, IC/IR, multi-factor regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


# ======================== Module 2: CAPM Screening + Single-Factor Test ========================

def capm_screening(train_df: pd.DataFrame, mkt_df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: CAPM regression per stock. Ri-Rf = α + β×MKT + ε.

    Returns DataFrame of CAPM results per stock.
    """
    results = []
    mkt_excess = mkt_df["mkt_excess"].dropna()

    for code, group in train_df.groupby("code"):
        df = group.set_index("date")[["excess_return"]].join(
            mkt_excess.rename("mkt"), how="inner"
        ).dropna()

        if len(df) < 12:
            continue

        X = sm.add_constant(df["mkt"])
        y = df["excess_return"]
        try:
            model = sm.OLS(y, X).fit()
            results.append({
                "code": code,
                "name": group["name"].iloc[0],
                "alpha": model.params["const"],
                "beta": model.params["mkt"],
                "alpha_p": model.pvalues["const"],
                "beta_p": model.pvalues["mkt"],
                "r2": model.rsquared,
                "nobs": int(model.nobs),
                "beta_significant": model.pvalues["mkt"] < 0.05,
            })
        except Exception:
            continue

    return pd.DataFrame(results)


def single_factor_test(train_df: pd.DataFrame, factors: list[str]) -> dict:
    """Step 2-3: Single factor cross-sectional regression (Fama-MacBeth).

    Factors are Z-Score standardized within each month's cross-section
    before regression to ensure comparable betas across factors.

    Each month: next_excess_return = α + β×Factor_std + ε
    Significance via Fama-MacBeth t-test:
      t = mean(beta) / (std(beta) / sqrt(n_months))
    """
    results = {}
    valid_factors = []

    for factor in factors:
        beta_list = []
        months = sorted(train_df["date"].unique())

        for month in months:
            sub = train_df[train_df["date"] == month].dropna(subset=[factor, "next_excess_return"])
            if len(sub) < 10:
                continue

            # Z-Score standardize factor within this month's cross-section
            f_vals = sub[factor].values
            f_mean, f_std = f_vals.mean(), f_vals.std(ddof=0)
            f_std_safe = f_std if f_std > 0 else 1.0
            f_z = (f_vals - f_mean) / f_std_safe

            X = sm.add_constant(f_z)
            y = sub["next_excess_return"].values
            try:
                model = sm.OLS(y, X).fit()
                beta_list.append(model.params[1])
            except Exception:
                continue

        if len(beta_list) < 3:
            results[factor] = {
                "beta_mean": np.nan, "beta_std": np.nan,
                "n_months": len(beta_list), "fm_t": np.nan, "fm_p": np.nan,
                "valid": False,
            }
            continue

        mean_beta = float(np.mean(beta_list))
        std_beta = float(np.std(beta_list, ddof=1))
        n = len(beta_list)
        fm_t = mean_beta / (std_beta / np.sqrt(n)) if std_beta > 0 else 0.0
        fm_p = float(2 * stats.t.sf(abs(fm_t), df=n - 1))
        is_valid = fm_p < 0.05

        results[factor] = {
            "beta_mean": mean_beta,
            "beta_std": std_beta,
            "n_months": n,
            "fm_t": fm_t,
            "fm_p": fm_p,
            "beta_series": beta_list,
            "valid": is_valid,
        }

        if is_valid:
            valid_factors.append(factor)

    return {"factor_details": results, "valid_factors": valid_factors}


# ======================== Module 3: IC/IR Factor Quality Check ========================

def factor_standardize(train_df: pd.DataFrame, valid_factors: list[str]) -> pd.DataFrame:
    """Z-Score standardize valid factors: (factor - μ) / σ (cross-sectional per month)."""
    out = train_df.copy()
    for f in valid_factors:
        std_name = f"{f}_std"
        out[std_name] = out.groupby("date")[f].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
        )
        out[std_name] = out[std_name].fillna(0.0)
    return out


def compute_ic_ir(train_df: pd.DataFrame, valid_factors: list[str]) -> dict:
    """Compute monthly cross-sectional IC and IR for each valid factor.

    IC = Pearson correlation between standardized factor and next month's excess return.
    IR = IC_mean / IC_std.
    """
    ic_results = {}

    for f in valid_factors:
        std_col = f"{f}_std"
        ic_list = []
        months = sorted(train_df["date"].unique())

        for month in months:
            sub = train_df[train_df["date"] == month].dropna(subset=[std_col, "next_excess_return"])
            if len(sub) < 10:
                continue
            # Skip if factor has zero variance (all values identical)
            if sub[std_col].std() == 0:
                continue
            ic = sub[[std_col, "next_excess_return"]].corr(method="pearson").iloc[0, 1]
            if not np.isnan(ic):
                ic_list.append(ic)

        if not ic_list:
            ic_results[f] = {"IC_mean": np.nan, "IC_std": np.nan, "IR": np.nan, "grade": "无效"}
            continue

        ic_mean = float(np.mean(ic_list))
        ic_std = float(np.std(ic_list))
        ir = ic_mean / ic_std if ic_std != 0 else 0.0

        if ic_mean > 0.05:
            grade = "优秀因子"
        elif ic_mean > 0.02:
            grade = "具备预测能力"
        else:
            grade = "预测能力弱"

        ic_results[f] = {
            "IC_mean": ic_mean,
            "IC_std": ic_std,
            "IR": float(ir),
            "grade": grade,
            "ic_series": ic_list,
        }

    return ic_results


# ======================== Module 4: Multi-Factor Regression Static Weighting ========================

def multi_factor_regression(
    train_df: pd.DataFrame, valid_factors: list[str],
) -> tuple[dict, callable]:
    """Static factor weighting via pooled OLS.

    Model: next_excess_return = α + w1×Factor1_std + w2×Factor2_std + ... + ε
    Uses entire training set (panel regression across all months and stocks).

    Returns:
        weights: dict mapping factor name → weight
        score_func: callable that computes composite score from a DataFrame row
    """
    std_cols = [f"{f}_std" for f in valid_factors]
    df = train_df.dropna(subset=std_cols + ["next_excess_return"])

    X = df[std_cols].values
    X = sm.add_constant(X)
    y = df["next_excess_return"].values

    model = sm.OLS(y, X).fit()

    weights = {}
    for i, f in enumerate(valid_factors):
        weights[f] = float(model.params[i + 1])

    # P-values for weights
    for i, f in enumerate(valid_factors):
        weights[f"{f}_pvalue"] = float(model.pvalues[i + 1])

    weights["alpha"] = float(model.params[0])
    weights["alpha_pvalue"] = float(model.pvalues[0])
    weights["r2"] = float(model.rsquared)
    weights["nobs"] = int(model.nobs)

    def score_func(row: pd.Series) -> float:
        """Compute composite score = Σ w_i × factor_i_std."""
        total = 0.0
        for f in valid_factors:
            std_col = f"{f}_std"
            total += weights[f] * row.get(std_col, 0.0)
        return total

    return weights, score_func
