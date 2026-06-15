"""Factor preprocessing, IC/IR, VIF and single-factor OLS tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import config


def _cross_section_winsor_zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)

    median = x.median()
    x = x.fillna(median)
    mu = x.mean()
    sigma = x.std(ddof=0)
    if pd.isna(sigma) or sigma == 0:
        return pd.Series(0.0, index=series.index)
    clipped = x.clip(mu - 3 * sigma, mu + 3 * sigma)
    return (clipped - clipped.mean()) / (clipped.std(ddof=0) + 1e-10)


def _cross_section_winsor_raw(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = x.dropna()
    if len(valid) < 2:
        return x
    mu = valid.mean()
    sigma = valid.std(ddof=0)
    if pd.isna(sigma) or sigma == 0:
        return x
    return x.clip(mu - 3 * sigma, mu + 3 * sigma)


def preprocess_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """Positive-orient, winsorize and standardize all value factors by year."""
    df = panel.copy()
    for factor, meta in config.FACTOR_META.items():
        pos_col = f"{factor}_pos"
        z_col = f"{factor}_z"
        direction = meta["direction"]
        df[pos_col] = pd.to_numeric(df[factor], errors="coerce") * direction
        df[z_col] = df.groupby("year", group_keys=False)[pos_col].apply(_cross_section_winsor_zscore)

    df["value_equal_weight_score"] = df[config.FACTOR_Z_COLUMNS].mean(axis=1)
    df["target_fcff_growth_1y"] = df.groupby("year", group_keys=False)["fcff_growth_1y"].apply(
        _cross_section_winsor_raw
    )
    df["target_fcff_growth_3y_ann"] = df.groupby("year", group_keys=False)["fcff_growth_3y_ann"].apply(
        _cross_section_winsor_raw
    )
    return df


def add_traditional_scores(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply fixed Buffett-Munger style financial threshold rules."""
    df = panel.copy()
    df["rule_profit_growth_positive"] = df["F1_profit_cagr"] > 0
    df["rule_net_margin_positive"] = df["F3_net_margin"] > 0
    df["rule_light_asset"] = df["F4_light_asset"] < 0.35
    df["rule_debt_control"] = df["F5_low_debt"] < 85
    df["rule_roe_quality"] = df["F6_roe"] > 8
    df["rule_expense_control"] = df["F7_low_expense"] < 35
    df["rule_cash_quality"] = df["F8_ocf_profit"] > 50
    df["rule_dividend_positive"] = df["F9_dividend_payout"] > 0.05
    df["rule_dividend_yield"] = df["F10_dividend_yield"] > 0.5
    df["rule_fcff_positive"] = df["fcff"] > 0
    rule_cols = list(config.TRADITIONAL_RULES.keys())
    df["traditional_rule_count"] = df[rule_cols].sum(axis=1)
    df["traditional_score"] = df["traditional_rule_count"] + 0.01 * df["value_equal_weight_score"]
    return df


def compute_ic_ir(panel: pd.DataFrame, train_end_year: int = config.MODEL_TRAIN_END_YEAR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute annual cross-section Spearman IC and IC/IR summary."""
    train = panel[
        (panel["year"] >= config.TRAIN_START_YEAR)
        & (panel["year"] <= train_end_year)
        & panel["target_fcff_growth_1y"].notna()
    ].copy()
    rows = []
    yearly_rows = []
    for factor in config.FACTOR_COLUMNS:
        z_col = f"{factor}_z"
        for year, group in train.groupby("year"):
            valid = group[[z_col, "target_fcff_growth_1y"]].dropna()
            if len(valid) < 8 or valid[z_col].nunique() <= 1 or valid["target_fcff_growth_1y"].nunique() <= 1:
                continue
            ic, p_value = stats.spearmanr(valid[z_col], valid["target_fcff_growth_1y"])
            yearly_rows.append(
                {
                    "factor": factor,
                    "因子": config.FACTOR_META[factor]["label"],
                    "year": int(year),
                    "IC": float(ic),
                    "p_value": float(p_value),
                    "n": int(len(valid)),
                }
            )

    yearly_ic = pd.DataFrame(yearly_rows)
    if yearly_ic.empty:
        raise ValueError("训练期没有可用于IC检验的年度截面")

    for factor, sub in yearly_ic.groupby("factor"):
        ic_mean = sub["IC"].mean()
        ic_std = sub["IC"].std(ddof=1)
        ir = ic_mean / (ic_std + 1e-10)
        rows.append(
            {
                "factor": factor,
                "因子": config.FACTOR_META[factor]["label"],
                "IC均值": ic_mean,
                "IC标准差": ic_std,
                "IR": ir,
                "年度截面数": int(len(sub)),
                "IC有效": bool(ic_mean > config.IC_THRESHOLD),
                "IR达标": bool(abs(ir) > config.IR_THRESHOLD),
            }
        )
    summary = pd.DataFrame(rows).sort_values("IC均值", ascending=False).reset_index(drop=True)
    return summary, yearly_ic.sort_values(["factor", "year"]).reset_index(drop=True)


def _compute_vif_once(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    x = df[columns].dropna()
    rows = []
    for col in columns:
        other = [c for c in columns if c != col]
        if not other or x[col].nunique() <= 1:
            vif = 1.0
            r_squared = 0.0
        else:
            y = x[col].values
            x_other = sm.add_constant(x[other], has_constant="add")
            try:
                model = sm.OLS(y, x_other).fit()
                r_squared = float(max(0.0, min(1.0, model.rsquared)))
                vif = 1.0 / (1.0 - r_squared + 1e-10)
            except Exception:
                r_squared = np.nan
                vif = np.nan
        factor = col.removesuffix("_z")
        rows.append(
            {
                "factor": factor,
                "因子": config.FACTOR_META[factor]["label"],
                "feature": col,
                "VIF": vif,
                "R2": r_squared,
            }
        )
    return pd.DataFrame(rows)


def compute_vif(panel: pd.DataFrame, train_end_year: int = config.MODEL_TRAIN_END_YEAR) -> tuple[pd.DataFrame, list[str]]:
    """Compute VIF and iteratively drop columns above the configured threshold."""
    train = panel[
        (panel["year"] >= config.TRAIN_START_YEAR) & (panel["year"] <= train_end_year)
    ].copy()
    columns = config.FACTOR_Z_COLUMNS.copy()
    dropped: list[str] = []
    history = []
    while len(columns) > 2:
        vif_df = _compute_vif_once(train, columns)
        max_row = vif_df.sort_values("VIF", ascending=False).iloc[0]
        vif_df["round"] = len(dropped) + 1
        history.append(vif_df)
        if pd.isna(max_row["VIF"]) or max_row["VIF"] <= config.VIF_THRESHOLD:
            break
        dropped.append(str(max_row["feature"]))
        columns.remove(str(max_row["feature"]))

    final_vif = _compute_vif_once(train, columns)
    final_vif["round"] = len(dropped) + 1
    final_vif["判定"] = np.where(final_vif["VIF"] > config.VIF_THRESHOLD, "高共线", "保留")
    for feature in dropped:
        factor = feature.removesuffix("_z")
        final_vif = pd.concat(
            [
                final_vif,
                pd.DataFrame(
                    [
                        {
                            "factor": factor,
                            "因子": config.FACTOR_META[factor]["label"],
                            "feature": feature,
                            "VIF": np.nan,
                            "R2": np.nan,
                            "round": np.nan,
                            "判定": "已迭代剔除",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return final_vif.sort_values("判定").reset_index(drop=True), columns


def single_factor_ols(panel: pd.DataFrame, train_end_year: int = config.MODEL_TRAIN_END_YEAR) -> pd.DataFrame:
    """Run annual single-factor cross-section OLS against future 1Y FCFF growth."""
    train = panel[
        (panel["year"] >= config.TRAIN_START_YEAR)
        & (panel["year"] <= train_end_year)
        & panel["target_fcff_growth_1y"].notna()
    ].copy()
    rows = []
    for factor in config.FACTOR_COLUMNS:
        z_col = f"{factor}_z"
        beta_list = []
        p_list = []
        t_list = []
        section_count = 0
        for _, group in train.groupby("year"):
            valid = group[[z_col, "target_fcff_growth_1y"]].dropna()
            if len(valid) < 8 or valid[z_col].nunique() <= 1:
                continue
            x = sm.add_constant(valid[z_col], has_constant="add")
            try:
                model = sm.OLS(valid["target_fcff_growth_1y"], x).fit()
            except Exception:
                continue
            beta_list.append(float(model.params[z_col]))
            p_list.append(float(model.pvalues[z_col]))
            t_list.append(float(model.tvalues[z_col]))
            section_count += 1

        p_mean = float(np.mean(p_list)) if p_list else np.nan
        rows.append(
            {
                "factor": factor,
                "因子": config.FACTOR_META[factor]["label"],
                "β均值": float(np.mean(beta_list)) if beta_list else np.nan,
                "p值均值": p_mean,
                "t统计量均值": float(np.mean(t_list)) if t_list else np.nan,
                "有效截面数": section_count,
                "判定": "线性有效" if pd.notna(p_mean) and p_mean < config.OLS_P_THRESHOLD else "待定(非线性)",
            }
        )
    return pd.DataFrame(rows).sort_values("p值均值").reset_index(drop=True)


def select_model_features(ic_ir: pd.DataFrame, vif_keep_features: list[str]) -> list[str]:
    """Select LGBM features after VIF screening; IC-effective factors are highlighted in reports."""
    if len(vif_keep_features) >= 3:
        return vif_keep_features
    fallback = ic_ir.sort_values("IC均值", ascending=False)["factor"].head(5).tolist()
    return [f"{factor}_z" for factor in fallback]
