"""因子双质检：IC/IR 有效性检验 + 共线性检验（作业 1.3.3）。

IC（信息系数）：每月末因子截面值与下月前瞻收益的 Spearman 秩相关。
IR（信息比率）：IC 均值 / IC 标准差，衡量因子预测稳定性。
共线性：四因子 Z-Score 后的 Pearson 相关矩阵，|corr| 超阈值提示剔除。

质检结论给出保留因子清单（kept_factors），供后续 LGBM 与打分使用。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg


def compute_ic_series(panel: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    """逐月计算各因子与前瞻收益的截面 IC（Spearman）。

    返回长表：signal_date, factor, ic。
    """
    rows = []
    for d, grp in panel.groupby("signal_date"):
        y = grp["fwd_return"].values
        for f in factor_cols:
            x = grp[f].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 5:
                continue
            rho, _ = stats.spearmanr(x[mask], y[mask])
            rows.append({"signal_date": d, "factor": f, "ic": rho})
    return pd.DataFrame(rows)


def ic_ir_summary(ic_series: pd.DataFrame) -> pd.DataFrame:
    """汇总各因子 IC 均值、标准差、IR、IC 胜率（IC>0 占比）、绝对 IR。"""
    g = ic_series.groupby("factor")["ic"]
    summary = pd.DataFrame({
        "ic_mean": g.mean(),
        "ic_std": g.std(ddof=1),
        "ir": g.mean() / g.std(ddof=1),
        "ic_positive_rate": g.apply(lambda s: (s > 0).mean()),
        "n_months": g.size(),
    })
    summary["abs_ir"] = summary["ir"].abs()
    summary = summary.sort_values("abs_ir", ascending=False)
    return summary


def collinearity_matrix(panel: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    """四因子 Z-Score 后的 Pearson 相关矩阵（池化全部截面）。"""
    return panel[factor_cols].corr(method="pearson")


def select_factors(ic_summary: pd.DataFrame, corr: pd.DataFrame,
                   ir_floor: float = cfg.IR_FLOOR,
                   collinear_threshold: float = cfg.COLLINEAR_THRESHOLD) -> tuple[list[str], list[str], pd.DataFrame]:
    """根据双质检结果决定保留/剔除因子。

    规则（保守，保留作业规定的四因子体系）：
    1) 共线性：|corr| > collinear_threshold 的因子对中，剔除 IR 较弱的一方；
    2) 有效性：短样本(23月)下 IR 估计噪声大，仅当 |IR|<0.1 且与已保留因子高度共线
       时才作为共线性剔除；否则保留并标记为“弱因子”供报告讨论，不强行剔除。
    返回 (kept, dropped, advisory)，advisory 为各因子有效性/共线性提示。
    """
    dropped: set[str] = set()
    order = ic_summary.sort_values("abs_ir", ascending=False).index.tolist()
    kept: list[str] = []
    collinear_notes: dict[str, str] = {}
    for f in order:
        too_collinear = False
        for k in kept:
            c = corr.loc[f, k]
            if abs(c) > collinear_threshold:
                too_collinear = True
                dropped.add(f)
                collinear_notes[f] = f"与 {k} 高度共线(corr={c:.2f})，剔除较弱者"
                break
        if not too_collinear:
            kept.append(f)

    # 有效性提示（不强制剔除）
    advisory_rows = []
    for f in ic_summary.index:
        ir = ic_summary.loc[f, "ir"]
        pr = ic_summary.loc[f, "ic_positive_rate"]
        if f in dropped:
            note = collinear_notes.get(f, "剔除")
        elif abs(ir) < ir_floor:
            note = f"有效性偏弱(|IR|={abs(ir):.2f}<{ir_floor})，保留供LGBM非线性组合"
        else:
            note = "有效，保留"
        advisory_rows.append({"factor": f, "abs_ir": abs(ir), "ic_positive_rate": pr, "advisory": note})
    advisory = pd.DataFrame(advisory_rows).set_index("factor")

    return kept, sorted(dropped), advisory


def run_quality_check(panel: pd.DataFrame, factor_cols: list[str] | None = None) -> dict:
    """执行双质检，返回结果字典。"""
    if factor_cols is None:
        factor_cols = cfg.FACTOR_Z

    ic_series = compute_ic_series(panel, factor_cols)
    ic_summary = ic_ir_summary(ic_series)
    corr = collinearity_matrix(panel, factor_cols)
    kept, dropped, advisory = select_factors(ic_summary, corr)

    return {
        "ic_series": ic_series,
        "ic_summary": ic_summary,
        "corr_matrix": corr,
        "kept_factors": kept,
        "dropped_factors": dropped,
        "advisory": advisory,
        "factor_cols": factor_cols,
    }
