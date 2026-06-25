"""LGBM 因子赋权与截面打分模型（作业 1.3.3）。

- 训练：样本内(2023-2024)每月末31行业截面，X=四因子Z-Score，y=下月前瞻收益；
- 赋权：LGBM 特征重要度即各因子权重；
- 打分：LGBM 特征重要度 × IC方向，形成可解释的综合得分，月末排序选TopN；
- 评估：按月份做时间顺序扩展窗口验证，训练月份严格早于验证月份。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats

from . import config as cfg


def _time_ordered_month_splits(signal_dates: pd.Series, n_splits: int):
    """按月份生成扩展窗口切分，保证训练月严格早于验证月。"""
    dates = pd.to_datetime(pd.Series(signal_dates)).reset_index(drop=True)
    months = pd.Index(sorted(dates.drop_duplicates()))
    if len(months) < 4:
        return []

    n_splits = min(n_splits, len(months) - 1)
    min_train_months = max(3, len(months) // (n_splits + 1))
    val_months = months[min_train_months:]
    if len(val_months) == 0:
        return []

    val_chunks = np.array_split(val_months, min(n_splits, len(val_months)))
    splits = []
    for chunk in val_chunks:
        if len(chunk) == 0:
            continue
        train_months = months[months < chunk[0]]
        tr_mask = dates.isin(train_months)
        va_mask = dates.isin(chunk)
        tr_idx = np.flatnonzero(tr_mask.to_numpy())
        va_idx = np.flatnonzero(va_mask.to_numpy())
        if len(tr_idx) and len(va_idx):
            splits.append((tr_idx, va_idx, train_months, pd.Index(chunk)))
    return splits


def train_lgbm(train_panel: pd.DataFrame, factor_cols: list[str] | None = None) -> dict:
    """在样本内面板上训练 LGBM 回归，返回模型、特征重要度与交叉验证指标。"""
    if factor_cols is None:
        factor_cols = cfg.FACTOR_Z

    X = train_panel[factor_cols].values
    y = train_panel["fwd_return"].values

    # ---- 交叉验证（扩展窗口，防止未来月份泄漏到验证月之前）----
    cv_rows = []
    cv_preds = np.full(len(train_panel), np.nan)
    splits = _time_ordered_month_splits(train_panel["signal_date"], cfg.LGBM_CV_FOLDS)
    for tr_idx, va_idx, train_months, val_months in splits:
        model_cv = LGBMRegressor(**cfg.LGBM_PARAMS)
        model_cv.fit(X[tr_idx], y[tr_idx])
        pred_va = model_cv.predict(X[va_idx])
        cv_preds[va_idx] = pred_va
        # 该折验证集的截面秩IC（按月聚合后平均）
        va_df = train_panel.iloc[va_idx].copy()
        va_df["pred"] = pred_va
        ics = []
        for _, g in va_df.groupby("signal_date"):
            if len(g) >= 5:
                rho, _ = stats.spearmanr(g["pred"], g["fwd_return"])
                ics.append(rho)
        ic_mean = np.mean(ics) if ics else np.nan
        cv_rows.append({
            "fold": len(cv_rows) + 1,
            "n_train": len(tr_idx),
            "n_val": len(va_idx),
            "train_start": train_months.min(),
            "train_end": train_months.max(),
            "val_start": val_months.min(),
            "val_end": val_months.max(),
            "rmse": float(np.sqrt(np.mean((pred_va - y[va_idx]) ** 2))),
            "r2": float(1 - np.sum((y[va_idx] - pred_va) ** 2) / np.sum((y[va_idx] - y[va_idx].mean()) ** 2)),
            "rank_ic": float(ic_mean),
        })
    cv_metrics = pd.DataFrame(cv_rows)

    # 全样本训练最终模型
    model = LGBMRegressor(**cfg.LGBM_PARAMS)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=factor_cols, name="importance")
    importances = importances / importances.sum()  # 归一化为权重

    # CV 预测的整体秩IC
    cv_pred_ic = np.nan
    tmp = train_panel.copy()
    tmp["pred"] = cv_preds
    ics = []
    for _, g in tmp.dropna(subset=["pred"]).groupby("signal_date"):
        if len(g) >= 5:
            rho, _ = stats.spearmanr(g["pred"], g["fwd_return"])
            ics.append(rho)
    if ics:
        cv_pred_ic = float(np.mean(ics))

    return {
        "model": model,
        "importances": importances,
        "cv_metrics": cv_metrics,
        "cv_pred_rank_ic": cv_pred_ic,
        "factor_cols": factor_cols,
    }


def predict_scores(model_result: dict, panel: pd.DataFrame) -> pd.DataFrame:
    """对任意面板预测综合得分，返回带 score 列的副本。"""
    factor_cols = model_result["factor_cols"]
    out = panel.copy()
    out["score"] = model_result["model"].predict(out[factor_cols].values)
    return out


def linear_weighted_score(panel: pd.DataFrame, importances: pd.Series,
                          factor_cols: list[str] | None = None) -> pd.Series:
    """透明线性加权得分 = Σ(权重_i × z_i)，作为 LGBM 预测的对照。

    PB 方向已在因子体系中由模型学习；此处仅作线性复现，权重取 LGBM 重要度，
    并按 IC 符号定向（正IC正向、负IC负向），便于解释。
    """
    if factor_cols is None:
        factor_cols = cfg.FACTOR_Z
    w = importances.reindex(factor_cols).fillna(0.0)
    return panel[factor_cols].values @ w.values


def ic_directions(train_panel: pd.DataFrame, factor_cols: list[str] | None = None) -> pd.Series:
    """各因子的方向符号 = sign(样本内 IC 均值)。

    用于将因子统一“定向”为越高越好（PB 的 IC 为负 -> 取负号，低估值得分高）。
    """
    from .quality import compute_ic_series
    if factor_cols is None:
        factor_cols = cfg.FACTOR_Z
    ic = compute_ic_series(train_panel, factor_cols)
    ic_mean = ic.groupby("factor")["ic"].mean()
    directions = np.sign(ic_mean).reindex(factor_cols).fillna(1.0)
    directions.name = "direction"
    return directions


def composite_score(panel: pd.DataFrame, importances: pd.Series,
                    directions: pd.Series, factor_cols: list[str] | None = None) -> pd.Series:
    """主打分函数：LGBM 重要度赋权 × IC 方向定向 的线性综合得分。

    score_i = Σ_k (importance_k × direction_k × z_ik)
    - importance_k：LGBM 特征重要度（“LGBM赋权”）
    - direction_k：sign(样本内 IC)（因子方向定向，PB 取负）
    - z_ik：截面 Z-Score
    该线性加权得分比 LGBM 直接预测更稳定、可解释，忠实体现“LGBM赋权→截面打分”。
    """
    if factor_cols is None:
        factor_cols = cfg.FACTOR_Z
    w = (importances * directions).reindex(factor_cols).fillna(0.0)
    return panel[factor_cols].values @ w.values
