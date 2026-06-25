"""模块5 鲁棒性与迭代优化 + 模块4 风险控制演示。

必做两项测试：
- 参数敏感性：入选行业数量 Top3 vs Top5（2025样本外）；
- 时间敏感性：2024(样本内) vs 2025(样本外) 两个完整年度，是否均跑赢沪深300。
附加：
- 打分方法敏感性：LGBM综合得分 vs LGBM直接预测；
- 风险控制：回撤触发减仓的收益-回撤权衡谱系。
"""

from __future__ import annotations

import pandas as pd

from . import config as cfg
from .backtest import BacktestResult, apply_drawdown_control, backtest
from .data_loader import MarketData
from .model import composite_score, predict_scores


def _metrics_row(r: BacktestResult) -> dict:
    M = r.metrics
    return {
        "name": r.name,
        "cumulative_return": M["cumulative_return"],
        "annualized_return": M["annualized_return"],
        "max_drawdown": M["max_drawdown"],
        "sharpe": M["sharpe"],
        "benchmark_annualized": M["benchmark_annualized"],
        "excess_cumulative": M["excess_cumulative"],
        "monthly_win_rate_vs_bench": M["monthly_win_rate_vs_bench"],
        "total_turnover": M["total_turnover"],
        "total_cost": M["total_cost"],
        "target_return_ok": M["target_return_ok"],
        "target_dd_ok": M["target_dd_ok"],
    }


def run_parameter_sensitivity(scored_panel: pd.DataFrame, market: MarketData,
                              year_start: str, year_end: str) -> dict:
    """参数敏感性：Top3 vs Top5。"""
    sub = scored_panel[(scored_panel["fwd_date"] >= pd.Timestamp(year_start)) &
                       (scored_panel["fwd_date"] <= pd.Timestamp(year_end))]
    results = {}
    for n in cfg.TOP_N_GRID:
        r = backtest(sub, market, top_n=n, name=f"Top{n}_{year_start[:4]}")
        results[f"top{n}"] = r
    table = pd.DataFrame([_metrics_row(r) for r in results.values()])
    return {"results": results, "table": table}


def run_time_sensitivity(scored_panel: pd.DataFrame, market: MarketData) -> dict:
    """时间敏感性：2024(样本内) vs 2025(样本外)。"""
    results = {}
    for label, (s, e) in cfg.ROBUSTNESS_YEARS.items():
        sub = scored_panel[(scored_panel["fwd_date"] >= pd.Timestamp(s)) &
                           (scored_panel["fwd_date"] <= pd.Timestamp(e))]
        r = backtest(sub, market, top_n=cfg.TOP_N_DEFAULT, name=label)
        results[label] = r
    table = pd.DataFrame([_metrics_row(r) for r in results.values()])
    table["beat_hs300"] = table["excess_cumulative"] > 0
    return {"results": results, "table": table}


def run_method_sensitivity(panel: pd.DataFrame, market: MarketData,
                           model_result: dict, directions: pd.Series) -> dict:
    """打分方法敏感性：综合得分 vs LGBM直接预测（2025样本外）。"""
    s, e = cfg.ROBUSTNESS_YEARS["out_sample_2025"]
    # 综合得分
    p_comp = panel.copy()
    p_comp["score"] = composite_score(p_comp, model_result["importances"], directions,
                                       model_result["factor_cols"])
    sub_comp = p_comp[(p_comp["fwd_date"] >= pd.Timestamp(s)) & (p_comp["fwd_date"] <= pd.Timestamp(e))]
    r_comp = backtest(sub_comp, market, top_n=cfg.TOP_N_DEFAULT, name="composite_score")
    # LGBM直接预测
    p_dir = panel.copy()
    p_dir["score"] = predict_scores(model_result, p_dir)["score"]
    sub_dir = p_dir[(p_dir["fwd_date"] >= pd.Timestamp(s)) & (p_dir["fwd_date"] <= pd.Timestamp(e))]
    r_dir = backtest(sub_dir, market, top_n=cfg.TOP_N_DEFAULT, name="lgbm_direct")
    table = pd.DataFrame([_metrics_row(r_comp), _metrics_row(r_dir)])
    return {"composite": r_comp, "lgbm_direct": r_dir, "table": table}


def run_risk_control_spectrum(base_result: BacktestResult) -> dict:
    """模块4 风险控制：回撤触发减仓的收益-回撤权衡谱系。"""
    configs = [
        ("base_no_rc", None, None, None),
        ("trig-5%_exp50%", -0.05, 0.5, -0.02),
        ("trig-5%_exp40%", -0.05, 0.4, -0.02),
        ("trig-4%_exp50%", -0.04, 0.5, -0.02),
        ("trig-4%_exp40%", -0.04, 0.4, -0.02),
    ]
    rows = []
    results = {}
    for name, trig, exp, rec in configs:
        if trig is None:
            r = base_result
        else:
            r = apply_drawdown_control(base_result, trigger_dd=trig,
                                       de_risk_exposure=exp, recover_dd=rec,
                                       cost_rate=cfg.COST_RATE)
            r.name = name
        results[name] = r
        row = _metrics_row(r)
        row["config"] = name
        rows.append(row)
    table = pd.DataFrame(rows)[["config", "annualized_return", "max_drawdown", "sharpe",
                                "excess_cumulative", "total_turnover", "total_cost",
                                "target_return_ok", "target_dd_ok"]]
    return {"results": results, "table": table}


def run_all(scored_panel: pd.DataFrame, market: MarketData, panel: pd.DataFrame,
            model_result: dict, directions: pd.Series, base_2025: BacktestResult) -> dict:
    """执行全部鲁棒性与风险控制测试。"""
    return {
        "parameter": run_parameter_sensitivity(scored_panel, market,
                                               cfg.OUT_SAMPLE_START, cfg.OUT_SAMPLE_END),
        "time": run_time_sensitivity(scored_panel, market),
        "method": run_method_sensitivity(panel, market, model_result, directions),
        "risk_control": run_risk_control_spectrum(base_2025),
    }
