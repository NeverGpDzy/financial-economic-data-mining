"""月度行业轮动回测引擎（作业 1.3.4 / 1.3.5 / 模块3）。

回测口径：
- 信号：每月末交易日生成（因子得分排序），无未来函数；
- 执行：行业指数仅有收盘价，故以月末收盘价执行（见 README 说明），等权配置 TopN；
- 持有：下一月全部交易日，日频复利计算 NAV；
- 成本：单边佣金万分之三，按 |Δ权重| 换手率计入；
- 风控：单行业仓位上限 30%（Top5 等权20%不触发；Top3 触发后超出部分留现金）；
- 基准：沪深300同期日频净值。

输出日频策略净值、基准净值、持仓、月度收益与核心指标。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as cfg
from .data_loader import MarketData


@dataclass
class BacktestResult:
    name: str
    nav: pd.Series                      # 策略日频净值（基期=1）
    benchmark_nav: pd.Series            # 沪深300日频净值（基期=1）
    holdings: pd.DataFrame              # 每月持仓：signal_date, industry, weight, score
    monthly_returns: pd.DataFrame       # 每月：signal_date, fwd_date, strategy_ret, benchmark_ret, turnover, cost
    metrics: dict = field(default_factory=dict)
    top_n: int = 5
    weighting: str = "equal"

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.metrics, index=[self.name]).T


def _select_weights(scored: pd.DataFrame, top_n: int, weighting: str,
                    cap: float) -> tuple[list[str], np.ndarray, float]:
    """给定一个月末截面得分，返回 (入选行业代码, 权重数组, 换手预留)。"""
    ranked = scored.sort_values("score", ascending=False).head(top_n)
    codes = ranked["industry"].tolist()
    n = len(codes)
    if n == 0:
        return [], np.array([]), 0.0
    if weighting == "score":
        s = ranked["score"].values.astype(float)
        s = s - s.min()
        if s.sum() <= 0:
            w = np.ones(n) / n
        else:
            w = s / s.sum()
    else:  # equal
        w = np.ones(n) / n
    # 单行业仓位上限
    w = np.minimum(w, cap)
    # 不再向上重分配（避免破坏 TopN 语义）；超出部分留现金
    return codes, w, float(w.sum())


def backtest(scored_panel: pd.DataFrame, market: MarketData,
             top_n: int = cfg.TOP_N_DEFAULT, weighting: str = cfg.WEIGHTING,
             cost_rate: float = cfg.COST_RATE, cap: float = cfg.SINGLE_INDUSTRY_CAP,
             name: str = "strategy") -> BacktestResult:
    """对带 score 列的面板执行月度轮动回测。

    scored_panel 需包含列：signal_date, industry, score, fwd_date（已按目标区间筛选）。
    """
    # 日频行业收益率与基准收益率
    ind_ret = market.ind_close.pct_change()
    hs_ret = market.hs_close.pct_change()
    dates = market.dates

    # 按信号日排序
    signal_dates = sorted(scored_panel["signal_date"].unique())

    daily_rets: list[tuple[pd.Timestamp, float, float]] = []  # (date, strat_gross, bench)
    holdings_rows = []
    monthly_rows = []
    prev_weights: dict[str, float] = {}

    for d in signal_dates:
        sub = scored_panel[scored_panel["signal_date"] == d]
        if sub.empty:
            continue
        d_next = sub["fwd_date"].iloc[0]
        codes, w, _ = _select_weights(sub, top_n, weighting, cap)
        if len(codes) == 0:
            continue
        new_weights = dict(zip(codes, w))

        # 持有期交易日：(d, d_next]
        hold_days = dates[(dates > d) & (dates <= d_next)]
        if len(hold_days) == 0:
            continue

        # 换手率与成本（首次建仓 prev_weights 为空 -> turnover=1）
        all_codes = set(new_weights) | set(prev_weights)
        turnover = sum(abs(new_weights.get(c, 0.0) - prev_weights.get(c, 0.0)) for c in all_codes)
        cost = cost_rate * turnover

        # 记录持仓
        for c, wi in new_weights.items():
            holdings_rows.append({
                "signal_date": d, "fwd_date": d_next, "industry": c,
                "weight": wi, "score": sub.set_index("industry").loc[c, "score"],
            })

        # 日频组合收益
        first = True
        period_strat = 1.0
        period_bench = 1.0
        for t in hold_days:
            gross = 0.0
            for c, wi in new_weights.items():
                r = ind_ret.loc[t, c] if c in ind_ret.columns else 0.0
                gross += wi * (r if not np.isnan(r) else 0.0)
            if first:
                net = (1.0 + gross) * (1.0 - cost) - 1.0  # 调仓日扣除成本
                first = False
            else:
                net = gross
            br = hs_ret.loc[t] if not np.isnan(hs_ret.loc[t]) else 0.0
            daily_rets.append((t, net, br))
            period_strat *= (1.0 + net)
            period_bench *= (1.0 + br)

        monthly_rows.append({
            "signal_date": d, "fwd_date": d_next,
            "strategy_ret": period_strat - 1.0,
            "benchmark_ret": period_bench - 1.0,
            "turnover": turnover, "cost": cost,
            "n_holdings": len(codes),
        })
        prev_weights = new_weights

    if not daily_rets:
        raise ValueError("回测无有效交易日，请检查区间与数据。")

    df = pd.DataFrame(daily_rets, columns=["date", "strategy_ret", "benchmark_ret"]).set_index("date").sort_index()
    nav = (1.0 + df["strategy_ret"]).cumprod()
    benchmark_nav = (1.0 + df["benchmark_ret"]).cumprod()

    holdings = pd.DataFrame(holdings_rows)
    monthly = pd.DataFrame(monthly_rows)

    metrics = compute_metrics(nav, benchmark_nav, df, monthly)
    return BacktestResult(name=name, nav=nav, benchmark_nav=benchmark_nav,
                          holdings=holdings, monthly_returns=monthly,
                          metrics=metrics, top_n=top_n, weighting=weighting)


def apply_drawdown_control(result: BacktestResult, trigger_dd: float = -0.06,
                           de_risk_exposure: float = 0.5, recover_dd: float = -0.03,
                           cost_rate: float = cfg.COST_RATE) -> BacktestResult:
    """模块4 交易风控：回撤触发减仓。

    当策略从净值高点的回撤 <= trigger_dd 时，将仓位降至 de_risk_exposure（余下持现金）；
    当回撤恢复到 recover_dd 之上时恢复满仓。仅使用历史净值状态，无未来函数。
    暴露调整产生的额外换手按 cost_rate 扣费。
    返回新的风险控制后回测结果（基准不变）。
    """
    daily_ret = result.nav.pct_change().fillna(0.0)
    new_rets = []
    nav_vals = []
    exposures = []
    extra_turnovers = []
    extra_costs = []
    cur = 1.0
    peak = 1.0
    de_risked = False
    prev_exposure = 1.0
    for t, r in daily_ret.items():
        exposure = de_risk_exposure if de_risked else 1.0
        extra_turnover = abs(exposure - prev_exposure)
        extra_cost = cost_rate * extra_turnover
        nr = exposure * r - extra_cost
        cur *= (1.0 + nr)
        peak = max(peak, cur)
        dd = cur / peak - 1.0
        if dd <= trigger_dd:
            de_risked = True
        elif dd >= recover_dd:
            de_risked = False
        new_rets.append(nr)
        nav_vals.append(cur)
        exposures.append(exposure)
        extra_turnovers.append(extra_turnover)
        extra_costs.append(extra_cost)
        prev_exposure = exposure

    new_nav = pd.Series(nav_vals, index=daily_ret.index, name="nav_rc")
    new_daily = pd.DataFrame({
        "strategy_ret": new_rets,
        "exposure": exposures,
        "risk_control_turnover": extra_turnovers,
        "risk_control_cost": extra_costs,
    }, index=daily_ret.index)
    # 月度收益按原 holding period 聚合
    monthly = result.monthly_returns.copy()
    if not monthly.empty:
        new_monthly_ret = []
        extra_monthly_turnover = []
        extra_monthly_cost = []
        for _, row in monthly.iterrows():
            seg = new_daily.loc[(new_daily.index > row["signal_date"]) & (new_daily.index <= row["fwd_date"])]
            new_monthly_ret.append((1.0 + seg["strategy_ret"]).prod() - 1.0 if len(seg) else 0.0)
            extra_monthly_turnover.append(float(seg["risk_control_turnover"].sum()) if len(seg) else 0.0)
            extra_monthly_cost.append(float(seg["risk_control_cost"].sum()) if len(seg) else 0.0)
        monthly["strategy_ret"] = new_monthly_ret
        monthly["risk_control_turnover"] = extra_monthly_turnover
        monthly["risk_control_cost"] = extra_monthly_cost
        monthly["turnover"] = monthly["turnover"] + monthly["risk_control_turnover"]
        monthly["cost"] = monthly["cost"] + monthly["risk_control_cost"]
    metrics = compute_metrics(new_nav, result.benchmark_nav, new_daily, monthly)
    return BacktestResult(name=result.name + "_风控", nav=new_nav, benchmark_nav=result.benchmark_nav,
                          holdings=result.holdings, monthly_returns=monthly, metrics=metrics,
                          top_n=result.top_n, weighting=result.weighting)


def compute_metrics(nav: pd.Series, benchmark_nav: pd.Series,
                    daily_df: pd.DataFrame, monthly: pd.DataFrame) -> dict:
    """计算核心回测指标。"""
    n_days = len(nav)
    total_return = nav.iloc[-1] - 1.0
    ann_return = nav.iloc[-1] ** (cfg.ANN_TRADING_DAYS / n_days) - 1.0
    cummax = nav.cummax()
    max_dd = (nav / cummax - 1.0).min()
    daily_ret = daily_df["strategy_ret"]
    sharpe = (daily_ret.mean() / daily_ret.std(ddof=1) * np.sqrt(cfg.ANN_TRADING_DAYS)
              if daily_ret.std(ddof=1) > 0 else 0.0)
    win_rate = (daily_ret > 0).mean()

    bench_total = benchmark_nav.iloc[-1] - 1.0
    bench_ann = benchmark_nav.iloc[-1] ** (cfg.ANN_TRADING_DAYS / n_days) - 1.0
    excess_total = total_return - bench_total

    # 月度胜率（相对基准）
    if not monthly.empty:
        monthly_win = (monthly["strategy_ret"] > monthly["benchmark_ret"]).mean()
        monthly_excess = (monthly["strategy_ret"] - monthly["benchmark_ret"])
        monthly_ir = (monthly_excess.mean() / monthly_excess.std(ddof=1)
                      if monthly_excess.std(ddof=1) > 0 else 0.0)
    else:
        monthly_win = np.nan
        monthly_ir = np.nan

    # 达标判断
    ann_lo, ann_hi = cfg.TARGET_ANN_RETURN
    target_return_ok = ann_lo <= ann_return <= ann_hi
    target_dd_ok = max_dd >= -cfg.TARGET_MAX_DRAWDOWN  # 回撤不低于 -10%

    return {
        "n_days": int(n_days),
        "n_months": int(len(monthly)),
        "cumulative_return": total_return,
        "annualized_return": ann_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "daily_win_rate": win_rate,
        "benchmark_cumulative": bench_total,
        "benchmark_annualized": bench_ann,
        "excess_cumulative": excess_total,
        "monthly_win_rate_vs_bench": monthly_win,
        "monthly_ir_vs_bench": monthly_ir,
        "total_turnover": float(monthly["turnover"].sum()) if not monthly.empty else 0.0,
        "total_cost": float(monthly["cost"].sum()) if not monthly.empty else 0.0,
        "target_return_ok": bool(target_return_ok),
        "target_dd_ok": bool(target_dd_ok),
        "target_annualized_range": f"{ann_lo:.0%}-{ann_hi:.0%}",
        "target_max_drawdown": -cfg.TARGET_MAX_DRAWDOWN,
    }
