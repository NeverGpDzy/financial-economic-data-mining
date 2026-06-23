"""Backtest and risk-control analysis for Homework 9."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from . import config


@dataclass(frozen=True)
class Calibration:
    mu: float
    sigma: float
    adf_stat: float
    p_value: float
    critical_5pct: float
    is_stationary: bool
    obs: int


def adf_test(series: pd.Series) -> dict:
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.nunique() < 2 or len(clean) < 30:
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


def calibrate_spread(spread: pd.Series) -> Calibration:
    clean = pd.Series(spread).replace([np.inf, -np.inf], np.nan).dropna()
    adf = adf_test(clean)
    sigma = float(clean.std(ddof=1))
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("价差标准差无效，无法生成交易阈值")
    return Calibration(
        mu=float(clean.mean()),
        sigma=sigma,
        adf_stat=adf["adf_stat"],
        p_value=adf["p_value"],
        critical_5pct=adf["critical_5pct"],
        is_stationary=adf["is_stationary"],
        obs=int(len(clean)),
    )


def build_spread_panel(close: pd.DataFrame) -> pd.DataFrame:
    """Build log prices, spread and daily pair returns."""
    panel = close.copy()
    panel["log_maotai"] = np.log(panel[config.MAOTAI])
    panel["log_laojiao"] = np.log(panel[config.LAOJIAO])
    panel["spread"] = panel["log_maotai"] - panel["log_laojiao"]
    panel["maotai_return"] = panel[config.MAOTAI].pct_change().fillna(0.0)
    panel["laojiao_return"] = panel[config.LAOJIAO].pct_change().fillna(0.0)
    panel["pair_return_raw_200pct_gross"] = panel["maotai_return"] - panel["laojiao_return"]
    panel["pair_return"] = config.PAIR_LEG_WEIGHT * panel["pair_return_raw_200pct_gross"]
    return panel


def backtest_start(panel: pd.DataFrame) -> pd.Timestamp:
    train_end = pd.Timestamp(config.INITIAL_TRAIN_END)
    eligible = panel.index[panel.index > train_end]
    if len(eligible) == 0:
        raise ValueError("回测期没有可用交易日")
    return pd.Timestamp(eligible[0])


def initial_calibration(panel: pd.DataFrame) -> Calibration:
    train = panel.loc[panel.index <= pd.Timestamp(config.INITIAL_TRAIN_END), "spread"]
    return calibrate_spread(train)


def _half_year_window_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    first_month = 1 if start.month <= 6 else 7
    first = pd.Timestamp(year=start.year, month=first_month, day=1)
    starts = list(pd.date_range(first, end, freq=f"{config.ROLLING_RESET_MONTHS}MS"))
    return [pd.Timestamp(s) for s in starts if s <= end]


def build_static_regime(panel: pd.DataFrame, calibration: Calibration) -> pd.DataFrame:
    start = backtest_start(panel)
    dates = panel.loc[start : config.BACKTEST_END].index
    regime = pd.DataFrame(index=dates)
    regime["window"] = "静态中枢"
    regime["calibration_start"] = panel.index.min()
    regime["calibration_end"] = pd.Timestamp(config.INITIAL_TRAIN_END)
    regime["mu"] = calibration.mu
    regime["sigma"] = calibration.sigma
    regime["upper"] = calibration.mu + config.THRESHOLD_MULTIPLIER * calibration.sigma
    regime["lower"] = calibration.mu - config.THRESHOLD_MULTIPLIER * calibration.sigma
    regime["adf_stat"] = calibration.adf_stat
    regime["p_value"] = calibration.p_value
    regime["critical_5pct"] = calibration.critical_5pct
    regime["can_trade"] = True
    regime["reason"] = "固定2015-2017期初中枢"
    return regime


def build_dynamic_regime(panel: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a half-year reset regime.

    ``mode="instruction"`` follows the assignment literally: each half-year
    window is tested and calibrated with that same window's spread. ``mode=
    "causal"`` estimates each future half-year from the latest three years
    available before the window starts.
    """
    if mode not in {"instruction", "causal"}:
        raise ValueError("mode必须是 instruction 或 causal")

    start = backtest_start(panel)
    end = pd.Timestamp(config.BACKTEST_END)
    starts = _half_year_window_starts(start, end)
    regime_parts = []
    diagnostics = []

    for i, nominal_start in enumerate(starts):
        trade_start = max(nominal_start, start)
        next_start = starts[i + 1] if i + 1 < len(starts) else end + pd.Timedelta(days=1)
        trade_end = min(next_start - pd.Timedelta(days=1), end)
        trade_dates = panel.loc[trade_start:trade_end].index
        if trade_dates.empty:
            continue

        if mode == "instruction":
            calibration_start = trade_dates[0]
            calibration_end = trade_dates[-1]
        else:
            calibration_end = trade_dates[0] - pd.Timedelta(days=1)
            calibration_start = calibration_end - pd.DateOffset(years=config.ROLLING_LOOKBACK_YEARS) + pd.Timedelta(days=1)
        train = panel.loc[calibration_start:calibration_end, "spread"]
        try:
            calibration = calibrate_spread(train)
            can_trade = calibration.is_stationary
            reason = "ADF通过，允许交易" if can_trade else "ADF未通过，风控空仓"
            mu = calibration.mu
            sigma = calibration.sigma
            adf_stat = calibration.adf_stat
            p_value = calibration.p_value
            critical_5pct = calibration.critical_5pct
            obs = calibration.obs
        except ValueError:
            can_trade = False
            reason = "样本不足，风控空仓"
            mu = sigma = adf_stat = p_value = critical_5pct = np.nan
            obs = int(train.dropna().shape[0])

        raw_upper = mu + config.THRESHOLD_MULTIPLIER * sigma if pd.notna(mu) and pd.notna(sigma) else np.nan
        raw_lower = mu - config.THRESHOLD_MULTIPLIER * sigma if pd.notna(mu) and pd.notna(sigma) else np.nan
        label = f"{trade_dates[0].date()}~{trade_dates[-1].date()}"
        part = pd.DataFrame(index=trade_dates)
        part["window"] = label
        part["mode"] = "教师指令版" if mode == "instruction" else "防未来函数版"
        part["calibration_start"] = pd.Timestamp(calibration_start)
        part["calibration_end"] = pd.Timestamp(calibration_end)
        part["mu"] = mu
        part["sigma"] = sigma
        part["raw_upper"] = raw_upper
        part["raw_lower"] = raw_lower
        part["upper"] = raw_upper if can_trade else np.nan
        part["lower"] = raw_lower if can_trade else np.nan
        part["adf_stat"] = adf_stat
        part["p_value"] = p_value
        part["critical_5pct"] = critical_5pct
        part["can_trade"] = can_trade
        part["reason"] = reason
        regime_parts.append(part)

        diagnostics.append(
            {
                "trade_window": label,
                "mode": "教师指令版" if mode == "instruction" else "防未来函数版",
                "calibration_start": pd.Timestamp(calibration_start).date().isoformat(),
                "calibration_end": pd.Timestamp(calibration_end).date().isoformat(),
                "obs": obs,
                "mu": mu,
                "sigma": sigma,
                "raw_upper": raw_upper,
                "raw_lower": raw_lower,
                "trade_upper": part["upper"].iloc[0],
                "trade_lower": part["lower"].iloc[0],
                "adf_stat": adf_stat,
                "p_value": p_value,
                "critical_5pct": critical_5pct,
                "can_trade": can_trade,
                "reason": reason,
            }
        )

    return pd.concat(regime_parts).sort_index(), pd.DataFrame(diagnostics)


def run_threshold_backtest(panel: pd.DataFrame, regime: pd.DataFrame, label: str) -> pd.DataFrame:
    """Run the threshold pair-trading strategy for a supplied regime table."""
    bt = panel.join(regime, how="inner").copy()
    positions = []
    signal = []
    position = 0

    for _, row in bt.iterrows():
        if not bool(row["can_trade"]) or pd.isna(row["mu"]) or pd.isna(row["sigma"]):
            next_position = 0
            signal_text = "风控空仓"
        elif row["spread"] > row["upper"]:
            next_position = -1
            signal_text = "上轨：做空价差"
        elif row["spread"] < row["lower"]:
            next_position = 1
            signal_text = "下轨：做多价差"
        elif position == 1 and row["spread"] >= row["mu"]:
            next_position = 0
            signal_text = "回归中枢：平多"
        elif position == -1 and row["spread"] <= row["mu"]:
            next_position = 0
            signal_text = "回归中枢：平空"
        else:
            next_position = position
            signal_text = "持仓/观望"

        positions.append(next_position)
        signal.append(signal_text)
        position = next_position

    bt["target_position"] = positions
    bt["executed_position"] = bt["target_position"].shift(1).fillna(0)
    bt["strategy_return"] = bt["executed_position"] * bt["pair_return"]
    bt["nav"] = (1 + bt["strategy_return"]).cumprod()
    bt["signal"] = signal
    bt["strategy"] = label
    bt["trade_flag"] = bt["target_position"].diff().fillna(bt["target_position"]).ne(0)
    return bt.reset_index(names="date")


def run_buy_and_hold(panel: pd.DataFrame, asset: str, label: str) -> pd.DataFrame:
    start = backtest_start(panel)
    bt = panel.loc[start : config.BACKTEST_END].copy()
    ret_col = "maotai_return" if asset == config.MAOTAI else "laojiao_return"
    bt["strategy_return"] = bt[ret_col]
    bt.iloc[0, bt.columns.get_loc("strategy_return")] = 0.0
    bt["nav"] = (1 + bt["strategy_return"]).cumprod()
    bt["target_position"] = 1
    bt["executed_position"] = 1
    bt["signal"] = f"长期持有{asset}"
    bt["strategy"] = label
    bt["trade_flag"] = False
    if len(bt) > 0:
        bt.iloc[0, bt.columns.get_loc("trade_flag")] = True
    return bt.reset_index(names="date")


def max_drawdown(nav: pd.Series) -> float:
    running_max = nav.cummax()
    drawdown = nav / running_max - 1
    return float(drawdown.min())


def summarize_backtest(bt: pd.DataFrame, label: str) -> dict:
    returns = bt["strategy_return"].fillna(0.0)
    nav = bt["nav"]
    n_days = int(len(bt))
    annual_return = float(nav.iloc[-1] ** (252 / max(n_days - 1, 1)) - 1) if n_days else 0.0
    volatility = float(returns.std(ddof=1) * np.sqrt(252)) if n_days > 1 else 0.0
    sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)) if returns.std(ddof=1) > 0 else 0.0
    return {
        "strategy": label,
        "start": pd.to_datetime(bt["date"].iloc[0]).date().isoformat(),
        "end": pd.to_datetime(bt["date"].iloc[-1]).date().isoformat(),
        "trading_days": n_days,
        "cumulative_return": float(nav.iloc[-1] - 1),
        "annual_return": annual_return,
        "max_drawdown": max_drawdown(nav),
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "trade_count": int(bt["trade_flag"].sum()),
        "active_days": int((bt["executed_position"] != 0).sum()),
        "cash_days": int((bt["executed_position"] == 0).sum()),
        "risk_off_days": int((bt.get("can_trade", pd.Series(True, index=bt.index)) == False).sum()),
    }


def analyze(close: pd.DataFrame) -> dict:
    panel = build_spread_panel(close)
    initial = initial_calibration(panel)
    static_regime = build_static_regime(panel, initial)
    dynamic_instruction_regime, dynamic_instruction_windows = build_dynamic_regime(panel, mode="instruction")
    dynamic_causal_regime, dynamic_causal_windows = build_dynamic_regime(panel, mode="causal")

    static_bt = run_threshold_backtest(panel, static_regime, "方案A：静态中枢无风控")
    dynamic_instruction_bt = run_threshold_backtest(
        panel,
        dynamic_instruction_regime,
        "方案B1：半年窗口ADF风控（教师指令版）",
    )
    dynamic_causal_bt = run_threshold_backtest(
        panel,
        dynamic_causal_regime,
        "方案B2：滚动ADF风控（防未来函数）",
    )
    hold_maotai = run_buy_and_hold(panel, config.MAOTAI, "方案C1：贵州茅台长期持有")
    hold_laojiao = run_buy_and_hold(panel, config.LAOJIAO, "方案C2：泸州老窖长期持有")

    metrics = pd.DataFrame(
        [
            summarize_backtest(static_bt, "方案A：静态中枢无风控"),
            summarize_backtest(dynamic_instruction_bt, "方案B1：半年窗口ADF风控（教师指令版）"),
            summarize_backtest(dynamic_causal_bt, "方案B2：滚动ADF风控（防未来函数）"),
            summarize_backtest(hold_maotai, "方案C1：贵州茅台长期持有"),
            summarize_backtest(hold_laojiao, "方案C2：泸州老窖长期持有"),
        ]
    )

    for windows in (dynamic_instruction_windows, dynamic_causal_windows):
        windows["mu_drift_vs_static"] = windows["mu"] - initial.mu
        windows["abs_mu_drift_vs_static"] = windows["mu_drift_vs_static"].abs()

    return {
        "panel": panel.reset_index(names="date"),
        "initial_calibration": initial,
        "static_regime": static_regime.reset_index(names="date"),
        "dynamic_instruction_regime": dynamic_instruction_regime.reset_index(names="date"),
        "dynamic_instruction_windows": dynamic_instruction_windows,
        "dynamic_causal_regime": dynamic_causal_regime.reset_index(names="date"),
        "dynamic_causal_windows": dynamic_causal_windows,
        "static_backtest": static_bt,
        "dynamic_instruction_backtest": dynamic_instruction_bt,
        "dynamic_causal_backtest": dynamic_causal_bt,
        "hold_maotai": hold_maotai,
        "hold_laojiao": hold_laojiao,
        "metrics": metrics,
    }
