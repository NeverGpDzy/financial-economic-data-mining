"""PEG strategy backtest for Homework 3."""

from __future__ import annotations

import pandas as pd


def run_peg_backtest(
    stock_data: dict[str, pd.DataFrame],
    peg_df: pd.DataFrame,
    alpha_rank: dict[str, float],
    mkt_df: pd.DataFrame,
    initial_capital: float,
    commission: float,
    slippage: float,
    peg_buy: float,
    peg_sell: float,
) -> tuple[pd.DataFrame, dict]:
    """Run PEG-based trading strategy.

    Rules:
    - PEG < peg_buy: buy (if multiple, pick best alpha; if already holding, hold)
    - PEG > peg_sell: sell (if already in cash, hold)
    - Otherwise: maintain current position
    """
    dates = peg_df.index
    stocks = list(peg_df.columns)

    # Pre-compute daily data arrays
    stock_returns = {s: stock_data[s]["return"].reindex(dates).fillna(0) for s in stocks}
    stock_closes = {s: stock_data[s]["close"].reindex(dates) for s in stocks}
    mkt_close = mkt_df["close"].reindex(dates)

    # State
    cash = initial_capital
    shares = 0
    holding = None
    trade_count = 0
    holding_days = 0
    records = []

    for date in dates:
        # Determine action
        action = "hold"
        target = None

        # Check sell first
        if holding is not None:
            peg_val = peg_df.loc[date, holding] if pd.notna(peg_df.loc[date, holding]) else None
            if peg_val is not None and peg_val > peg_sell:
                action = "sell"

        # Check buy
        if action != "sell" and holding is None:
            candidates = []
            for s in stocks:
                peg_val = peg_df.loc[date, s] if pd.notna(peg_df.loc[date, s]) else None
                if peg_val is not None and peg_val < peg_buy:
                    candidates.append(s)
            if candidates:
                if len(candidates) == 1:
                    target = candidates[0]
                else:
                    target = min(candidates, key=lambda s: alpha_rank.get(s, 999))
                action = "buy"

        # Execute
        if action == "sell" and holding is not None:
            price = stock_closes[holding].loc[date]
            sell_cost = price * shares * (commission + slippage)
            cash = price * shares - sell_cost
            trade_count += 1
            holding = None
            shares = 0

        elif action == "buy" and target is not None:
            price = stock_closes[target].loc[date]
            buy_cost_rate = commission + slippage
            shares = int(cash / (price * (1 + buy_cost_rate)))
            if shares > 0:
                cost = price * shares * buy_cost_rate
                cash = cash - price * shares - cost
                holding = target
                trade_count += 1

        # Track holding days
        if holding is not None:
            holding_days += 1

        # Compute portfolio value
        if holding is not None:
            price = stock_closes[holding].loc[date]
            port_value = cash + price * shares
        else:
            port_value = cash

        records.append({
            "date": date,
            "portfolio_value": port_value,
            "action": action,
            "holding": holding if holding else "空仓",
            "cash": cash,
        })

    result = pd.DataFrame(records).set_index("date")

    # Add market cumulative return
    mkt = mkt_close.reindex(dates)
    mkt_cum = (mkt / mkt.iloc[0]) * initial_capital
    result["market_value"] = mkt_cum.values

    # Compute metrics
    result["cum_return"] = result["portfolio_value"] / initial_capital - 1
    result["market_cum_return"] = result["market_value"] / initial_capital - 1
    result["peak"] = result["portfolio_value"].cummax()
    result["drawdown"] = (result["portfolio_value"] - result["peak"]) / result["peak"]

    total_days = len(result)
    total_return = float(result["cum_return"].iloc[-1])
    annual_return = (1 + total_return) ** (250 / total_days) - 1 if total_days > 0 else 0
    max_dd = float(result["drawdown"].min())
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

    mkt_total = float(result["market_cum_return"].iloc[-1])
    mkt_annual = (1 + mkt_total) ** (250 / total_days) - 1 if total_days > 0 else 0
    mkt_peak = result["market_value"].cummax()
    mkt_dd = ((result["market_value"] - mkt_peak) / mkt_peak).min()
    mkt_calmar = mkt_annual / abs(mkt_dd) if mkt_dd != 0 else 0

    # Holding period stats
    total_holding_days = holding_days
    holding_pct = holding_days / total_days * 100 if total_days > 0 else 0

    metrics = {
        "strategy_total_return": total_return,
        "strategy_annual_return": annual_return,
        "strategy_max_drawdown": float(max_dd),
        "strategy_calmar": calmar,
        "market_total_return": mkt_total,
        "market_annual_return": mkt_annual,
        "market_max_drawdown": float(mkt_dd),
        "market_calmar": mkt_calmar,
        "trade_count": trade_count,
        "holding_days": total_holding_days,
        "holding_pct": holding_pct,
    }

    return result, metrics
