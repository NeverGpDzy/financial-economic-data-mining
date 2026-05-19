"""Buy-and-hold backtest for the selected high-alpha stock."""

from __future__ import annotations

import pandas as pd


def buy_and_hold_curve(
    stock_df: pd.DataFrame,
    market_df: pd.DataFrame,
    initial_capital: float,
) -> pd.DataFrame:
    """Compare a full-position stock holding with the HS300 benchmark."""
    aligned = pd.DataFrame(
        {
            "stock_close": stock_df["close"],
            "market_close": market_df["close"],
        }
    ).dropna()

    stock_cum = aligned["stock_close"] / aligned["stock_close"].iloc[0]
    market_cum = aligned["market_close"] / aligned["market_close"].iloc[0]

    out = pd.DataFrame(index=aligned.index)
    out["stock_value"] = initial_capital * stock_cum
    out["market_value"] = initial_capital * market_cum
    out["stock_cum_return"] = stock_cum - 1
    out["market_cum_return"] = market_cum - 1
    return out


def max_drawdown(cum_return: pd.Series) -> float:
    wealth = 1 + cum_return
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1
    return float(drawdown.min())


def backtest_metrics(curve: pd.DataFrame) -> dict[str, float]:
    """Calculate cumulative return, max drawdown and Calmar ratio."""
    stock_total = float(curve["stock_cum_return"].iloc[-1])
    market_total = float(curve["market_cum_return"].iloc[-1])
    stock_mdd = max_drawdown(curve["stock_cum_return"])
    market_mdd = max_drawdown(curve["market_cum_return"])

    return {
        "stock_total_return": stock_total,
        "market_total_return": market_total,
        "excess_total_return": stock_total - market_total,
        "stock_max_drawdown": stock_mdd,
        "market_max_drawdown": market_mdd,
        "stock_calmar": stock_total / abs(stock_mdd) if stock_mdd != 0 else float("inf"),
        "market_calmar": market_total / abs(market_mdd) if market_mdd != 0 else float("inf"),
        "stock_final_value": float(curve["stock_value"].iloc[-1]),
        "market_final_value": float(curve["market_value"].iloc[-1]),
    }


def run_horizon_backtests(
    stock_df: pd.DataFrame,
    market_df: pd.DataFrame,
    initial_capital: float,
    horizons: list[tuple[str, str | None]],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run buy-and-hold tests from the same start date to multiple end dates."""
    rows = []
    curves = {}

    for label, end_date in horizons:
        if end_date is None:
            stock_period = stock_df
            market_period = market_df
        else:
            end_ts = pd.Timestamp(end_date)
            stock_period = stock_df.loc[:end_ts]
            market_period = market_df.loc[:end_ts]

        curve = buy_and_hold_curve(stock_period, market_period, initial_capital)
        metrics = backtest_metrics(curve)
        curves[label] = curve

        rows.append(
            {
                "period": label,
                "start_date": curve.index[0].date().isoformat(),
                "end_date": curve.index[-1].date().isoformat(),
                "trading_days": len(curve),
                **metrics,
            }
        )

    return pd.DataFrame(rows), curves
