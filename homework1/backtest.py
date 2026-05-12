"""作业1：量化回测。"""

import numpy as np
import pandas as pd


def run_backtest(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
    initial_capital: float = 1_000_000,
    commission: float = 0.0003,
    buy_threshold: float = 0.005,
    sell_threshold: float = 0.005,
) -> dict:
    """执行量化回测。"""
    close_prices = df_test["close"].values
    n = len(y_pred)

    capital = initial_capital
    shares = 0
    daily_capital = []
    trades = 0
    position = False

    for i in range(n):
        pred = y_pred[i]
        price = close_prices[i]

        if pred > buy_threshold and not position:
            shares = capital * (1 - commission) / price
            capital = 0
            position = True
            trades += 1
        elif pred < -sell_threshold and position:
            capital = shares * price * (1 - commission)
            shares = 0
            position = False
            trades += 1

        total = capital + shares * price
        daily_capital.append(total)

    if position and n > 0:
        capital = shares * close_prices[-1] * (1 - commission)
        shares = 0
        daily_capital[-1] = capital

    daily_capital = np.array(daily_capital)

    total_return = (daily_capital[-1] / initial_capital - 1) if len(daily_capital) > 0 else 0
    peak = np.maximum.accumulate(daily_capital)
    drawdown = (peak - daily_capital) / peak
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

    actual_returns = df_test["label"].values[:n]
    pred_direction = np.sign(y_pred)
    actual_direction = np.sign(actual_returns)
    win_rate = np.mean(pred_direction == actual_direction) if n > 0 else 0

    return {
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "trades": trades,
        "daily_capital": daily_capital,
    }
