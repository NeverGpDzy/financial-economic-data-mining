"""量化回测模块。"""

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
    """执行量化回测。

    规则：
      - 预测次日涨幅 > buy_threshold → 满仓买入
      - 预测次日跌幅 > sell_threshold → 全部卖出
      - 手续费：买入/卖出各收一次（双向）

    Args:
        df_test: 测试集 DataFrame，需包含 close 列和 price_return 列
        y_pred: 模型预测的次日价格收益率
        initial_capital: 初始资金，默认 100 万
        commission: 手续费率，默认 0.03%
        buy_threshold: 买入阈值，默认 0.5%
        sell_threshold: 卖出阈值，默认 0.5%

    Returns:
        字典，包含回测指标和每日资金曲线
    """
    close_prices = df_test["close"].values
    n = len(y_pred)

    capital = initial_capital
    shares = 0
    daily_capital = []
    trades = 0
    wins = 0
    position = False  # 是否持仓

    for i in range(n):
        pred = y_pred[i]
        price = close_prices[i]

        # 买入信号
        if pred > buy_threshold and not position:
            shares = capital * (1 - commission) / price
            capital = 0
            position = True
            trades += 1
        # 卖出信号
        elif pred < -sell_threshold and position:
            capital = shares * price * (1 - commission)
            shares = 0
            position = False
            trades += 1

        # 当日总资产
        total = capital + shares * price
        daily_capital.append(total)

    # 最终清仓结算
    if position and n > 0:
        capital = shares * close_prices[-1] * (1 - commission)
        shares = 0
        daily_capital[-1] = capital

    daily_capital = np.array(daily_capital)

    # 计算指标
    total_return = (daily_capital[-1] / initial_capital - 1) if len(daily_capital) > 0 else 0

    # 最大回撤
    peak = np.maximum.accumulate(daily_capital)
    drawdown = (peak - daily_capital) / peak
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

    # 胜率：预测方向与实际方向一致的比例
    actual_returns = df_test["price_return"].values[:n]
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
