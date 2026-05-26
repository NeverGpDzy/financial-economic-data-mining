"""Module 5: Cross-sectional stock selection and out-of-sample backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd


def standardize_test_factors(
    test_df: pd.DataFrame, train_df: pd.DataFrame, valid_factors: list[str],
) -> pd.DataFrame:
    """Standardize test-set factors: cross-sectional Z-Score per month.

    Uses monthly cross-sectional mean/std, consistent with training
    standardization approach (factor_standardize in models.py).
    """
    out = test_df.copy()
    for f in valid_factors:
        std_name = f"{f}_std"
        out[std_name] = out.groupby("date")[f].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
        )
        out[std_name] = out[std_name].fillna(0.0)
    return out


def select_top_stocks(
    month_df: pd.DataFrame, score_func: callable, top_n: int = 3,
) -> pd.DataFrame:
    """Score and rank stocks for a single month cross-section."""
    sub = month_df.copy()
    sub["score"] = sub.apply(score_func, axis=1)
    sub = sub.sort_values("score", ascending=False).reset_index(drop=True)
    sub["rank"] = range(1, len(sub) + 1)
    return sub


def run_backtest(
    test_df: pd.DataFrame,
    mkt_df: pd.DataFrame,
    score_func: callable,
    valid_factors: list[str],
    init_capital: float = 1_000_000.0,
    fee: float = 0.003,
    top_n: int = 3,
    max_single_weight: float = 0.40,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    """Out-of-sample monthly-rebalanced backtest with NO look-ahead bias.

    Key principle: At end of month t, use Factors(t) to predict Return(t+1).
    Select stocks at t's close → hold through t+1 → realize P&L at t+1 close.

    Shares are tracked explicitly to avoid fee double-counting.
    """
    months = sorted(test_df["date"].unique())
    if len(months) < 2:
        raise ValueError("Need at least 2 months for backtest.")

    # Pre-build price lookup
    price_map = {}
    for month in months:
        sub = test_df[test_df["date"] == month]
        price_map[month] = dict(zip(sub["code"], sub["close"]))

    cash = init_capital
    # holdings: {code: {"shares": int}}  — shares held
    holdings: dict[str, dict] = {}
    nav_history = []
    trade_records = []

    # --- Month 0: initial entry ---
    month0 = months[0]
    sub0 = test_df[test_df["date"] == month0]
    ranked0 = select_top_stocks(sub0, score_func, top_n=top_n)
    selected0 = _pick_tradeable(ranked0, top_n)

    if not selected0:
        raise RuntimeError("First month has no tradeable stocks.")

    cash, holdings = _buy_stocks(selected0, price_map[month0], cash, fee, max_single_weight)
    port_value = cash + _holdings_value(holdings, price_map[month0])
    nav_history.append({"date": month0, "nav": port_value, "selected": selected0})

    # --- Subsequent months ---
    for mi in range(1, len(months)):
        month = months[mi]

        # Step 1: Mark-to-market holdings at this month's prices
        port_value = cash + _holdings_value(holdings, price_map[month])

        # Step 2: Compute month return and record
        nav = port_value
        month_return = (nav / nav_history[-1]["nav"] - 1) if nav_history[-1]["nav"] > 0 else 0.0

        # Step 3: Sell all holdings at this month's close price (with fee)
        sell_proceeds = 0.0
        for code in list(holdings.keys()):
            price = price_map[month].get(code, np.nan)
            if not pd.isna(price) and price > 0:
                shares = holdings[code]["shares"]
                sell_proceeds += shares * price * (1 - fee)
            del holdings[code]
        cash = sell_proceeds

        # Step 4: Score and select for next month
        sub = test_df[test_df["date"] == month]
        ranked = select_top_stocks(sub, score_func, top_n=top_n)
        selected = _pick_tradeable(ranked, top_n)

        if selected:
            cash, holdings = _buy_stocks(selected, price_map[month], cash, fee, max_single_weight)
            trade_records.append({
                "month": str(month),
                "holdings": ", ".join(selected),
                "n_holdings": len(selected),
                "month_return": float(month_return),
            })
        else:
            cash = nav
            holdings = {}
            trade_records.append({
                "month": str(month),
                "holdings": "空仓",
                "n_holdings": 0,
                "month_return": float(month_return),
            })

        port_value = cash + _holdings_value(holdings, price_map[month])
        nav_history.append({
            "date": month,
            "nav": port_value,
            "selected": selected if selected else [],
        })

    # Build result DataFrame
    result = pd.DataFrame(nav_history)
    result["date"] = pd.to_datetime(result["date"].astype(str))
    result = result.set_index("date").sort_index()

    # Market comparison
    mkt_close = mkt_df["close"]
    mkt_aligned = mkt_close.reindex(result.index, method="ffill").ffill().bfill()
    if not mkt_aligned.empty and mkt_aligned.iloc[0] > 0:
        result["market_nav"] = (mkt_aligned / mkt_aligned.iloc[0]) * init_capital
    else:
        result["market_nav"] = float(init_capital)

    # Performance metrics
    nav_series = result["nav"]
    total_return = nav_series.iloc[-1] / init_capital - 1
    n_years = (nav_series.index[-1] - nav_series.index[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    peak = nav_series.cummax()
    drawdown = (nav_series - peak) / peak
    max_dd = drawdown.min()

    monthly_rets = nav_series.pct_change().dropna()
    win_rate = (monthly_rets > 0).mean()

    mkt_total_return = result["market_nav"].iloc[-1] / init_capital - 1
    mkt_annual_return = (1 + mkt_total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    excess_return = total_return - mkt_total_return

    mkt_peak = result["market_nav"].cummax()
    mkt_dd = (result["market_nav"] - mkt_peak) / mkt_peak
    mkt_max_dd = mkt_dd.min()

    excess_monthly = monthly_rets - 0.0015
    sharpe = (excess_monthly.mean() / excess_monthly.std()) * np.sqrt(12) if excess_monthly.std() > 0 else 0
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

    metrics = {
        "累计收益率": total_return,
        "年化收益率": annual_return,
        "最大回撤": max_dd,
        "月度胜率": win_rate,
        "超额收益(vs上证指数)": excess_return,
        "上证指数累计收益": mkt_total_return,
        "上证指数年化收益": mkt_annual_return,
        "上证指数最大回撤": mkt_max_dd,
        "夏普比率": sharpe,
        "卡玛比率": calmar,
        "调仓次数": len(trade_records),
        "回测月数": len(months),
    }

    return result, metrics, trade_records


def _pick_tradeable(ranked: pd.DataFrame, top_n: int) -> list[str]:
    """Pick top-N stocks from ranked DataFrame."""
    selected = []
    for _, row in ranked.iterrows():
        if len(selected) >= top_n:
            break
        selected.append(row["code"])
    return selected


def _buy_stocks(
    selected: list[str],
    prices: dict[str, float],
    cash: float,
    fee: float,
    max_single_weight: float,
) -> tuple[float, dict[str, dict]]:
    """Buy selected stocks with explicit share tracking.

    Returns (remaining_cash, holdings_dict).
    holdings_dict: {code: {"shares": int}}
    """
    holdings = {}
    n = len(selected)
    weight_per_stock = min(1.0 / n, max_single_weight)
    total_capital = cash

    for code in selected:
        price = prices.get(code, np.nan)
        if pd.isna(price) or price <= 0:
            continue

        alloc = total_capital * weight_per_stock
        shares = int(alloc / (price * (1 + fee))) if price > 0 else 0
        if shares > 0:
            cost = shares * price * (1 + fee)
            holdings[code] = {"shares": shares}
            cash -= cost

    return cash, holdings


def _holdings_value(holdings: dict[str, dict], prices: dict[str, float]) -> float:
    """Compute total market value of holdings at given prices."""
    total = 0.0
    for code, pos in holdings.items():
        price = prices.get(code, np.nan)
        if not pd.isna(price) and price > 0:
            total += pos["shares"] * price
    return total
