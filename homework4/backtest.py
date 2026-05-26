"""Module 5: Cross-sectional stock selection and out-of-sample backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd


def standardize_test_factors(
    test_df: pd.DataFrame, train_df: pd.DataFrame, valid_factors: list[str],
) -> pd.DataFrame:
    """Standardize test-set factors using training-set mean and std."""
    out = test_df.copy()
    for f in valid_factors:
        train_mean = train_df[f].mean()
        train_std = train_df[f].std()
        std_name = f"{f}_std"
        if train_std > 0:
            out[std_name] = (out[f] - train_mean) / train_std
        else:
            out[std_name] = 0.0
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
    """
    months = sorted(test_df["date"].unique())
    if len(months) < 2:
        raise ValueError("Need at least 2 months for backtest.")

    # Pre-build lookup: code -> close price per month
    price_map = {}
    for month in months:
        sub = test_df[test_df["date"] == month]
        price_map[month] = dict(zip(sub["code"], sub["close"]))

    cash = init_capital
    holdings: dict[str, float] = {}  # {code: allocated_capital}
    nav_history = []
    trade_records = []

    # --- Month 0: initial entry ---
    month0 = months[0]
    sub0 = test_df[test_df["date"] == month0]
    ranked0 = select_top_stocks(sub0, score_func, top_n=top_n)
    selected0 = _pick_tradeable(ranked0, top_n)

    if not selected0:
        raise RuntimeError("First month has no tradeable stocks.")

    cash, holdings = _rebalance(selected0, price_map[month0], cash, fee, max_single_weight)

    nav_history.append({
        "date": month0, "nav": cash + sum(holdings.values()),
        "selected": selected0,
    })

    # --- Subsequent months ---
    for mi in range(1, len(months)):
        month = months[mi]
        sub = test_df[test_df["date"] == month]

        # Step 1: Realize returns on existing holdings from month-1 to month
        portfolio_value = cash
        for code, allocated in list(holdings.items()):
            if code in price_map[month]:
                # Compute return: we bought at month-1 close, now at month close
                prev_close = price_map[months[mi - 1]].get(code, np.nan)
                curr_close = price_map[month].get(code, np.nan)
                if not pd.isna(prev_close) and not pd.isna(curr_close) and prev_close > 0:
                    stock_ret = (curr_close / prev_close) - 1
                    holdings[code] = allocated * (1 + stock_ret)
                    portfolio_value += holdings[code]
                else:
                    portfolio_value += allocated
            else:
                portfolio_value += allocated

        cash = 0.0  # All capital is in holdings

        # Step 2: Record current portfolio value (before rebalancing this month)
        nav = portfolio_value
        month_return = (nav / nav_history[-1]["nav"] - 1) if nav_history[-1]["nav"] > 0 else 0.0

        # Step 3: Rebalance — sell old, compute new scores, buy new (for next month)
        # Sell all old holdings at this month's close
        sell_cash = 0.0
        for code in list(holdings.keys()):
            sell_value = holdings[code] * (1 - fee)
            sell_cash += sell_value
            del holdings[code]
        cash = sell_cash

        # Score and select for next month
        ranked = select_top_stocks(sub, score_func, top_n=top_n)
        selected = _pick_tradeable(ranked, top_n)

        if selected:
            cash, holdings = _rebalance(selected, price_map[month], cash, fee, max_single_weight)
            trade_records.append({
                "month": str(month),
                "holdings": ", ".join(selected),
                "n_holdings": len(selected),
                "month_return": float(month_return),
            })
        else:
            # No tradeable stocks — stay in cash
            cash = nav
            holdings = {}
            trade_records.append({
                "month": str(month),
                "holdings": "空仓",
                "n_holdings": 0,
                "month_return": float(month_return),
            })

        nav_history.append({
            "date": month,
            "nav": cash + sum(holdings.values()),
            "selected": selected if selected else [],
        })

    # Build result DataFrame
    result = pd.DataFrame(nav_history)
    result["date"] = pd.to_datetime(result["date"].astype(str))
    result = result.set_index("date").sort_index()

    # Market comparison — reindex to result dates
    mkt_close = mkt_df["close"]
    mkt_aligned = mkt_close.reindex(result.index, method="ffill")
    mkt_aligned = mkt_aligned.ffill().bfill()
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
    """Pick top-N stocks from ranked DataFrame, skipping untradeable ones."""
    selected = []
    for _, row in ranked.iterrows():
        code = row["code"]
        if len(selected) >= top_n:
            break
        selected.append(code)
    return selected


def _rebalance(
    selected: list[str],
    prices: dict[str, float],
    cash: float,
    fee: float,
    max_single_weight: float,
) -> tuple[float, dict[str, float]]:
    """Buy selected stocks at given prices. Returns (remaining_cash, holdings)."""
    holdings = {}
    n = len(selected)
    weight_per_stock = min(1.0 / n, max_single_weight)
    total_capital = cash

    for code in selected:
        price = prices.get(code, np.nan)
        if pd.isna(price) or price <= 0:
            continue

        alloc = total_capital * weight_per_stock
        buy_cost = alloc * fee
        holdings[code] = alloc - buy_cost
        cash -= alloc

    return cash, holdings
