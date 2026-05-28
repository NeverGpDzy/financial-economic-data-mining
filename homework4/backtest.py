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
    """Out-of-sample monthly-rebalanced backtest using return data.

    Key principle: At end of month t, use Factors(t) to predict Return(t+1).
    Select stocks at t → hold through t+1 → realize P&L at t+1.

    Uses monthly returns directly (no price tracking needed).
    """
    months = sorted(test_df["date"].unique())
    if len(months) < 2:
        raise ValueError("Need at least 2 months for backtest.")

    nav = init_capital
    nav_history = [{"date": months[0], "nav": nav, "selected": []}]
    trade_records = []

    # For each month except the last, select stocks and compute next month return
    for mi in range(len(months) - 1):
        month = months[mi]
        next_month = months[mi + 1]

        # Step 1: Score and select stocks for this month
        sub = test_df[test_df["date"] == month]
        ranked = select_top_stocks(sub, score_func, top_n=top_n)
        selected = _pick_tradeable(ranked, top_n)

        if not selected:
            # No tradeable stocks, keep cash
            trade_records.append({
                "month": str(month)[:10],
                "holdings": "空仓",
                "n_holdings": 0,
                "month_return": 0.0,
            })
            nav_history.append({"date": next_month, "nav": nav, "selected": []})
            continue

        # Step 2: Get next month's returns for selected stocks
        next_sub = test_df[(test_df["date"] == next_month) & (test_df["code"].isin(selected))]
        if next_sub.empty:
            # No return data, keep current nav
            trade_records.append({
                "month": str(month)[:10],
                "holdings": ", ".join(selected),
                "n_holdings": len(selected),
                "month_return": 0.0,
            })
            nav_history.append({"date": next_month, "nav": nav, "selected": selected})
            continue

        # Step 3: Compute portfolio return (equal weight)
        stock_returns = next_sub.set_index("code")["return"]
        # Equal weight portfolio return
        port_return = stock_returns.mean()

        # Step 4: Deduct transaction costs (buy + sell)
        # Assume full turnover each month
        port_return_after_fee = port_return - 2 * fee

        # Step 5: Update NAV
        nav = nav * (1 + port_return_after_fee)

        trade_records.append({
            "month": str(month)[:10],
            "holdings": ", ".join(selected),
            "n_holdings": len(selected),
            "month_return": float(port_return_after_fee),
        })
        nav_history.append({"date": next_month, "nav": nav, "selected": selected})

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
