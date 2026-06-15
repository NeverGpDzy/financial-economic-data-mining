"""FCFF grouping and annual price backtests for Homework 6."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _assign_groups(one_year: pd.DataFrame, score_col: str) -> pd.Series:
    ordered = one_year.sort_values(score_col, ascending=False, na_position="last")
    splits = np.array_split(ordered.index.to_numpy(), 3)
    result = pd.Series(index=one_year.index, dtype=object)
    for label, idx in zip(config.GROUP_LABELS, splits):
        result.loc[idx] = label
    return result


def build_strategy_groups(scored_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    schemes = {
        "方案A_传统规则": "traditional_score",
        "方案B_LGBM打分": "ml_score",
    }
    for scheme, score_col in schemes.items():
        for year, group in scored_panel.groupby("year"):
            valid = group.dropna(subset=[score_col]).copy()
            if len(valid) < 9:
                continue
            valid["group"] = _assign_groups(valid, score_col)
            valid["scheme"] = scheme
            valid["score_col"] = score_col
            rows.append(valid)
    if not rows:
        raise ValueError("没有足够股票形成A/B/C三档分组")
    grouped = pd.concat(rows, ignore_index=True)
    return grouped


def fcff_group_backtest(strategy_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = strategy_panel.dropna(subset=["target_fcff_growth_3y_ann"]).copy()
    available["样本"] = np.where(available["year"] <= config.TRAIN_END_YEAR, "训练集", "测试集")
    yearly = (
        available.groupby(["scheme", "样本", "year", "group"], as_index=False)
        .agg(
            stock_count=("ts_code", "nunique"),
            fcff_growth_3y_ann_mean=("target_fcff_growth_3y_ann", "mean"),
            fcff_growth_1y_mean=("target_fcff_growth_1y", "mean"),
            raw_fcff_growth_3y_ann_mean=("fcff_growth_3y_ann", "mean"),
            raw_fcff_growth_1y_mean=("fcff_growth_1y", "mean"),
            avg_score=("ml_score", "mean"),
            traditional_score_mean=("traditional_score", "mean"),
        )
    )
    summary = (
        yearly.groupby(["scheme", "样本", "group"], as_index=False)
        .agg(
            years=("year", "nunique"),
            avg_stock_count=("stock_count", "mean"),
            fcff_growth_3y_ann_mean=("fcff_growth_3y_ann_mean", "mean"),
            fcff_growth_1y_mean=("fcff_growth_1y_mean", "mean"),
        )
    )

    diff_rows = []
    for (scheme, sample), sub in summary.groupby(["scheme", "样本"]):
        values = sub.set_index("group")["fcff_growth_3y_ann_mean"]
        diff_rows.append(
            {
                "scheme": scheme,
                "样本": sample,
                "A组3年FCFF增速均值": values.get("A", np.nan),
                "B组3年FCFF增速均值": values.get("B", np.nan),
                "C组3年FCFF增速均值": values.get("C", np.nan),
                "A-C差值": values.get("A", np.nan) - values.get("C", np.nan),
                "是否单调A>B>C": bool(
                    values.get("A", -np.inf) > values.get("B", np.inf)
                    > values.get("C", np.inf)
                ),
            }
        )
    diff_df = pd.DataFrame(diff_rows)
    summary = summary.merge(diff_df, on=["scheme", "样本"], how="left")
    return yearly, summary


def _price_metrics(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    rows = []
    mask = (df["trade_date"].dt.year >= start_year) & (df["trade_date"].dt.year <= end_year)
    for (scheme, group), sub in df[mask].groupby(["scheme", "group"]):
        sub = sub.sort_values("trade_date")
        if sub.empty:
            continue
        daily_ret = sub["daily_ret"].fillna(0)
        benchmark_ret = sub["benchmark_ret"].fillna(0)
        cumulative = (1 + daily_ret).prod() - 1
        benchmark_cumulative = (1 + benchmark_ret).prod() - 1
        years = max(len(sub) / config.TRADING_DAYS, 1 / config.TRADING_DAYS)
        annual = (1 + cumulative) ** (1 / years) - 1
        benchmark_annual = (1 + benchmark_cumulative) ** (1 / years) - 1
        nav = (1 + daily_ret).cumprod()
        drawdown = nav / nav.cummax() - 1
        rows.append(
            {
                "scheme": scheme,
                "group": group,
                "period": f"{start_year}-{end_year}",
                "交易日数": int(len(sub)),
                "累计收益率": cumulative,
                "年化收益率": annual,
                "基准累计收益率": benchmark_cumulative,
                "基准年化收益率": benchmark_annual,
                "超额收益": cumulative - benchmark_cumulative,
                "最大回撤": float(drawdown.min()),
                "年化波动率": float(daily_ret.std(ddof=0) * np.sqrt(config.TRADING_DAYS)),
                "夏普比率": float(
                    np.sqrt(config.TRADING_DAYS) * daily_ret.mean() / (daily_ret.std(ddof=0) + 1e-10)
                ),
            }
        )
    return pd.DataFrame(rows)


def price_group_backtest(
    strategy_panel: pd.DataFrame,
    stock_daily: pd.DataFrame,
    hs300: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdings = strategy_panel[["scheme", "year", "group", "ts_code"]].drop_duplicates().copy()
    stock = stock_daily.copy()
    stock["trade_date"] = pd.to_datetime(stock["trade_date"])
    stock["year"] = stock["trade_date"].dt.year
    hs300 = hs300[["trade_date", "benchmark_ret"]].drop_duplicates().copy()
    hs300["trade_date"] = pd.to_datetime(hs300["trade_date"])

    all_records = []
    annual_records = []
    previous_holdings: dict[tuple[str, str], set[str]] = {}

    for scheme in sorted(holdings["scheme"].unique()):
        for group in config.GROUP_LABELS:
            nav = config.INITIAL_CAPITAL
            prev_set: set[str] | None = None
            for hold_year in range(config.TEST_START_YEAR, config.DATA_END_YEAR + 1):
                score_year = hold_year - 1
                current = holdings[
                    (holdings["scheme"] == scheme)
                    & (holdings["group"] == group)
                    & (holdings["year"] == score_year)
                ]["ts_code"].tolist()
                if not current:
                    continue
                current_set = set(current)
                year_days = stock[stock["year"].eq(hold_year)][["trade_date"]].drop_duplicates()
                if year_days.empty:
                    continue

                pivot = (
                    stock[(stock["year"].eq(hold_year)) & (stock["ts_code"].isin(current))]
                    .pivot_table(index="trade_date", columns="ts_code", values="ret", aggfunc="mean")
                    .reindex(year_days["trade_date"])
                    .fillna(0.0)
                )
                daily_group_ret = pivot.mean(axis=1)
                if prev_set is None:
                    cost = config.COMMISSION_RATE
                else:
                    sold = prev_set - current_set
                    bought = current_set - prev_set
                    denom = max(len(current_set), 1)
                    cost = (len(sold) + len(bought)) / denom * config.COMMISSION_RATE
                prev_set = current_set

                year_start_nav = nav
                for i, (date, ret) in enumerate(daily_group_ret.items()):
                    net_ret = float(ret) - (cost if i == 0 else 0.0)
                    nav *= 1 + net_ret
                    all_records.append(
                        {
                            "trade_date": pd.Timestamp(date),
                            "scheme": scheme,
                            "group": group,
                            "hold_year": hold_year,
                            "score_year": score_year,
                            "holding_count": len(current),
                            "daily_ret": net_ret,
                            "nav": nav,
                        }
                    )
                annual_records.append(
                    {
                        "scheme": scheme,
                        "group": group,
                        "hold_year": hold_year,
                        "score_year": score_year,
                        "holding_count": len(current),
                        "annual_return": nav / year_start_nav - 1,
                        "turnover_cost": cost,
                        "holdings": ",".join(current),
                    }
                )
                previous_holdings[(scheme, group)] = current_set

    nav_df = pd.DataFrame(all_records)
    nav_df = nav_df.merge(hs300, on="trade_date", how="left")
    nav_df["benchmark_ret"] = nav_df["benchmark_ret"].fillna(0.0)
    # Recompute benchmark by date and merge back so every strategy shares the same curve.
    benchmark_curve = (
        nav_df[["trade_date", "benchmark_ret"]]
        .drop_duplicates()
        .sort_values("trade_date")
        .assign(benchmark_nav=lambda d: config.INITIAL_CAPITAL * (1 + d["benchmark_ret"]).cumprod())
    )
    nav_df = nav_df.merge(
        benchmark_curve[["trade_date", "benchmark_nav"]], on="trade_date", how="left"
    )
    nav_df["excess_nav"] = nav_df["nav"] - nav_df["benchmark_nav"]

    annual_df = pd.DataFrame(annual_records)
    metrics = _price_metrics(nav_df, config.TEST_START_YEAR, config.TEST_END_YEAR)
    return nav_df.sort_values(["scheme", "group", "trade_date"]), annual_df, metrics


def save_backtest_outputs(
    strategy_panel: pd.DataFrame,
    fcff_yearly: pd.DataFrame,
    fcff_summary: pd.DataFrame,
    price_nav: pd.DataFrame,
    annual_returns: pd.DataFrame,
    price_metrics: pd.DataFrame,
) -> None:
    out = config.OUTPUT_DIR
    strategy_panel.to_csv(out / "strategy_yearly_groups.csv", index=False, encoding="utf-8-sig")
    fcff_yearly.to_csv(out / "fcff_group_yearly.csv", index=False, encoding="utf-8-sig")
    fcff_summary.to_csv(out / "fcff_group_summary.csv", index=False, encoding="utf-8-sig")
    price_nav.to_csv(out / "price_nav_daily.csv", index=False, encoding="utf-8-sig")
    annual_returns.to_csv(out / "price_annual_returns.csv", index=False, encoding="utf-8-sig")
    price_metrics.to_csv(out / "price_metrics.csv", index=False, encoding="utf-8-sig")
