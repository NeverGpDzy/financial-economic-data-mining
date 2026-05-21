"""Data access for Homework 3: close prices, PE_TTM, profit growth."""

from __future__ import annotations

from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize


def _fetch_k_data(code: str, start: str, end: str, fields: str) -> pd.DataFrame:
    """Generic baostock daily data fetcher."""
    rs = bs.query_history_k_data_plus(
        code, fields,
        start_date=start, end_date=end,
        frequency="d", adjustflag="3",
    )
    if rs.error_code != "0":
        raise RuntimeError(f"Baostock query failed for {code}: {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    if df.empty:
        raise ValueError(f"No data for {code} from {start} to {end}.")
    df["date"] = pd.to_datetime(df["date"])
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date"]).set_index("date").sort_index()


def fetch_close_price(code: str, start: str, end: str) -> pd.DataFrame:
    return _fetch_k_data(code, start, end, "date,close")


def fetch_price_pe(code: str, start: str, end: str) -> pd.DataFrame:
    return _fetch_k_data(code, start, end, "date,close,peTTM")


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["raw_return"] = out["close"].pct_change()
    out = out.dropna(subset=["raw_return"])
    out["return"] = winsorize(out["raw_return"], limits=[0.01, 0.01]).filled()
    return out


def fetch_profit_growth(code: str, start_year: int, end_year: int) -> float:
    """Arithmetic average of annual YoY net-profit growth rates."""
    profits = []
    for year in range(start_year, end_year + 1):
        rs = bs.query_profit_data(code=code, year=year, quarter=4)
        if rs.error_code != "0":
            continue
        while rs.next():
            row = rs.get_row_data()
            fields = rs.fields
            np_val = row[fields.index("netProfit")]
            stat = row[fields.index("statDate")]
            profits.append({"year": year, "netProfit": float(np_val), "statDate": stat})
    if len(profits) < 2:
        return np.nan

    pdf = pd.DataFrame(profits).drop_duplicates(subset="year").sort_values("year")
    pdf = pdf.set_index("year")
    growth = pdf["netProfit"].pct_change().dropna()

    if (growth < 0).any():
        neg_years = [str(y) for y in growth[growth < 0].index]
        print(f"  [!] {code} 存在负增长率年份: {', '.join(neg_years)}")

    return float(growth.mean())


def _load_or_fetch_price_pe(code: str, start: str, end: str, cache: Path, refresh: bool) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{code.replace('.', '_')}_{start}_{end}_price_pe.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df = fetch_price_pe(code, start, end)
    df.to_csv(path, index_label="date")
    return df


def fetch_train_data(
    stocks: dict[str, str],
    market_code: str,
    start: str,
    end: str,
    data_dir: Path,
    start_year: int,
    end_year: int,
    refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """Fetch training data: prices + PE_TTM + profit growth. Returns (stock_data, growth_rates)."""
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"Baostock login failed: {login.error_msg}")

    try:
        cache = data_dir / "train"

        # Fetch market (close price only)
        mkt_path = cache / f"{market_code.replace('.', '_')}_{start}_{end}.csv"
        cache.mkdir(parents=True, exist_ok=True)
        if mkt_path.exists() and not refresh:
            mkt_df = pd.read_csv(mkt_path, parse_dates=["date"]).set_index("date").sort_index()
        else:
            mkt_df = fetch_close_price(market_code, start, end)
            mkt_df.to_csv(mkt_path, index_label="date")
        mkt_df = add_returns(mkt_df)
        mkt_ret = mkt_df[["return"]].rename(columns={"return": "mkt_return"})

        # Fetch stocks (close price + PE_TTM)
        stock_data = {}
        for name, code in stocks.items():
            df = _load_or_fetch_price_pe(code, start, end, cache, refresh)
            df = add_returns(df)
            stock_data[name] = df

        # Merge
        for name in stock_data:
            sd = stock_data[name][["close", "return", "peTTM"]]
            merged = sd.join(mkt_ret, how="inner").dropna()
            # add_returns() already winsorized the returns, no need to winsorize again
            merged["stock_return"] = merged["return"]
            merged["mkt_return_w"] = winsorize(merged["mkt_return"].values, limits=[0.01, 0.01]).filled()
            # Winsorize PE_TTM per assignment requirement
            pe_raw = merged["peTTM"].values
            pe_w = winsorize(pe_raw, limits=[0.01, 0.01]).filled()
            merged["peTTM_w"] = pe_w
            pe_min, pe_max = float(pe_w.min()), float(pe_w.max())
            merged["peTTM_norm"] = (pe_w - pe_min) / (pe_max - pe_min) if pe_max > pe_min else 0.5
            merged["pe_ttm_min"] = pe_min
            merged["pe_ttm_max"] = pe_max
            stock_data[name] = merged

        # Profit growth
        print("获取利润增长率数据...")
        growth_rates = {}
        for name, code in stocks.items():
            g = fetch_profit_growth(code, start_year, end_year)
            growth_rates[name] = g
            print(f"  {name} ({code}): 平均增长率 = {g:.4f}" if not np.isnan(g) else f"  {name} ({code}): 无数据")

        return stock_data, growth_rates
    finally:
        bs.logout()


def fetch_backtest_data(
    stocks: dict[str, str],
    market_code: str,
    start: str,
    end: str,
    data_dir: Path,
    pe_bounds: dict[str, tuple[float, float]],
    growth_rates: dict[str, float],
    alpha_rank: dict[str, float],
    refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, float]]:
    """Fetch backtest data. Returns (stock_data, peg_df, alpha_rank)."""
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"Baostock login failed: {login.error_msg}")
    try:
        cache = data_dir / "backtest"
        cache.mkdir(parents=True, exist_ok=True)

        # Market
        mkt_path = cache / f"{market_code.replace('.', '_')}_{start}_{end}.csv"
        if mkt_path.exists() and not refresh:
            mkt_df = pd.read_csv(mkt_path, parse_dates=["date"]).set_index("date").sort_index()
        else:
            mkt_df = fetch_close_price(market_code, start, end)
            mkt_df.to_csv(mkt_path, index_label="date")
        mkt_df = add_returns(mkt_df)

        # Stocks
        stock_data = {}
        for name, code in stocks.items():
            df = _load_or_fetch_price_pe(code, start, end, cache, refresh)
            df = add_returns(df)
            stock_data[name] = df

        # Normalize PE_TTM using training bounds
        for name in stock_data:
            pe = stock_data[name]["peTTM"]
            pe_min, pe_max = pe_bounds[name]
            stock_data[name]["peTTM_norm"] = (pe - pe_min) / (pe_max - pe_min) if pe_max > pe_min else 0.5

        # PEG
        peg_df = pd.DataFrame({
            name: stock_data[name]["peTTM_norm"] / growth_rates[name]
            for name in stocks if not np.isnan(growth_rates[name])
        })
        peg_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        return stock_data, peg_df, alpha_rank
    finally:
        bs.logout()
