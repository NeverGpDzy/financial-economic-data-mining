"""Data fetching and caching for Homework 7."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def baostock_to_display(code: str) -> str:
    exchange, symbol = code.split(".")
    return f"{symbol}.{exchange.upper()}"


def fetch_daily_close(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily close prices from Baostock."""
    import baostock as bs

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"Baostock登录失败: {login.error_msg}")

    try:
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,close,volume,amount",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3",
        )
        if rs.error_code != "0":
            raise RuntimeError(f"Baostock查询失败 {code}: {rs.error_msg}")

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
    finally:
        bs.logout()

    if df.empty:
        raise RuntimeError(f"Baostock未返回行情数据: {code}")

    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return (
        df.dropna(subset=["date", "close"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_or_fetch_stock(
    name: str,
    code: str,
    start_date: str = config.START_DATE,
    end_date: str = config.END_DATE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load one stock from local cache, or fetch it from Baostock."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RAW_DIR / f"{baostock_to_display(code)}_{start_date}_{end_date}.csv"

    if path.exists() and not refresh:
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        df = fetch_daily_close(code, start_date, end_date)
        df.to_csv(path, index=False, encoding="utf-8-sig")

    df = df.copy()
    df["name"] = name
    df["display_code"] = baostock_to_display(code)
    return df


def load_all_prices(refresh: bool = False) -> pd.DataFrame:
    """Load or fetch all stocks required by the assignment."""
    frames = [
        load_or_fetch_stock(name, code, refresh=refresh)
        for name, code in config.STOCKS.items()
    ]
    return pd.concat(frames, ignore_index=True)

