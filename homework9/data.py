"""Data fetching and preparation for Homework 9."""

from __future__ import annotations

import pandas as pd

from . import config


def baostock_to_display(code: str) -> str:
    exchange, symbol = code.split(".")
    return f"{symbol}.{exchange.upper()}"


def fetch_daily_close(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily close prices from Baostock after caller has logged in."""
    import baostock as bs

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


def load_all_prices(refresh: bool = False) -> pd.DataFrame:
    """Load both assets in long format from cache or Baostock."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    missing: list[tuple[str, str, str]] = []

    for name, code in config.ASSETS.items():
        display_code = baostock_to_display(code)
        path = config.RAW_DIR / f"{display_code}_{config.DATA_START}_{config.BACKTEST_END}.csv"
        if path.exists() and not refresh:
            df = pd.read_csv(path, parse_dates=["date"])
        else:
            missing.append((name, code, display_code))
            continue
        df["name"] = name
        df["display_code"] = display_code
        frames.append(df)

    if missing:
        import baostock as bs

        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(f"Baostock登录失败: {login.error_msg}")
        try:
            for name, code, display_code in missing:
                df = fetch_daily_close(code, config.DATA_START, config.BACKTEST_END)
                path = config.RAW_DIR / f"{display_code}_{config.DATA_START}_{config.BACKTEST_END}.csv"
                df.to_csv(path, index=False, encoding="utf-8-sig")
                df["name"] = name
                df["display_code"] = display_code
                frames.append(df)
        finally:
            bs.logout()

    return pd.concat(frames, ignore_index=True)


def build_close_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Pivot close prices into a date-aligned close price matrix."""
    close = prices.pivot(index="date", columns="name", values="close")
    close = close.sort_index().dropna(how="any")
    return close[[config.MAOTAI, config.LAOJIAO]]
