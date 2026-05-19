"""Data access and return preprocessing for CAPM."""

from __future__ import annotations

from pathlib import Path

import baostock as bs
import pandas as pd
from scipy.stats.mstats import winsorize


def fetch_close_price(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch forward-adjusted daily close prices from Baostock."""
    rs = bs.query_history_k_data_plus(
        code,
        "date,close",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3",
    )
    if rs.error_code != "0":
        raise RuntimeError(f"Baostock query failed for {code}: {rs.error_msg}")

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())

    df = pd.DataFrame(rows, columns=rs.fields)
    if df.empty:
        raise ValueError(f"No data returned for {code} from {start_date} to {end_date}.")

    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).set_index("date").sort_index()
    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add raw returns and 1%/99% winsorized returns."""
    out = df.copy()
    out["raw_return"] = out["close"].pct_change()
    out = out.dropna(subset=["raw_return"])
    out["return"] = winsorize(out["raw_return"], limits=[0.01, 0.01]).filled()

    return out


def load_or_fetch(
    code: str,
    start_date: str,
    end_date: str,
    data_dir: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Use a local CSV cache when available, otherwise fetch from Baostock."""
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"{code.replace('.', '_')}_{start_date}_{end_date}.csv"

    if csv_path.exists() and not refresh:
        df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
        return df

    df = fetch_close_price(code, start_date, end_date)
    df.to_csv(csv_path, index_label="date")
    return df


def fetch_many(
    codes: dict[str, str],
    start_date: str,
    end_date: str,
    data_dir: Path,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch all named instruments inside one Baostock session."""
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"Baostock login failed: {login.error_msg}")

    try:
        return {
            name: add_returns(load_or_fetch(code, start_date, end_date, data_dir, refresh))
            for name, code in codes.items()
        }
    finally:
        bs.logout()
