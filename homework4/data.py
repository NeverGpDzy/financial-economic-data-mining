"""Module 1: Data fetching, factor calculation, cleaning, and train/test split."""

from __future__ import annotations

import pickle
from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd


def _get_sz50_stocks() -> pd.DataFrame:
    """Query SSE 50 constituent stocks via baostock."""
    rs = bs.query_sz50_stocks()
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    return df


def _fetch_daily(code: str, start: str, end: str, fields: str) -> pd.DataFrame:
    """Fetch daily K-line data and resample to month-end."""
    rs = bs.query_history_k_data_plus(
        code, fields, start_date=start, end_date=end,
        frequency="d", adjustflag="2",
    )
    if rs.error_code != "0":
        return pd.DataFrame()
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=rs.fields)
    df["date"] = pd.to_datetime(df["date"])
    numeric_cols = [c for c in df.columns if c not in ("date", "code", "code_name", "tradestatus")]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Resample to month-end: keep last trading day of each calendar month
    monthly = df.set_index("date").resample("M").last().dropna(subset=["close"]).reset_index()
    return monthly


def _fetch_financials(code: str, years: list[int]) -> dict[int, dict]:
    """Fetch annual financial data (Q4 profit + growth + dividend) for a stock."""
    result = {}
    for year in years:
        fin: dict = {"year": year}

        # Q4 profit data
        rs = bs.query_profit_data(code=code, year=year, quarter=4)
        if rs.error_code == "0":
            while rs.next():
                row = rs.get_row_data()
                fields = rs.fields
                for i, f in enumerate(fields):
                    fin[f] = row[i]

        # Growth data
        rs = bs.query_growth_data(code=code, year=year)
        if rs.error_code == "0":
            while rs.next():
                row = rs.get_row_data()
                fields = rs.fields
                for i, f in enumerate(fields):
                    fin[f"g_{f}"] = row[i]

        # Dividend data
        rs = bs.query_dividend_data(code=code, year=year, yearType="report")
        div_records = []
        if rs.error_code == "0":
            while rs.next():
                row = rs.get_row_data()
                fields = rs.fields
                d = dict(zip(fields, row))
                div_records.append(d)
        fin["dividend_records"] = div_records

        result[year] = fin
    return result


def _compute_factors(df: pd.DataFrame, financials: dict[int, dict]) -> pd.DataFrame:
    """Merge financial data and compute factors: SMB, PE_inv, Quality."""
    out = df.copy()

    last_np, last_shares, last_roe, last_div_ratio, last_growth = np.nan, np.nan, np.nan, np.nan, np.nan

    for yr in sorted(financials.keys()):
        fin = financials[yr]
        # Net profit & total shares (from profit data)
        np_val = pd.to_numeric(fin.get("netProfit", np.nan), errors="coerce")
        shares_val = pd.to_numeric(fin.get("totalShare", np.nan), errors="coerce")
        if not pd.isna(np_val):
            last_np = np_val
        if not pd.isna(shares_val) and shares_val > 0:
            last_shares = shares_val

        # ROE (already in decimal, e.g. 0.264 = 26.4%)
        roe_val = pd.to_numeric(fin.get("roeAvg", np.nan), errors="coerce")
        if not pd.isna(roe_val):
            last_roe = roe_val

        # Dividend ratio = total dividend / net profit
        div_records = fin.get("dividend_records", [])
        if div_records and not pd.isna(last_np) and last_np != 0 and not pd.isna(last_shares):
            total_div = 0.0
            for d in div_records:
                cash_div = pd.to_numeric(d.get("dividCashPsBeforeTax", 0), errors="coerce")
                if pd.isna(cash_div):
                    cash_div = 0.0
                total_div += cash_div
            last_div_ratio = total_div * last_shares / last_np

        # Net profit YoY growth
        g_val = pd.to_numeric(fin.get("g_YOYNI", np.nan), errors="coerce")
        if not pd.isna(g_val):
            last_growth = g_val / 100.0

        # Assign to months in this year
        mask = out["date"].dt.year == yr
        if mask.any():
            if not pd.isna(last_shares):
                out.loc[mask, "total_shares"] = last_shares
            if not pd.isna(last_roe):
                out.loc[mask, "roe"] = last_roe
            if not pd.isna(last_div_ratio):
                out.loc[mask, "div_ratio"] = last_div_ratio
            if not pd.isna(last_growth):
                out.loc[mask, "profit_growth"] = last_growth

    # Forward fill financial data
    for col in ["total_shares", "roe", "div_ratio", "profit_growth"]:
        if col in out.columns:
            out[col] = out[col].ffill()

    # Market cap = total_shares × close price (yuan)
    if "total_shares" in out.columns and "close" in out.columns:
        out["market_cap"] = out["total_shares"] * out["close"]
    else:
        out["market_cap"] = np.nan

    # PE_inv = 1 / PE_TTM (handle negative PE → set PE_inv to 0)
    out["pe_ttm"] = out["peTTM"].replace(0, np.nan)
    out["PE_inv"] = 1.0 / out["pe_ttm"]
    out.loc[out["pe_ttm"] < 0, "PE_inv"] = 0.0

    # Quality = ROE × dividend ratio × profit growth
    out["Quality"] = (
        out.get("roe", np.nan) *
        out.get("div_ratio", np.nan) *
        out.get("profit_growth", np.nan)
    )
    out["Quality"] = out["Quality"].clip(-10, 10)

    # SMB = market_cap
    out["SMB"] = out["market_cap"]

    return out


def _winsorize_3sigma(s: pd.Series) -> pd.Series:
    """3-sigma winsorization: clip values beyond mu ± 3*sigma."""
    mu, sigma = s.mean(), s.std()
    if sigma == 0 or pd.isna(sigma):
        return s
    return s.clip(mu - 3 * sigma, mu + 3 * sigma)


def get_data(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Main data pipeline: fetch, compute factors, clean, split.

    Returns (train_df, test_df, mkt_df).
    """
    from homework4.config import DATA_DIR, MARKET_CODE, TRAIN_START, TRAIN_END, TEST_START, TEST_END

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / "processed_data.pkl"

    if cache_path.exists() and not refresh:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        print("从缓存加载已处理数据...")
        return data["train"], data["test"], data["mkt"]

    bs.login()

    try:
        # 1. Get SSE 50 stock list
        print("获取上证50成分股...")
        sz50 = _get_sz50_stocks()
        stocks = sz50["code"].tolist()
        stock_names = dict(zip(sz50["code"], sz50["code_name"]))
        print(f"  共 {len(stocks)} 只股票")

        # 2. Get market index (monthly close only, using frequency="m" for index)
        print("获取上证指数月度数据...")
        rs = bs.query_history_k_data_plus(
            MARKET_CODE, "date,close", start_date=TRAIN_START, end_date=TEST_END,
            frequency="m", adjustflag="2",
        )
        mkt_rows = []
        while rs.next():
            mkt_rows.append(rs.get_row_data())
        mkt_raw = pd.DataFrame(mkt_rows, columns=rs.fields)
        if mkt_raw.empty:
            raise RuntimeError("无法获取上证指数数据")
        mkt_raw["date"] = pd.to_datetime(mkt_raw["date"])
        mkt_raw["close"] = pd.to_numeric(mkt_raw["close"], errors="coerce")
        mkt_raw = mkt_raw.set_index("date").sort_index()
        mkt_raw["mkt_return"] = mkt_raw["close"].pct_change()
        mkt_raw["mkt_excess"] = mkt_raw["mkt_return"] - 0.0015
        mkt_df = mkt_raw.copy()

        # 3. Get stock data (monthly from daily resampling for PE_TTM support)
        print("获取个股数据 (月度K线 + PE_TTM)...")
        years_list = list(range(2015, 2026))

        all_data = []
        for i, code in enumerate(stocks):
            name = stock_names.get(code, code)
            print(f"  [{i+1}/{len(stocks)}] {code} {name}")

            # Fetch daily with close + peTTM, resample to month-end
            df = _fetch_daily(code, TRAIN_START, TEST_END, "date,close,peTTM")
            if df.empty:
                print(f"    → 无K线数据，跳过")
                continue

            # Financials
            financials = _fetch_financials(code, years_list)

            # Compute factors
            df = _compute_factors(df, financials)
            df["code"] = code
            df["name"] = name
            all_data.append(df)

        if not all_data:
            raise RuntimeError("未获取到任何股票数据")

        # 4. Merge all stocks
        all_df = pd.concat(all_data, ignore_index=True)
        all_df = all_df.sort_values(["date", "code"]).reset_index(drop=True)

        # 5. Compute returns and excess returns
        all_df["return"] = all_df.groupby("code")["close"].transform(lambda x: x.pct_change())
        all_df["excess_return"] = all_df["return"] - 0.0015
        all_df["next_excess_return"] = all_df.groupby("code")["excess_return"].transform(lambda x: x.shift(-1))

        # 6. Forward fill missing financial data
        fill_cols = ["market_cap", "PE_inv", "Quality", "SMB", "roe", "div_ratio", "profit_growth"]
        for col in fill_cols:
            if col in all_df.columns:
                all_df[col] = all_df.groupby("code")[col].transform(
                    lambda x: x.replace([np.inf, -np.inf], np.nan).ffill()
                )

        # 7. Drop rows with NaN in critical fields
        all_df = all_df.dropna(subset=["close", "return", "excess_return"])

        # Factor NaNs → fill with cross-sectional mean per month
        for f in ["PE_inv", "Quality", "SMB"]:
            if f in all_df.columns:
                all_df[f] = all_df.groupby("date")[f].transform(lambda x: x.fillna(x.mean()))

        all_df = all_df.dropna(subset=["PE_inv", "Quality", "SMB", "next_excess_return"])

        # 8. Winsorize factors (3-sigma within each month cross-section)
        print("数据清洗：3σ缩尾处理...")
        for f in ["SMB", "PE_inv", "Quality"]:
            all_df[f] = all_df.groupby("date")[f].transform(_winsorize_3sigma)

        # 9. Split train / test
        TS = pd.to_datetime(TRAIN_START)
        TE = pd.to_datetime(TRAIN_END)
        TTS = pd.to_datetime(TEST_START)
        TTE = pd.to_datetime(TEST_END)

        train_df = all_df[(all_df["date"] >= TS) & (all_df["date"] <= TE)].copy()
        test_df = all_df[(all_df["date"] >= TTS) & (all_df["date"] <= TTE)].copy()

        print(f"训练集: {len(train_df)} 条, 回测集: {len(test_df)} 条")
        print(f"训练集股票数: {train_df['code'].nunique()}, 回测集股票数: {test_df['code'].nunique()}")

        # Cache
        with open(cache_path, "wb") as f:
            pickle.dump({"train": train_df, "test": test_df, "mkt": mkt_df}, f)

        return train_df, test_df, mkt_df

    finally:
        bs.logout()
