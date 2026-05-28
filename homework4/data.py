"""Module 1: Data loading from pre-processed supporting data files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _winsorize_3sigma(s: pd.Series) -> pd.Series:
    """3-sigma winsorization: clip values beyond mu ± 3*sigma."""
    mu, sigma = s.mean(), s.std()
    if sigma == 0 or pd.isna(sigma):
        return s
    return s.clip(mu - 3 * sigma, mu + 3 * sigma)


def load_supporting_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-processed data from supporting CSV files.

    Returns (train_df, test_df, mkt_df).
    """
    from homework4.config import DATA_DIR, RF_MONTHLY

    csv_dir = DATA_DIR / "配套数据" / "csv"

    if not csv_dir.exists():
        raise FileNotFoundError(f"配套数据目录不存在: {csv_dir}")

    print("从配套数据文件加载...")

    # 1. Load stock list
    stock_list = pd.read_csv(csv_dir / "stock_list.csv", encoding="utf-8-sig")
    print(f"  股票池: {len(stock_list)} 只股票")

    # 2. Load index monthly data
    mkt_df = pd.read_csv(csv_dir / "index_monthly.csv", parse_dates=["trade_date"], encoding="utf-8-sig")
    mkt_df = mkt_df.rename(columns={"trade_date": "date", "mkt_close": "close", "mkt_ret": "mkt_return"})
    mkt_df = mkt_df.set_index("date").sort_index()
    mkt_df["mkt_excess"] = mkt_df["mkt_return"] - RF_MONTHLY
    print(f"  上证指数: {mkt_df.index.min().date()} ~ {mkt_df.index.max().date()}")

    # 3. Load train factor data
    train_factor = pd.read_csv(csv_dir / "train_factor.csv", parse_dates=["trade_date"], encoding="utf-8-sig")
    train_factor = train_factor.rename(columns={"trade_date": "date", "stock_code": "code"})

    # 4. Load train return data
    train_ret = pd.read_csv(csv_dir / "train_ret.csv", parse_dates=["trade_date"], encoding="utf-8-sig")
    train_ret = train_ret.rename(columns={"trade_date": "date", "stock_code": "code", "monthly_ret": "return"})

    # 5. Load test factor data
    test_factor = pd.read_csv(csv_dir / "test_factor.csv", parse_dates=["trade_date"], encoding="utf-8-sig")
    test_factor = test_factor.rename(columns={"trade_date": "date", "stock_code": "code"})

    # 6. Load test return data
    test_ret = pd.read_csv(csv_dir / "test_ret.csv", parse_dates=["trade_date"], encoding="utf-8-sig")
    test_ret = test_ret.rename(columns={"trade_date": "date", "stock_code": "code", "monthly_ret": "return"})

    # 7. Merge factor and return data
    train_df = pd.merge(train_factor, train_ret, on=["date", "code"], how="inner")
    test_df = pd.merge(test_factor, test_ret, on=["date", "code"], how="inner")

    # 8. Add stock names from stock_list
    # Note: stock_list only has stock_code, we'll use code as name for simplicity
    train_df["name"] = train_df["code"]
    test_df["name"] = test_df["code"]

    # 9. Compute next excess return (target variable)
    train_df = train_df.sort_values(["code", "date"]).reset_index(drop=True)
    test_df = test_df.sort_values(["code", "date"]).reset_index(drop=True)

    train_df["next_excess_return"] = train_df.groupby("code")["excess_ret"].transform(lambda x: x.shift(-1))
    test_df["next_excess_return"] = test_df.groupby("code")["excess_ret"].transform(lambda x: x.shift(-1))

    # 10. Rename columns to match expected names
    # smb -> SMB, pe_recip -> PE_inv, quality -> Quality, excess_ret -> excess_return
    rename_map = {"smb": "SMB", "pe_recip": "PE_inv", "quality": "Quality", "excess_ret": "excess_return"}
    train_df = train_df.rename(columns=rename_map)
    test_df = test_df.rename(columns=rename_map)

    # 11. Apply 3-sigma winsorization to factors
    print("数据清洗：3σ缩尾处理...")
    for f in ["SMB", "PE_inv", "Quality"]:
        train_df[f] = train_df.groupby("date")[f].transform(_winsorize_3sigma)
        test_df[f] = test_df.groupby("date")[f].transform(_winsorize_3sigma)

    # 12. Drop rows with NaN in critical fields
    train_df = train_df.dropna(subset=["SMB", "PE_inv", "Quality", "next_excess_return"])
    test_df = test_df.dropna(subset=["SMB", "PE_inv", "Quality", "next_excess_return"])

    print(f"训练集: {len(train_df)} 条, 股票数: {train_df['code'].nunique()}")
    print(f"回测集: {len(test_df)} 条, 股票数: {test_df['code'].nunique()}")

    return train_df, test_df, mkt_df


def get_data(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Main data pipeline: load from supporting data files.

    Returns (train_df, test_df, mkt_df).
    """
    import pickle
    from homework4.config import DATA_DIR

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / "processed_data.pkl"

    if cache_path.exists() and not refresh:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        print("从缓存加载已处理数据...")
        return data["train"], data["test"], data["mkt"]

    # Load from supporting data
    train_df, test_df, mkt_df = load_supporting_data()

    # Cache
    with open(cache_path, "wb") as f:
        pickle.dump({"train": train_df, "test": test_df, "mkt": mkt_df}, f)

    return train_df, test_df, mkt_df
