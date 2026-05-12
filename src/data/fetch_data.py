"""从 baostock 获取股票日线数据。"""

import baostock as bs
import pandas as pd
from pathlib import Path


def fetch_stock_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取单只股票的日收盘价（前复权）和成交量。

    Args:
        stock_code: 股票代码，如 "sh.600519"
        start_date: 起始日期，如 "2024-01-01"
        end_date: 结束日期，如 "2025-12-31"

    Returns:
        DataFrame，包含 date, close, volume 列
    """
    rs = bs.query_history_k_data_plus(
        stock_code,
        "date,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3",  # 前复权
    )
    if rs.error_code != "0":
        raise RuntimeError(f"baostock 查询失败: {rs.error_msg}")

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)

    # 类型转换
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.dropna()

    return df


def fetch_all_stocks(
    stock_codes: list[str], start_date: str, end_date: str, save_dir: str | None = None
) -> dict[str, pd.DataFrame]:
    """批量获取多只股票数据。

    Args:
        stock_codes: 股票代码列表
        start_date: 起始日期
        end_date: 结束日期
        save_dir: 可选，保存 CSV 的目录

    Returns:
        字典，key 为股票代码，value 为 DataFrame
    """
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")

    result = {}
    try:
        for code in stock_codes:
            df = fetch_stock_data(code, start_date, end_date)
            result[code] = df
            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                df.to_csv(Path(save_dir) / f"{code.replace('.', '_')}.csv")
            print(f"  {code}: {len(df)} 条记录")
    finally:
        bs.logout()

    return result
