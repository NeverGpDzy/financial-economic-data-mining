"""公共数据获取模块。各次作业复用。"""

import baostock as bs
import pandas as pd
from pathlib import Path


def fetch_stock_data(stock_code: str, start_date: str, end_date: str,
                     fields: str = "date,close,volume") -> pd.DataFrame:
    """获取单只股票日线数据（前复权）。

    Args:
        stock_code: 股票代码，如 "sh.600519"
        start_date: 起始日期
        end_date: 结束日期
        fields: 查询字段，默认 "date,close,volume"

    Returns:
        DataFrame，index 为日期
    """
    rs = bs.query_history_k_data_plus(
        stock_code, fields,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3",
    )
    if rs.error_code != "0":
        raise RuntimeError(f"baostock 查询失败: {rs.error_msg}")

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)

    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.dropna()

    return df


def fetch_all_stocks(
    stock_codes: list[str], start_date: str, end_date: str,
    fields: str = "date,close,volume", save_dir: str | None = None,
) -> dict[str, pd.DataFrame]:
    """批量获取多只股票数据。"""
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")

    result = {}
    try:
        for code in stock_codes:
            df = fetch_stock_data(code, start_date, end_date, fields)
            result[code] = df
            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                df.to_csv(Path(save_dir) / f"{code.replace('.', '_')}.csv")
            print(f"  {code}: {len(df)} 条记录")
    finally:
        bs.logout()

    return result
