"""数据加载与对齐。

读取三份配套 CSV，构建统一的日频面板：
- 申万一级 31 行业：收盘价 / 成交额 / PB（date × industry 宽表）
- 沪深 300：收盘价 / 成交额（Series）
- 月末交易日清单（每月最后一个交易日）

行情日期在 CSV 中为 YYYYMMDD 整数，统一解析为 Timestamp。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg
from .industry_codes import all_codes


@dataclass
class MarketData:
    """对齐后的市场数据面板。"""

    dates: pd.DatetimeIndex                 # 全部交易日（升序）
    ind_close: pd.DataFrame                 # date × industry 收盘价
    ind_amount: pd.DataFrame                # date × industry 成交额
    ind_pb: pd.DataFrame                    # date × industry PB
    hs_close: pd.Series                     # 沪深300收盘价
    hs_amount: pd.Series                    # 沪深300成交额
    month_ends: pd.DatetimeIndex            # 月末交易日（每月最后一个交易日）

    @property
    def n_industries(self) -> int:
        return self.ind_close.shape[1]


def _parse_date(s: pd.Series) -> pd.Series:
    """YYYYMMDD 整数/字符串 -> Timestamp。"""
    return pd.to_datetime(s.astype(str), format="%Y%m%d")


def _read_csv(path: Path_like) -> pd.DataFrame:
    import os
    if isinstance(path, str):
        path = os.path.fspath(path)
    return pd.read_csv(path, encoding="utf-8-sig")


Path_like = "pathlib.Path | str"


def load_market() -> MarketData:
    """加载三份 CSV 并对齐为日频面板。"""
    codes = all_codes()

    # ---- 申万行业日行情 ----
    sw = _read_csv(cfg.SW_DAILY_CSV)
    sw["date"] = _parse_date(sw["trade_date"])
    sw = sw[["date", "ts_code", "close", "amount"]].copy()
    ind_close = sw.pivot(index="date", columns="ts_code", values="close").sort_index()
    ind_amount = sw.pivot(index="date", columns="ts_code", values="amount").sort_index()
    # 仅保留 31 个标准行业，按代码升序排列列
    ind_close = ind_close.reindex(columns=codes)
    ind_amount = ind_amount.reindex(columns=codes)

    # ---- 申万行业 PB ----
    pb = _read_csv(cfg.SW_PB_CSV)
    pb["date"] = _parse_date(pb["trade_date"])
    pb = pb[["date", "ts_code", "pb"]].copy()
    ind_pb = pb.pivot(index="date", columns="ts_code", values="pb").sort_index()
    ind_pb = ind_pb.reindex(columns=codes)

    # ---- 沪深300日行情 ----
    hs = _read_csv(cfg.HS300_CSV)
    hs["date"] = _parse_date(hs["trade_date"])
    hs = hs.set_index("date").sort_index()
    hs_close = hs["close"].astype(float)
    hs_amount = hs["amount"].astype(float)

    # ---- 对齐到共同交易日（三张表日期完全一致，此处做防御性对齐）----
    common_dates = ind_close.index.intersection(ind_pb.index).intersection(hs_close.index)
    common_dates = common_dates.sort_values()
    ind_close = ind_close.loc[common_dates]
    ind_amount = ind_amount.loc[common_dates]
    ind_pb = ind_pb.loc[common_dates]
    hs_close = hs_close.loc[common_dates]
    hs_amount = hs_amount.loc[common_dates]

    # ---- 月末交易日：每个日历月的最后一个交易日 ----
    month_ends = _month_end_dates(common_dates)

    return MarketData(
        dates=common_dates,
        ind_close=ind_close,
        ind_amount=ind_amount,
        ind_pb=ind_pb,
        hs_close=hs_close,
        hs_amount=hs_amount,
        month_ends=month_ends,
    )


def _month_end_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """返回每月最后一个交易日。"""
    s = pd.Series(dates, index=dates)
    # 按年月分组取最大（即该月最后一个交易日）
    me = s.groupby([s.index.year, s.index.month]).max()
    return pd.DatetimeIndex(sorted(me.values))


def next_month_end(month_ends: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp | None:
    """返回 d 之后的下一个月末交易日；d 为最后一个月末时返回 None。"""
    idx = month_ends.get_indexer([d])[0]
    if idx < 0 or idx + 1 >= len(month_ends):
        return None
    return month_ends[idx + 1]
