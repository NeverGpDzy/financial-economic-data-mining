"""因子计算（口径严格遵循作业要求 1.4.2）。

四类因子均在月末交易日截面计算：
1. 景气度因子 mom_mid  : 近60个交易日累计涨跌幅 = close[d]/close[d-60] - 1
2. 估值因子   pb       : 月末 pb_lf 数值（数值越小估值越低，模型自动学习反向关系）
3. 动量因子   mom_short: 近20个交易日累计涨跌幅 = close[d]/close[d-20] - 1
4. 资金流因子 flow     : 近20日日均成交额(行业) / 近20日日均成交额(沪深300)

随后对每个因子做截面 Z-Score 标准化（每月末跨31个行业），并计算下一个月的前瞻收益
fwd_return = close[d_next]/close[d] - 1，用于 IC/IR 质检、LGBM 训练与回测。

无未来函数：因子与前瞻收益严格按月末日期对齐，前瞻收益用的是 d 之后下一月的收益。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .data_loader import MarketData


def _zscore_cross_section(s: pd.Series) -> pd.Series:
    """截面 Z-Score：(x - mean) / std；std=0 时置 0。"""
    mu = s.mean()
    sd = s.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def compute_daily_factors(market: MarketData) -> dict[str, pd.DataFrame]:
    """在日频宽表上预计算四类因子（向量化），返回每日 × 行业的因子面板。"""
    close = market.ind_close
    amount = market.ind_amount
    pb = market.ind_pb
    hs_amount = market.hs_amount

    # 1) 景气度因子（中期动量，60日累计涨跌幅）
    mom_mid = close.pct_change(periods=cfg.MOM_MID_WINDOW)
    # 3) 动量因子（短期动量，20日累计涨跌幅）
    mom_short = close.pct_change(periods=cfg.MOM_SHORT_WINDOW)
    # 2) 估值因子（PB，直接取月末值）
    pb_factor = pb.copy()
    # 4) 资金流因子（20日日均成交额 / 沪深300同期20日日均成交额）
    ind_amt_ma20 = amount.rolling(cfg.FLOW_WINDOW).mean()
    hs_amt_ma20 = hs_amount.rolling(cfg.FLOW_WINDOW).mean()
    flow = ind_amt_ma20.div(hs_amt_ma20, axis=0)

    return {
        "mom_mid": mom_mid,
        "mom_short": mom_short,
        "pb": pb_factor,
        "flow": flow,
    }


def build_factor_panel(market: MarketData) -> pd.DataFrame:
    """构建月末截面的长表因子面板。

    返回列：
        signal_date, industry, mom_mid, pb, mom_short, flow,
        mom_mid_z, pb_z, mom_short_z, flow_z, fwd_date, fwd_return
    其中 signal_date 为月末交易日；fwd_date 为下一月末交易日；fwd_return 为该行业
    在 [signal_date, fwd_date] 期间的月度持有收益（与回测口径一致）。
    仅保留存在下一月末（即存在前瞻收益）的信号日。
    """
    daily = compute_daily_factors(market)
    close = market.ind_close
    month_ends = market.month_ends

    records = []
    # 仅遍历存在“下一月末”的信号日（最后一月末无前瞻收益，剔除）
    for i, d in enumerate(month_ends):
        if i + 1 >= len(month_ends):
            break
        d_next = month_ends[i + 1]
        # 该月末各行业因子原始值
        try:
            row_mid = daily["mom_mid"].loc[d]
            row_short = daily["mom_short"].loc[d]
            row_pb = daily["pb"].loc[d]
            row_flow = daily["flow"].loc[d]
            close_d = close.loc[d]
            close_next = close.loc[d_next]
        except KeyError:
            continue

        # 前瞻收益 = 下一月末收盘 / 本月末收盘 - 1
        fwd_return = close_next / close_d - 1.0

        for code in close.columns:
            records.append({
                "signal_date": d,
                "industry": code,
                "mom_mid": row_mid.get(code, np.nan),
                "pb": row_pb.get(code, np.nan),
                "mom_short": row_short.get(code, np.nan),
                "flow": row_flow.get(code, np.nan),
                "fwd_date": d_next,
                "fwd_return": fwd_return.get(code, np.nan),
            })

    panel = pd.DataFrame(records)
    # 丢弃因子或收益缺失的行（早期回溯不足）
    panel = panel.dropna(subset=["mom_mid", "mom_short", "pb", "flow", "fwd_return"]).reset_index(drop=True)

    # 截面 Z-Score 标准化（按 signal_date 分组）
    for raw in cfg.FACTOR_RAW:
        z_col = raw + "_z"
        panel[z_col] = panel.groupby("signal_date", group_keys=False)[raw].apply(_zscore_cross_section)

    # 行业名称便于阅读
    from .industry_codes import name_of
    panel["industry_name"] = panel["industry"].map(name_of)
    return panel


def filter_panel(panel: pd.DataFrame, start: str, end: str, by: str = "fwd_date") -> pd.DataFrame:
    """按区间筛选子面板。

    by='fwd_date'（默认）：按前瞻收益所属月份筛选。用于训练/IC/回测区间，可保证
        训练集只含已实现于样本内的收益，杜绝未来函数泄漏（例如 2024-12 信号的
        前瞻收益在 2025-01，属样本外，必须排除出训练集）。
    by='signal_date'：按信号生成月份筛选。
    """
    col = "fwd_date" if by == "fwd_date" else "signal_date"
    mask = (panel[col] >= pd.Timestamp(start)) & (panel[col] <= pd.Timestamp(end))
    return panel.loc[mask].copy()
