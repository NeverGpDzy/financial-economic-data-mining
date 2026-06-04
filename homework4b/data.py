"""
作业4B 数据加载与预处理模块
功能：读取Parquet数据、计算中频因子、数据清洗、数据集拆分
"""
import os
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Dict

from . import config

warnings.filterwarnings('ignore')


def load_parquet_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    批量读取本地Parquet格式数据

    Returns:
        daily_all: 所有股票日线行情合并表
        basic_all: 所有股票日线基本面合并表
        index_df: 沪深300指数日线行情
        components: 沪深300成分股列表
    """
    root = config.PARQUET_ROOT
    print(f"[数据加载] Parquet数据根目录: {root}")

    # 1. 读取沪深300成分股列表
    components = pd.read_parquet(os.path.join(root, 'hs300_components.parquet'))
    stock_codes = components['con_code'].unique().tolist()
    print(f"[数据加载] 沪深300成分股数量: {len(stock_codes)}")

    # 2. 批量读取个股日线行情
    daily_dir = os.path.join(root, 'daily')
    daily_list = []
    for code in stock_codes:
        fpath = os.path.join(daily_dir, f'{code}.parquet')
        if os.path.exists(fpath):
            df = pd.read_parquet(fpath)
            if len(df) >= 200:  # 过滤数据量过少的股票
                daily_list.append(df)
    daily_all = pd.concat(daily_list, ignore_index=True)
    daily_all['trade_date'] = pd.to_datetime(daily_all['trade_date'], format='%Y%m%d')
    print(f"[数据加载] 日线行情: {len(daily_list)}只股票, {len(daily_all)}条记录")

    # 3. 批量读取个股日线基本面
    basic_dir = os.path.join(root, 'daily_basic')
    basic_list = []
    for code in stock_codes:
        fpath = os.path.join(basic_dir, f'{code}.parquet')
        if os.path.exists(fpath):
            df = pd.read_parquet(fpath)
            if len(df) >= 200:
                basic_list.append(df)
    basic_all = pd.concat(basic_list, ignore_index=True)
    basic_all['trade_date'] = pd.to_datetime(basic_all['trade_date'], format='%Y%m%d')
    print(f"[数据加载] 日线基本面: {len(basic_list)}只股票, {len(basic_all)}条记录")

    # 4. 读取沪深300指数日线行情
    index_df = pd.read_parquet(os.path.join(root, 'index_daily.parquet'))
    index_df['trade_date'] = pd.to_datetime(index_df['trade_date'], format='%Y%m%d')
    index_df = index_df.sort_values('trade_date').reset_index(drop=True)
    print(f"[数据加载] 指数日线: {len(index_df)}条记录, "
          f"日期范围 {index_df['trade_date'].min().date()} ~ {index_df['trade_date'].max().date()}")

    return daily_all, basic_all, index_df, components


def compute_factors(daily_all: pd.DataFrame, basic_all: pd.DataFrame,
                    index_df: pd.DataFrame) -> pd.DataFrame:
    """
    合并数据并计算四大中频因子

    因子定义：
    - Reversal: -前1日个股收益率（超短反转）
    - Liquidity: Amihud非流动性指标（|日收益率| / 日成交额）
    - MoneyFlow: 成交量×价格方向 / 流通市值（资金流代理）
    - Value: 1 / PE_TTM（价值因子）

    Returns:
        df: 包含所有因子的完整DataFrame
    """
    print("[因子计算] 开始合并数据...")

    # 合并日线行情和基本面数据
    df = pd.merge(
        daily_all[['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                    'pre_close', 'pct_chg', 'vol', 'amount']],
        basic_all[['ts_code', 'trade_date', 'pe_ttm', 'circ_mv', 'free_share',
                    'total_mv', 'turnover_rate']],
        on=['ts_code', 'trade_date'],
        how='inner'
    )
    df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    # --- 计算个股收益率 ---
    df['ret'] = df.groupby('ts_code')['close'].pct_change()

    # --- 计算市场收益率（沪深300指数） ---
    index_ret = index_df[['trade_date', 'pct_chg']].copy()
    index_ret['mkt_ret'] = index_ret['pct_chg'] / 100.0  # 转为小数
    index_ret = index_ret[['trade_date', 'mkt_ret']]
    df = pd.merge(df, index_ret, on='trade_date', how='left')

    # --- 计算超额收益 ---
    df['excess_ret'] = df['ret'] - config.RISK_FREE_RATE_DAILY

    # --- 合成中频因子 ---
    print("[因子计算] 合成中频因子...")

    # ===== 四大标准因子（作业要求） =====

    # 1. Reversal因子：-前1日个股收益率（超短反转）
    df['Reversal'] = -df['ret']

    # 2. Liquidity因子：Amihud非流动性指标 = |日收益率| / 日成交额
    df['Liquidity'] = df['ret'].abs() / (df['amount'] + 1e-10)

    # 3. MoneyFlow因子：资金流代理
    #    使用 成交额×价格方向 / 流通市值 作为资金流代理
    #    上涨日为正向流入，下跌日为流出；除以流通市值标准化
    df['MoneyFlow'] = (df['amount'] * np.sign(df['ret'])) / (df['circ_mv'] + 1e-10)

    # 4. Value因子：PE_TTM倒数（正向化，低估值=高因子值）
    pe_valid = df['pe_ttm'].where(df['pe_ttm'] > 0, np.nan)
    df['Value'] = 1.0 / pe_valid

    # ===== 扩展因子（增强模型预测能力） =====

    # 5. Momentum因子：5日动量（短期趋势延续）
    df['Momentum'] = df.groupby('ts_code')['ret'].transform(
        lambda x: x.rolling(5, min_periods=3).sum()
    )

    # 6. Volatility因子：5日波动率（风险度量，负向因子）
    df['Volatility'] = df.groupby('ts_code')['ret'].transform(
        lambda x: x.rolling(5, min_periods=3).std()
    )

    # 7. Turnover因子：换手率（流动性度量）
    df['Turnover'] = df['turnover_rate']

    # 8. VolumeChange因子：成交量变化率（今日成交量 / 5日均量 - 1）
    df['VolumeChange'] = df.groupby('ts_code')['vol'].transform(
        lambda x: x / (x.rolling(5, min_periods=3).mean() + 1e-10) - 1
    )

    # --- 计算次日超额收益（作为标签） ---
    df['next_excess_ret'] = df.groupby('ts_code')['excess_ret'].shift(-1)

    print(f"[因子计算] 完成，共{len(df)}条记录, {df['ts_code'].nunique()}只股票")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗：缺失值处理 + 3σ缩尾去极值

    Returns:
        df: 清洗后的DataFrame
    """
    print("[数据清洗] 开始清洗...")
    factor_cols = config.FACTOR_NAMES

    # 1. 缺失值处理：前值填充（按股票分组）
    for col in factor_cols + ['circ_mv', 'pe_ttm']:
        df[col] = df.groupby('ts_code')[col].ffill()

    # 收益率相关缺失值：剔除停牌日（收益率为NaN的行）
    # 但保留因子数据，仅在后续建模时过滤

    # 2. 3σ缩尾去极值
    for col in factor_cols:
        valid = df[col].dropna()
        if len(valid) > 0:
            mu = valid.mean()
            sigma = valid.std()
            lower = mu - 3 * sigma
            upper = mu + 3 * sigma
            n_clip = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = df[col].clip(lower, upper)
            print(f"  {col}: μ={mu:.6f}, σ={sigma:.6f}, 缩尾{n_clip}个极端值")

    # 3. 剔除收益率为NaN的行（停牌日）
    n_before = len(df)
    df = df.dropna(subset=['ret', 'next_excess_ret'])
    print(f"[数据清洗] 剔除停牌日: {n_before} -> {len(df)} 条记录")

    # 4. 剔除因子全为NaN的行
    df = df.dropna(subset=factor_cols, how='all')
    print(f"[数据清洗] 最终数据: {len(df)}条记录, {df['ts_code'].nunique()}只股票")

    return df


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按时间拆分训练集和回测集

    Returns:
        train_df: 训练集（2020-2023）
        test_df: 回测集（2024-2025）
    """
    train_mask = (df['trade_date'] >= config.TRAIN_START) & (df['trade_date'] <= config.TRAIN_END)
    test_mask = (df['trade_date'] >= config.TEST_START) & (df['trade_date'] <= config.TEST_END)

    train_df = df[train_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    print(f"[数据集拆分] 训练集: {len(train_df)}条 ({train_df['trade_date'].min().date()}"
          f" ~ {train_df['trade_date'].max().date()})")
    print(f"[数据集拆分] 回测集: {len(test_df)}条 ({test_df['trade_date'].min().date()}"
          f" ~ {test_df['trade_date'].max().date()})")

    return train_df, test_df


def save_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame,
                  full_df: pd.DataFrame) -> None:
    """保存清洗后的数据集为CSV"""
    out_dir = config.DATA_OUTPUT_DIR

    train_df.to_csv(os.path.join(out_dir, 'train_data.csv'), index=False)
    test_df.to_csv(os.path.join(out_dir, 'test_data.csv'), index=False)
    full_df.to_csv(os.path.join(out_dir, 'full_data.csv'), index=False)

    print(f"[数据保存] 数据集已保存至 {out_dir}")
    print(f"  - train_data.csv: {len(train_df)}行")
    print(f"  - test_data.csv:  {len(test_df)}行")
    print(f"  - full_data.csv:  {len(full_df)}行")


def load_and_preprocess() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    完整数据加载与预处理流水线

    Returns:
        train_df: 训练集
        test_df: 回测集
        full_df: 完整数据集
    """
    # Step 1: 加载Parquet数据
    daily_all, basic_all, index_df, components = load_parquet_data()

    # Step 2: 计算因子
    full_df = compute_factors(daily_all, basic_all, index_df)

    # Step 3: 数据清洗
    full_df = clean_data(full_df)

    # Step 4: 数据集拆分
    train_df, test_df = split_dataset(full_df)

    # Step 5: 保存数据集
    save_datasets(train_df, test_df, full_df)

    return train_df, test_df, full_df
