"""
作业4B 日频截面选股+1~2日持仓样本外回测模块
功能：LGBM打分选股、日频调仓回测、绩效指标计算
"""
import os
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Tuple

import lightgbm as lgb

from . import config

warnings.filterwarnings('ignore')


def daily_scoring(model: lgb.Booster, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    每日截面用LGBM模型打分

    综合得分 = LGBM模型预测的个股次日超额收益
    得分越高，1~2日持仓收益预期越强

    Returns:
        scored_df: 包含综合得分的DataFrame
    """
    feature_cols = [f'{c}_std' for c in config.FACTOR_NAMES]

    # 去除特征为NaN的行
    scored_df = test_df.dropna(subset=feature_cols).copy()

    # 模型预测
    scored_df['score'] = model.predict(scored_df[feature_cols])

    print(f"[打分] 完成{len(scored_df)}条记录的截面打分")
    return scored_df


def run_backtest(scored_df: pd.DataFrame, index_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    实现1~2日中频调仓回测

    流程：
    - 每隔holding_days天调仓一次
    - 调仓日：用当日得分选Top5，收盘买入
    - 持仓期间：每日计算持仓收益（使用当日ret）
    - 调仓日当天：用旧持仓计算收益，扣交易成本，然后换新持仓

    Returns:
        backtest_df: 每日回测结果
        trade_log: 交易日志
    """
    print("\n[回测] 开始样本外回测...")

    trade_dates = sorted(scored_df['trade_date'].unique())
    n_dates = len(trade_dates)
    holding_days = config.HOLDING_DAYS
    print(f"[回测] 回测期交易日数: {n_dates}, 持仓天数: {holding_days}")

    # 准备指数收益数据
    index_ret_map = index_df.set_index('trade_date')['pct_chg'] / 100.0

    # 初始化
    nav = config.INITIAL_CAPITAL
    holdings = []             # 当前持仓股票列表
    days_since_rebalance = 0  # 距上次调仓的天数

    # 记录列表
    records = []
    trade_log = []

    for i, date in enumerate(trade_dates):
        daily = scored_df[scored_df['trade_date'] == date].copy()
        # 策略与基准均从首个回测日收盘开始，首日不计入基准前一日涨跌。
        mkt_ret = 0.0 if i == 0 else index_ret_map.get(date, 0.0)
        is_last_day = i == n_dates - 1

        # ---- 步骤1：计算当日已有持仓收益 ----
        if holdings:
            # 用当前持仓在今日的收益率
            holdings_ret = []
            for stock in holdings:
                row = daily[daily['ts_code'] == stock]
                if len(row) > 0:
                    r = row['ret'].values[0]
                    holdings_ret.append(r if not np.isnan(r) else 0.0)
                else:
                    holdings_ret.append(0.0)
            portfolio_ret = np.mean(holdings_ret) if holdings_ret else 0.0
        else:
            portfolio_ret = 0.0

        net_ret = portfolio_ret

        # ---- 步骤2：建仓或调仓 ----
        if not holdings:
            if len(daily) >= config.TOP_N_STOCKS:
                daily_sorted = daily.sort_values('score', ascending=False)
                holdings = daily_sorted.head(config.TOP_N_STOCKS)['ts_code'].tolist()
                # 初始建仓只产生买入单边成本。
                net_ret = -config.COMMISSION_RATE
                nav = nav * (1 + net_ret)
                days_since_rebalance = 1
                trade_log.append({
                    'trade_date': date,
                    'holdings': holdings,
                    'sold': [],
                    'bought': holdings,
                    'turnover': 1.0,
                    'cost': config.COMMISSION_RATE,
                })
        elif (days_since_rebalance >= holding_days) and (not is_last_day) and len(daily) >= config.TOP_N_STOCKS:
            # 调仓日：先结算旧持仓当日收益，再卖出旧持仓、买入新持仓并扣成本。
            daily_sorted = daily.sort_values('score', ascending=False)
            new_holdings = daily_sorted.head(config.TOP_N_STOCKS)['ts_code'].tolist()

            sold = list(set(holdings) - set(new_holdings))
            bought = list(set(new_holdings) - set(holdings))
            turnover = len(bought) / config.TOP_N_STOCKS
            cost = (len(sold) + len(bought)) / config.TOP_N_STOCKS * config.COMMISSION_RATE

            # 记录交易日志
            trade_log.append({
                'trade_date': date,
                'holdings': new_holdings,
                'sold': sold,
                'bought': bought,
                'turnover': turnover,
                'cost': cost,
            })

            # 扣除成本后的净收益
            net_ret = portfolio_ret - cost
            nav = nav * (1 + net_ret)

            holdings = new_holdings
            days_since_rebalance = 1

        else:
            # 非调仓日或最后一个交易日：继续持有，无交易成本
            nav = nav * (1 + net_ret)
            days_since_rebalance += 1

        records.append({
            'trade_date': date,
            'nav': nav,
            'daily_ret': net_ret,
            'mkt_ret': mkt_ret,
        })

    # 构建回测结果DataFrame
    backtest_df = pd.DataFrame(records)
    backtest_df['cum_ret'] = backtest_df['nav'] / config.INITIAL_CAPITAL - 1
    backtest_df['mkt_cum_ret'] = (1 + backtest_df['mkt_ret']).cumprod() - 1
    backtest_df['excess_ret'] = backtest_df['daily_ret'] - backtest_df['mkt_ret']
    backtest_df['excess_cum_ret'] = backtest_df['cum_ret'] - backtest_df['mkt_cum_ret']

    trade_log_df = pd.DataFrame(trade_log)

    return backtest_df, trade_log_df


def compute_performance_metrics(backtest_df: pd.DataFrame) -> Dict:
    """
    计算回测绩效指标

    Returns:
        metrics: 绩效指标字典
    """
    print("\n[绩效指标] 计算回测核心指标...")

    df = backtest_df.copy()
    daily_ret = df['daily_ret']
    nav = df['nav']

    n_days = len(daily_ret)
    n_years = n_days / 252

    # 1. 累计收益率
    cum_return = nav.iloc[-1] / config.INITIAL_CAPITAL - 1

    # 2. 年化收益率
    annual_return = (1 + cum_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # 3. 夏普比率（年化）
    daily_rf = config.RISK_FREE_RATE_DAILY
    excess_daily = daily_ret - daily_rf
    sharpe = np.sqrt(252) * excess_daily.mean() / (excess_daily.std() + 1e-10)

    # 4. 最大回撤
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    max_drawdown = drawdown.min()

    # 5. 日胜率
    win_rate = (daily_ret > 0).sum() / n_days

    # 6. 超额收益
    total_excess = cum_return - df['mkt_cum_ret'].iloc[-1]
    mkt_cum = df['mkt_cum_ret'].iloc[-1]
    mkt_annual = (1 + mkt_cum) ** (1 / n_years) - 1 if n_years > 0 else 0
    annual_excess = annual_return - mkt_annual

    # 7. 信息比率
    excess_daily_ret = df['excess_ret']
    ir = np.sqrt(252) * excess_daily_ret.mean() / (excess_daily_ret.std() + 1e-10)

    # 8. Calmar比率
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    metrics = {
        '回测区间': f"{df['trade_date'].iloc[0].strftime('%Y-%m-%d')} ~ "
                    f"{df['trade_date'].iloc[-1].strftime('%Y-%m-%d')}",
        '回测天数': n_days,
        '累计收益率': f"{cum_return:.2%}",
        '年化收益率': f"{annual_return:.2%}",
        '夏普比率': round(sharpe, 4),
        '最大回撤': f"{max_drawdown:.2%}",
        '日胜率': f"{win_rate:.2%}",
        '超额收益(累计)': f"{total_excess:.2%}",
        '超额收益(年化)': f"{annual_excess:.2%}",
        '信息比率': round(ir, 4),
        'Calmar比率': round(calmar, 4),
        '初始资金': f"{config.INITIAL_CAPITAL:,.0f}",
        '最终净值': f"{nav.iloc[-1]:,.0f}",
    }

    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics


def save_backtest_results(backtest_df: pd.DataFrame, trade_log_df: pd.DataFrame,
                          metrics: Dict) -> None:
    """保存回测结果"""
    out_dir = config.OUTPUT_DIR

    backtest_df.to_csv(os.path.join(out_dir, 'backtest_result.csv'), index=False)
    trade_log_df.to_csv(os.path.join(out_dir, 'trade_log.csv'), index=False)

    import json
    metrics_json = {k: str(v) for k, v in metrics.items()}
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=2)

    print(f"[回测保存] 结果已保存至 {out_dir}")


def run_backtest_pipeline(model: lgb.Booster, test_df: pd.DataFrame,
                          index_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    完整回测流水线

    Returns:
        backtest_df: 回测结果
        metrics: 绩效指标
    """
    print("\n" + "=" * 60)
    print("模块5：日频截面选股+1~2日持仓样本外回测")
    print("=" * 60)

    # 1. 每日打分
    scored_df = daily_scoring(model, test_df)

    # 2. 回测
    backtest_df, trade_log_df = run_backtest(scored_df, index_df)

    # 3. 计算绩效指标
    metrics = compute_performance_metrics(backtest_df)

    # 4. 保存结果
    save_backtest_results(backtest_df, trade_log_df, metrics)

    return backtest_df, metrics
