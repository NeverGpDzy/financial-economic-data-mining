"""
作业4B 因子质检模块
功能：因子标准化、IC/IR计算、共线性检验、单因子OLS检验
"""
import os
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from scipy import stats

from . import config

warnings.filterwarnings('ignore')


def standardize_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-Score标准化因子

    公式: Factor_std = (Factor - μ) / σ
    范围: 4个中频因子（Reversal, Liquidity, MoneyFlow, Value）

    Returns:
        df: 添加了标准化因子列的DataFrame
    """
    print("[因子标准化] 对4个因子做Z-Score标准化...")
    factor_cols = config.FACTOR_NAMES

    for col in factor_cols:
        std_col = f'{col}_std'
        # 按日截面标准化（每天的截面均值和标准差）
        df[std_col] = df.groupby('trade_date')[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-10)
        )
        print(f"  {col} -> {std_col}")

    return df


def compute_ic_series(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    按日截面计算因子与次日收益的Spearman IC

    Returns:
        ic_dict: {因子名: 日度IC序列}
    """
    print("[IC计算] 按日截面计算Spearman IC...")
    factor_cols = config.FACTOR_NAMES
    ic_dict = {}

    dates = sorted(df['trade_date'].unique())

    for col in factor_cols:
        ic_list = []
        date_list = []
        for date in dates:
            daily = df[df['trade_date'] == date]
            if len(daily) < 30:  # 样本量过少跳过
                continue
            factor_vals = daily[col].dropna()
            next_ret = daily.loc[factor_vals.index, 'next_excess_ret'].dropna()
            common_idx = factor_vals.index.intersection(next_ret.index)
            if len(common_idx) < 30:
                continue
            ic, _ = stats.spearmanr(factor_vals.loc[common_idx],
                                     next_ret.loc[common_idx])
            ic_list.append(ic)
            date_list.append(date)

        ic_series = pd.Series(ic_list, index=date_list, name=col)
        ic_dict[col] = ic_series
        print(f"  {col}: {len(ic_series)}个截面IC值")

    return ic_dict


def compute_ir_statistics(ic_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    计算IC均值、IC标准差、IR值，做Newey-West调整t检验

    Returns:
        result_df: IC/IR统计结果表
    """
    print("[IR计算] 计算IC均值、标准差、IR值...")
    results = []

    for col, ic_series in ic_dict.items():
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ir = ic_mean / (ic_std + 1e-10)

        # Newey-West调整t检验
        n = len(ic_series)
        # 简化版：使用普通t检验 + Newey-West滞后阶数
        max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
        # 计算Newey-West调整的标准误
        demeaned = ic_series.values - ic_mean
        gamma_0 = np.mean(demeaned ** 2)
        nw_var = gamma_0
        for lag in range(1, max_lag + 1):
            weight = 1 - lag / (max_lag + 1)  # Bartlett核权重
            gamma_lag = np.mean(demeaned[lag:] * demeaned[:-lag])
            nw_var += 2 * weight * gamma_lag
        nw_se = np.sqrt(nw_var / n)
        t_stat = ic_mean / (nw_se + 1e-10)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

        # 判定有效性
        if abs(ic_mean) >= config.IC_THRESHOLD_EFFECTIVE:
            judgment = "有效"
        elif abs(ic_mean) >= config.IC_THRESHOLD_MARGINAL:
            judgment = "边缘"
        else:
            judgment = "无效"

        results.append({
            '因子': col,
            'IC均值': round(ic_mean, 6),
            'IC标准差': round(ic_std, 6),
            'IR值': round(ir, 4),
            't统计量': round(t_stat, 4),
            'p值': round(p_value, 6),
            '有效判定': judgment,
            '样本数': n
        })

        print(f"  {col}: IC均值={ic_mean:.6f}, IR={ir:.4f}, 判定={judgment}")

    result_df = pd.DataFrame(results)
    return result_df


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算因子VIF（方差膨胀因子）共线性检验

    VIF > 5 判定为高共线

    Returns:
        vif_df: VIF检验结果表
    """
    print("[VIF检验] 计算因子间共线性...")
    factor_cols = config.FACTOR_NAMES
    std_cols = [f'{c}_std' for c in factor_cols]

    # 取标准化后的因子数据，去除NaN
    X = df[std_cols].dropna()
    if len(X) < 100:
        print("  警告：样本量不足，VIF计算可能不稳定")

    vif_results = []
    for i, col in enumerate(std_cols):
        # 将当前因子作为因变量，其余因子作为自变量
        y = X[col].values
        other_cols = [c for c in std_cols if c != col]
        X_other = X[other_cols].values

        # 添加常数项
        X_other = np.column_stack([np.ones(len(X_other)), X_other])

        # OLS回归计算R²
        try:
            beta = np.linalg.lstsq(X_other, y, rcond=None)[0]
            y_pred = X_other @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - ss_res / (ss_tot + 1e-10)
            r_squared = max(0, min(r_squared, 1))  # 限制在[0,1]
            vif = 1.0 / (1.0 - r_squared + 1e-10)
        except Exception:
            vif = np.nan
            r_squared = np.nan

        factor_name = factor_cols[i]
        judgment = "高共线" if vif > config.VIF_THRESHOLD else "正常"

        vif_results.append({
            '因子': factor_name,
            'VIF': round(vif, 4),
            'R²': round(r_squared, 4),
            '判定': judgment
        })
        print(f"  {factor_name}: VIF={vif:.4f}, 判定={judgment}")

    vif_df = pd.DataFrame(vif_results)
    return vif_df


def single_factor_ols(df: pd.DataFrame) -> pd.DataFrame:
    """
    单因子横截面OLS回归检验

    回归公式: Return_{i,t+1} = α + β × Factor_{i,t} + ε
    因变量: 个股次日超额收益
    自变量: 4个标准化单因子（逐一回归）

    Returns:
        ols_df: OLS检验结果表
    """
    print("[单因子OLS] 按日截面做单因子OLS回归...")
    factor_cols = config.FACTOR_NAMES
    std_cols = [f'{c}_std' for c in factor_cols]

    results = []
    for i, (factor_name, std_col) in enumerate(zip(factor_cols, std_cols)):
        beta_list = []
        p_list = []
        t_list = []

        dates = sorted(df['trade_date'].unique())
        for date in dates:
            daily = df[df['trade_date'] == date].dropna(subset=[std_col, 'next_excess_ret'])
            if len(daily) < 30:
                continue

            y = daily['next_excess_ret'].values
            X = daily[std_col].values

            # 添加常数项
            X_mat = np.column_stack([np.ones(len(X)), X])

            try:
                # OLS回归
                beta = np.linalg.lstsq(X_mat, y, rcond=None)[0]
                y_pred = X_mat @ beta
                residuals = y - y_pred
                n, k = X_mat.shape

                # 计算标准误
                mse = np.sum(residuals ** 2) / (n - k)
                try:
                    cov_matrix = mse * np.linalg.inv(X_mat.T @ X_mat)
                    se = np.sqrt(np.diag(cov_matrix))
                except np.linalg.LinAlgError:
                    continue

                # t统计量和p值
                t_vals = beta / (se + 1e-10)
                p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), df=n - k))

                beta_list.append(beta[1])  # 因子系数
                p_list.append(p_vals[1])   # p值
                t_list.append(t_vals[1])   # t统计量
            except Exception:
                continue

        if len(beta_list) > 0:
            beta_mean = np.mean(beta_list)
            p_mean = np.mean(p_list)
            t_mean = np.mean(t_list)
            judgment = "线性有效" if p_mean < config.OLS_P_THRESHOLD else "待定(非线性)"

            results.append({
                '因子': factor_name,
                'β均值': round(beta_mean, 6),
                'p值均值': round(p_mean, 6),
                't统计量均值': round(t_mean, 4),
                '有效截面数': len(beta_list),
                '判定': judgment
            })
            print(f"  {factor_name}: β={beta_mean:.6f}, p={p_mean:.6f}, 判定={judgment}")
        else:
            results.append({
                '因子': factor_name,
                'β均值': np.nan,
                'p值均值': np.nan,
                't统计量均值': np.nan,
                '有效截面数': 0,
                '判定': '数据不足'
            })

    ols_df = pd.DataFrame(results)
    return ols_df


def run_factor_quality_check(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, pd.DataFrame]:
    """
    完整因子质检流水线

    Returns:
        df: 标准化后的数据
        ic_dict: IC序列字典
        ir_df: IC/IR统计结果
        vif_df: VIF检验结果
        ols_df: OLS检验结果
    """
    print("\n" + "=" * 60)
    print("模块2：因子质检（IC/IR+共线性检验）")
    print("=" * 60)

    # 1. 因子标准化
    df = standardize_factors(df)

    # 2. IC/IR计算
    ic_dict = compute_ic_series(df)
    ir_df = compute_ir_statistics(ic_dict)

    # 3. 共线性检验
    vif_df = compute_vif(df)

    # 4. 单因子OLS检验
    ols_df = single_factor_ols(df)

    # 保存结果
    ir_df.to_csv(os.path.join(config.OUTPUT_DIR, 'ic_ir_results.csv'), index=False)
    vif_df.to_csv(os.path.join(config.OUTPUT_DIR, 'vif_results.csv'), index=False)
    ols_df.to_csv(os.path.join(config.OUTPUT_DIR, 'ols_results.csv'), index=False)

    # 保存IC序列
    ic_df_list = []
    for col, ic_series in ic_dict.items():
        ic_temp = ic_series.reset_index()
        ic_temp.columns = ['trade_date', col]
        ic_df_list.append(ic_temp)
    if ic_df_list:
        ic_full = ic_df_list[0]
        for ic_temp in ic_df_list[1:]:
            ic_full = pd.merge(ic_full, ic_temp, on='trade_date', how='outer')
        ic_full.to_csv(os.path.join(config.OUTPUT_DIR, 'ic_series.csv'), index=False)

    return df, ic_dict, ir_df, vif_df, ols_df
