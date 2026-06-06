"""
作业4B 可视化模块
功能：绘制回测图表、IC分析图、因子重要性图等
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict

from . import config

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_nav_vs_market(backtest_df: pd.DataFrame) -> str:
    """
    绘制策略净值 vs 沪深300对比图

    Returns:
        fig_path: 图片保存路径
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.FIGURE_SIZE,
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('作业4B：策略净值 vs 沪深300', fontsize=14, fontweight='bold')

    dates = backtest_df['trade_date']

    # 上图：净值曲线
    strategy_nav = backtest_df['nav'] / config.INITIAL_CAPITAL
    market_nav = (1 + backtest_df['mkt_ret']).cumprod()

    ax1.plot(dates, strategy_nav, 'b-', linewidth=1.5, label='策略净值')
    ax1.plot(dates, market_nav, 'r--', linewidth=1.0, label='沪深300')
    ax1.fill_between(dates, strategy_nav, market_nav,
                      where=strategy_nav > market_nav, alpha=0.15, color='green')
    ax1.fill_between(dates, strategy_nav, market_nav,
                      where=strategy_nav < market_nav, alpha=0.15, color='red')
    ax1.set_ylabel('净值')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # 下图：超额收益曲线
    excess_cum = backtest_df['excess_cum_ret']
    ax2.fill_between(dates, 0, excess_cum,
                      where=excess_cum >= 0, alpha=0.4, color='green', label='正超额')
    ax2.fill_between(dates, 0, excess_cum,
                      where=excess_cum < 0, alpha=0.4, color='red', label='负超额')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('超额收益')
    ax2.set_xlabel('日期')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout()
    fig_path = os.path.join(config.OUTPUT_DIR, 'nav_vs_market.png')
    plt.savefig(fig_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"[图表] 策略净值对比图已保存: {fig_path}")
    return fig_path


def plot_drawdown(backtest_df: pd.DataFrame) -> str:
    """绘制回撤曲线图"""
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)

    nav = backtest_df['nav']
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax * 100  # 转为百分比

    dates = backtest_df['trade_date']
    ax.fill_between(dates, drawdown, 0, alpha=0.5, color='red')
    ax.plot(dates, drawdown, 'r-', linewidth=0.8)
    ax.set_title('策略回撤曲线', fontsize=14, fontweight='bold')
    ax.set_ylabel('回撤 (%)')
    ax.set_xlabel('日期')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # 标注最大回撤
    max_dd_idx = drawdown.idxmin()
    max_dd = drawdown.min()
    ax.annotate(f'最大回撤: {max_dd:.2f}%',
                xy=(dates.iloc[max_dd_idx], max_dd),
                xytext=(dates.iloc[max_dd_idx], max_dd - 2),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, ha='center')

    plt.tight_layout()
    fig_path = os.path.join(config.OUTPUT_DIR, 'drawdown.png')
    plt.savefig(fig_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"[图表] 回撤曲线图已保存: {fig_path}")
    return fig_path


def plot_ic_series(ic_dict: Dict[str, pd.Series]) -> str:
    """绘制IC时间序列图"""
    n_factors = len(ic_dict)
    n_cols = 2
    n_rows = (n_factors + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    fig.suptitle('因子IC时间序列', fontsize=14, fontweight='bold')

    factor_names = list(ic_dict.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for idx, (col, ic_series) in enumerate(ic_dict.items()):
        ax = axes[idx // n_cols][idx % n_cols] if n_rows > 1 else axes[idx % n_cols]
        dates = ic_series.index

        # IC序列
        ax.bar(dates, ic_series.values, alpha=0.5, color=colors[idx], width=1)

        # 20日移动平均
        if len(ic_series) > 20:
            ma20 = ic_series.rolling(20).mean()
            ax.plot(dates, ma20.values, 'r-', linewidth=1.5, label='20日MA')

        # 阈值线
        ax.axhline(y=config.IC_THRESHOLD_EFFECTIVE, color='green', linestyle='--',
                    alpha=0.5, label=f'有效阈值({config.IC_THRESHOLD_EFFECTIVE})')
        ax.axhline(y=-config.IC_THRESHOLD_EFFECTIVE, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='black', linewidth=0.5)

        # IC均值
        ic_mean = ic_series.mean()
        ax.set_title(f'{col} (IC均值={ic_mean:.4f})', fontsize=11)
        ax.set_ylabel('IC')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    fig_path = os.path.join(config.OUTPUT_DIR, 'ic_series.png')
    plt.savefig(fig_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"[图表] IC序列图已保存: {fig_path}")
    return fig_path


def plot_ic_ir_summary(ir_df: pd.DataFrame) -> str:
    """绘制IC/IR汇总柱状图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.FIGURE_SIZE)
    fig.suptitle('因子IC/IR汇总', fontsize=14, fontweight='bold')

    factors = ir_df['因子'].values
    ic_means = ir_df['IC均值'].values
    ir_vals = ir_df['IR值'].values
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # IC均值柱状图
    bars1 = ax1.bar(factors, ic_means, color=colors, alpha=0.7)
    ax1.axhline(y=config.IC_THRESHOLD_EFFECTIVE, color='green', linestyle='--',
                label=f'有效阈值({config.IC_THRESHOLD_EFFECTIVE})')
    ax1.axhline(y=-config.IC_THRESHOLD_EFFECTIVE, color='green', linestyle='--')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_title('IC均值', fontsize=12)
    ax1.set_ylabel('IC')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 在柱上标注数值
    for bar, val in zip(bars1, ic_means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # IR柱状图
    bars2 = ax2.bar(factors, ir_vals, color=colors, alpha=0.7)
    ax2.axhline(y=0.1, color='green', linestyle='--', label='稳定阈值(|IR|>0.1)')
    ax2.axhline(y=-0.1, color='green', linestyle='--')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_title('IR值', fontsize=12)
    ax2.set_ylabel('IR')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars2, ir_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(config.OUTPUT_DIR, 'ic_ir_summary.png')
    plt.savefig(fig_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"[图表] IC/IR汇总图已保存: {fig_path}")
    return fig_path


def plot_feature_importance(importance_df: pd.DataFrame) -> str:
    """绘制因子重要性柱状图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    factors = importance_df['因子'].values
    importance = importance_df['重要性'].values
    pct = importance_df['重要性占比'].values

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(factors)))
    bars = ax.barh(factors, importance, color=colors)

    # 标注百分比
    for bar, p in zip(bars, pct):
        ax.text(bar.get_width() + importance.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{p:.1%}', va='center', fontsize=10)

    ax.set_xlim(0, importance.max() * 1.18)
    ax.set_xlabel('重要性（Gain）')
    ax.set_title('LGBM因子特征重要性', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()
    fig_path = os.path.join(config.OUTPUT_DIR, 'factor_weights.png')
    plt.savefig(fig_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"[图表] 因子重要性图已保存: {fig_path}")
    return fig_path


def plot_metrics_table(metrics: Dict) -> str:
    """绘制绩效指标表格图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # 构建表格数据
    table_data = []
    for k, v in metrics.items():
        table_data.append([k, str(v)])

    table = ax.table(cellText=table_data,
                     colLabels=['指标', '数值'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.4, 0.4])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # 设置表头样式
    for j in range(2):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # 交替行颜色
    for i in range(1, len(table_data) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[i, j].set_facecolor('#D9E2F3')

    ax.set_title('回测核心绩效指标', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    fig_path = os.path.join(config.OUTPUT_DIR, 'metrics_table.png')
    plt.savefig(fig_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"[图表] 指标表格图已保存: {fig_path}")
    return fig_path


def generate_all_plots(backtest_df: pd.DataFrame, ic_dict: Dict[str, pd.Series],
                       ir_df: pd.DataFrame, importance_df: pd.DataFrame,
                       metrics: Dict) -> list:
    """生成所有图表"""
    print("\n[绘图] 生成所有图表...")
    fig_paths = []

    fig_paths.append(plot_nav_vs_market(backtest_df))
    fig_paths.append(plot_drawdown(backtest_df))
    fig_paths.append(plot_ic_series(ic_dict))
    fig_paths.append(plot_ic_ir_summary(ir_df))
    fig_paths.append(plot_feature_importance(importance_df))
    fig_paths.append(plot_metrics_table(metrics))

    print(f"\n[绘图] 共生成{len(fig_paths)}张图表")
    return fig_paths
