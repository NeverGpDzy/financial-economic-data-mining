"""可视化模块。"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_prediction_vs_actual(
    y_true: np.ndarray,
    y_pred_dict: dict[str, np.ndarray],
    title: str = "模型预测值 vs 真实值",
    save_path: str | None = None,
):
    """绘制各模型预测值与真实值对比图。"""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_true, label="真实值", alpha=0.7, linewidth=1)
    for name, y_pred in y_pred_dict.items():
        ax.plot(y_pred, label=name, alpha=0.7, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("样本序号")
    ax.set_ylabel("价格收益率")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_backtest_curves(
    backtest_results: dict[str, dict],
    initial_capital: float = 1_000_000,
    title: str = "2025年回测累计收益曲线",
    save_path: str | None = None,
):
    """绘制各模型回测资金收益曲线。"""
    fig, ax = plt.subplots(figsize=(14, 5))
    for name, result in backtest_results.items():
        daily = result["daily_capital"]
        ax.plot(daily / initial_capital, label=f"{name} (收益:{result['total_return']:.2%})", linewidth=1.2)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("交易日")
    ax.set_ylabel("净值 (初始=1.0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_backtest_summary(backtest_results: dict[str, dict]):
    """打印回测结果汇总表。"""
    rows = []
    for name, r in backtest_results.items():
        rows.append({
            "模型": name,
            "累计收益": f"{r['total_return']:.2%}",
            "最大回撤": f"{r['max_drawdown']:.2%}",
            "胜率": f"{r['win_rate']:.2%}",
            "交易次数": r["trades"],
        })
    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("回测结果汇总")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
