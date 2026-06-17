"""Plot generation for experiment 1 outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


def _setup_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_weekly_sentiment(weekly: pd.DataFrame) -> Path:
    _setup_font()
    path = config.OUTPUT_DIR / "weekly_sentiment_counts.png"
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(weekly["week"], weekly["WeekPositive"], label="正面", color="#2f9e44")
    ax.bar(weekly["week"], weekly["WeekNeutral"], bottom=weekly["WeekPositive"], label="中性", color="#868e96")
    ax.bar(weekly["week"], weekly["WeekNegative"], bottom=weekly["WeekPositive"] + weekly["WeekNeutral"], label="负面", color="#d9480f")
    ax.set_title("周度新闻情绪数量")
    ax.set_xlabel("周末交易日")
    ax.set_ylabel("新闻数量")
    ax.legend(ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_herd_index(herd: pd.DataFrame) -> Path:
    _setup_font()
    path = config.OUTPUT_DIR / "herd_index_timeseries.png"
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(herd["week"], herd["H1t"], label="H1 情绪偏离", color="#1971c2")
    ax.plot(herd["week"], herd["H2t"], label="H2 分歧度", color="#f08c00")
    ax.plot(herd["week"], herd["H3t"], label="H3 羊群强度", color="#c92a2a", linewidth=2)
    ax.set_title("周度羊群效应指标")
    ax.set_xlabel("周末交易日")
    ax.legend(ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_market_relation(modeling: pd.DataFrame) -> Path:
    _setup_font()
    path = config.OUTPUT_DIR / "herd_vs_hs300_return.png"
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.plot(modeling["week"], modeling["H3t"], color="#c92a2a", label="H3 羊群强度", linewidth=2)
    ax2.plot(modeling["week"], modeling["return"], color="#1864ab", label="沪深300周收益率", linewidth=1.8)
    ax1.set_title("羊群效应指标与沪深300周收益率")
    ax1.set_xlabel("周末交易日")
    ax1.set_ylabel("H3")
    ax2.set_ylabel("周收益率")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_feature_importance(importance_df: pd.DataFrame) -> Path:
    _setup_font()
    path = config.OUTPUT_DIR / "feature_importance.png"
    top = importance_df.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["importance"], color="#1971c2")
    ax.set_title("LightGBM Gain 特征重要性 (Top 10)")
    ax.set_xlabel("Gain")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_shap_dependence(shap_values: np.ndarray, X_test: pd.DataFrame, best_feature: str) -> Path:
    _setup_font()
    path = config.OUTPUT_DIR / "shap_dependence.png"
    if best_feature not in X_test.columns:
        best_feature = X_test.columns[0]
    feat_idx = list(X_test.columns).index(best_feature)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X_test[best_feature].values, shap_values[:, feat_idx], alpha=0.7, color="#c92a2a", edgecolors="k", linewidths=0.3)
    ax.set_xlabel(best_feature)
    ax.set_ylabel(f"SHAP value ({best_feature})")
    ax.set_title(f"SHAP 依赖图: {best_feature}")
    z = np.polyfit(X_test[best_feature].values, shap_values[:, feat_idx], 1)
    x_line = np.linspace(X_test[best_feature].min(), X_test[best_feature].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "--", color="#495057", alpha=0.7, label="趋势线")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_residuals(test_data: pd.DataFrame) -> Path:
    _setup_font()
    path = config.OUTPUT_DIR / "residual_timeseries.png"
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test_data["week"], test_data["residual"], color="#495057", linewidth=1.2)
    ax.axhline(0, color="#c92a2a", linestyle="--", alpha=0.7)
    ax.set_title("测试集残差时序图")
    ax.set_xlabel("周末交易日")
    ax.set_ylabel("残差 (实际 - 预测)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_bidirectional_comparison(comparison: pd.DataFrame) -> Path:
    _setup_font()
    path = config.OUTPUT_DIR / "bidirectional_comparison.png"
    metrics = ["mse", "mae", "r2"]
    labels = ["MSE", "MAE", "R²"]
    if comparison.empty or "direction" not in comparison.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "数据不足，无法绘制双向对比", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
    fwd_row = comparison[comparison["direction"].str.contains("forward")]
    bwd_row = comparison[comparison["direction"].str.contains("backward")]
    if fwd_row.empty or bwd_row.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "数据不足，无法绘制双向对比", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
    fwd = fwd_row[metrics].iloc[0].values
    bwd = bwd_row[metrics].iloc[0].values
    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w / 2, fwd, w, label="H3→收益 (正向)", color="#1971c2")
    ax.bar(x + w / 2, bwd, w, label="收益→H3 (反向)", color="#c92a2a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("双向建模效果对比")
    ax.legend()
    for i, (f, b) in enumerate(zip(fwd, bwd)):
        ax.text(i - w / 2, f + 0.001, f"{f:.4f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, b + 0.001, f"{b:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def generate_all_plots(
    weekly: pd.DataFrame,
    herd: pd.DataFrame,
    modeling: pd.DataFrame,
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    shap_importance: pd.DataFrame,
    lgbm_importance: pd.DataFrame,
    test_pred_df: pd.DataFrame,
    comparison: pd.DataFrame,
) -> list[str]:
    best_feature = shap_importance.iloc[0]["feature"] if len(shap_importance) > 0 else "H3_lag1"
    paths = [
        plot_weekly_sentiment(weekly),
        plot_herd_index(herd),
        plot_market_relation(modeling),
        plot_feature_importance(lgbm_importance),
        plot_shap_dependence(shap_values, X_test, best_feature),
        plot_residuals(test_pred_df),
        plot_bidirectional_comparison(comparison),
    ]
    return [str(p.relative_to(config.ROOT)) for p in paths]
