"""Visualization utilities for Homework 6."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from . import config


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {
    "A": "#1B7F5A",
    "B": "#D19A32",
    "C": "#A23E48",
    "ml": "#2F5597",
    "rule": "#6B7280",
    "bench": "#111827",
}


def _save(fig: plt.Figure, filename: str) -> str:
    path = config.OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_ic_ir(ic_ir: pd.DataFrame) -> str:
    fig, ax1 = plt.subplots(figsize=(12, 6))
    df = ic_ir.sort_values("IC均值")
    y = np.arange(len(df))
    ax1.barh(y, df["IC均值"], color="#2F5597", alpha=0.82, label="IC均值")
    ax1.axvline(config.IC_THRESHOLD, color="#A23E48", linestyle="--", linewidth=1.2, label="IC阈值0.02")
    ax1.set_yticks(y)
    ax1.set_yticklabels(df["因子"])
    ax1.set_xlabel("Spearman IC均值")
    ax1.set_title("因子IC有效性检验")
    ax1.legend(loc="lower right")
    return _save(fig, "factor_ic_ir.png")


def plot_feature_importance(importance: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(12, 6))
    df = importance.sort_values("gain_importance")
    ax.barh(df["因子"], df["importance_share"], color="#1B7F5A", alpha=0.85)
    ax.set_xlabel("重要性占比")
    ax.set_title("LGBM因子重要度")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    return _save(fig, "feature_importance.png")


def plot_fcff_groups(fcff_summary: pd.DataFrame) -> str:
    sample = fcff_summary[fcff_summary["样本"].eq("测试集")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (scheme, sub) in zip(axes, sample.groupby("scheme")):
        sub = sub.set_index("group").reindex(config.GROUP_LABELS).reset_index()
        ax.bar(
            sub["group"],
            sub["fcff_growth_3y_ann_mean"],
            color=[COLORS[g] for g in config.GROUP_LABELS],
            alpha=0.88,
        )
        ax.set_title(scheme)
        ax.set_xlabel("分组")
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    axes[0].set_ylabel("未来3年FCFF年化增速均值")
    fig.suptitle("样本外FCFF分层回测：A/B/C组")
    return _save(fig, "fcff_group_backtest.png")


def plot_price_nav(price_nav: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(13, 6))
    sample = price_nav[
        (price_nav["trade_date"].dt.year >= config.TEST_START_YEAR)
        & (price_nav["trade_date"].dt.year <= config.TEST_END_YEAR)
        & price_nav["group"].eq("A")
    ].copy()
    for scheme, sub in sample.groupby("scheme"):
        sub = sub.sort_values("trade_date")
        rebased = sub["nav"] / sub["nav"].iloc[0]
        color = COLORS["ml"] if "LGBM" in scheme else COLORS["rule"]
        ax.plot(sub["trade_date"], rebased, label=f"{scheme} A组", linewidth=2.0, color=color)
    bench = sample[["trade_date", "benchmark_nav"]].drop_duplicates().sort_values("trade_date")
    ax.plot(
        bench["trade_date"],
        bench["benchmark_nav"] / bench["benchmark_nav"].iloc[0],
        label="沪深300基准",
        linewidth=1.8,
        color=COLORS["bench"],
        linestyle="--",
    )
    ax.set_title("样本外A组组合净值对比")
    ax.set_ylabel("净值（期初=1）")
    ax.legend()
    ax.grid(alpha=0.2)
    return _save(fig, "price_nav_comparison.png")


def plot_annual_returns(annual_returns: pd.DataFrame) -> str:
    sample = annual_returns[
        annual_returns["hold_year"].between(config.TEST_START_YEAR, config.TEST_END_YEAR)
        & annual_returns["group"].eq("A")
    ].copy()
    pivot = sample.pivot_table(index="hold_year", columns="scheme", values="annual_return")
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", ax=ax, color=["#6B7280", "#2F5597"])
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_title("A组年度收益率对比")
    ax.set_xlabel("持有年份")
    ax.set_ylabel("年度收益率")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(title="")
    return _save(fig, "annual_returns_A_group.png")


def plot_radar(strategy_panel: pd.DataFrame) -> str:
    latest_year = min(config.TEST_END_YEAR - 1, int(strategy_panel["year"].max()))
    latest = strategy_panel[
        (strategy_panel["year"].eq(latest_year))
        & (strategy_panel["scheme"].eq("方案B_LGBM打分"))
        & (strategy_panel["group"].eq("A"))
    ].copy()
    if latest.empty:
        latest = strategy_panel[strategy_panel["scheme"].eq("方案B_LGBM打分")].copy()
    profile = latest[config.FACTOR_Z_COLUMNS].mean()
    labels = [config.FACTOR_META[col.removesuffix("_z")]["label"].split(" ", 1)[0] for col in config.FACTOR_Z_COLUMNS]
    values = profile.to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0)
    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
    values = np.r_[values, values[0]]
    angles = np.r_[angles, angles[0]]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, color="#1B7F5A", linewidth=2)
    ax.fill(angles, values, color="#1B7F5A", alpha=0.22)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(f"{latest_year}年LGBM A组财务画像")
    ax.grid(alpha=0.35)
    return _save(fig, "top_group_radar.png")


def plot_metric_table(price_metrics: pd.DataFrame) -> str:
    sample = price_metrics[price_metrics["group"].eq("A")].copy()
    cols = ["scheme", "累计收益率", "年化收益率", "超额收益", "最大回撤", "夏普比率"]
    display = sample[cols].copy()
    for col in ["累计收益率", "年化收益率", "超额收益", "最大回撤"]:
        display[col] = display[col].map(lambda v: f"{v:.2%}")
    display["夏普比率"] = display["夏普比率"].map(lambda v: f"{v:.3f}")

    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.axis("off")
    table = ax.table(cellText=display.values, colLabels=display.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1B7F5A")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#F3F4F6" if row % 2 == 0 else "white")
    ax.set_title("A组股价回测核心指标", pad=12)
    return _save(fig, "price_metrics_table.png")


def generate_all_plots(
    ic_ir: pd.DataFrame,
    importance: pd.DataFrame,
    fcff_summary: pd.DataFrame,
    price_nav: pd.DataFrame,
    annual_returns: pd.DataFrame,
    strategy_panel: pd.DataFrame,
    price_metrics: pd.DataFrame,
) -> dict[str, str]:
    return {
        "factor_ic_ir": plot_ic_ir(ic_ir),
        "feature_importance": plot_feature_importance(importance),
        "fcff_group_backtest": plot_fcff_groups(fcff_summary),
        "price_nav_comparison": plot_price_nav(price_nav),
        "annual_returns_A_group": plot_annual_returns(annual_returns),
        "top_group_radar": plot_radar(strategy_panel),
        "price_metrics_table": plot_metric_table(price_metrics),
    }
