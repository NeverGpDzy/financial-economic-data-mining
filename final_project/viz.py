"""可视化模块：因子质检、LGBM赋权、回测净值、鲁棒性与风险控制图表。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg
from .backtest import BacktestResult
from .industry_codes import name_of

_FONT_OK = False


def setup_font():
    global _FONT_OK
    if _FONT_OK:
        return
    for f in cfg.CHINESE_FONT:
        try:
            matplotlib.rcParams["font.sans-serif"] = [f] + matplotlib.rcParams["font.sans-serif"]
            break
        except Exception:
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False
    _FONT_OK = True


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=cfg.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 因子质检
# --------------------------------------------------------------------------- #
def plot_ic_ir(qc: dict, path: Path):
    setup_font()
    summary = qc["ic_summary"].copy()
    summary = summary.reindex(cfg.FACTOR_Z)
    labels = [cfg.FACTOR_LABELS[f.replace("_z", "")] for f in summary.index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#2E86AB" if v >= 0 else "#C73E1D" for v in summary["ic_mean"]]
    axes[0].bar(range(len(labels)), summary["ic_mean"], color=colors)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axes[0].set_title("因子月度 IC 均值（样本内 2023-2024）")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("IC 均值")
    colors2 = ["#2E86AB" if v >= 0 else "#C73E1D" for v in summary["ir"]]
    axes[1].bar(range(len(labels)), summary["ir"], color=colors2)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axes[1].set_title("因子 IR（IC均值/标准差）")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].axhline(cfg.IR_FLOOR, color="gray", ls="--", lw=0.8, label=f"有效性参考线 |IR|={cfg.IR_FLOOR}")
    axes[1].axhline(-cfg.IR_FLOOR, color="gray", ls="--", lw=0.8)
    axes[1].set_ylabel("IR")
    axes[1].legend(fontsize=8)
    _save(fig, path)


def plot_collinearity(corr: pd.DataFrame, path: Path):
    setup_font()
    labels = [cfg.FACTOR_LABELS[f.replace("_z", "")] for f in corr.index]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"因子共线性矩阵（阈值 {cfg.COLLINEAR_THRESHOLD}）")
    _save(fig, path)


def plot_lgbm_importances(importances: pd.Series, path: Path):
    setup_font()
    imp = importances.reindex(cfg.FACTOR_Z)
    labels = [cfg.FACTOR_LABELS[f.replace("_z", "")] for f in imp.index]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(labels, imp.values, color="#2E86AB")
    ax.set_ylabel("归一化重要度（因子权重）")
    ax.set_title("LGBM 因子赋权结果（特征重要度）")
    for i, v in enumerate(imp.values):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    _save(fig, path)


# --------------------------------------------------------------------------- #
# 回测
# --------------------------------------------------------------------------- #
def plot_nav(result: BacktestResult, path: Path, title: str | None = None):
    setup_font()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(result.nav.index, result.nav.values, label=f"策略({result.name})", color="#C73E1D", lw=1.6)
    ax.plot(result.benchmark_nav.index, result.benchmark_nav.values, label="沪深300", color="#2E86AB", lw=1.4)
    ax.axhline(1.0, color="gray", lw=0.7, ls="--")
    ax.set_title(title or f"策略 vs 沪深300 净值曲线 — {result.name}")
    ax.set_ylabel("净值（基期=1）")
    ax.legend(loc="upper left")
    M = result.metrics
    txt = (f"年化 {M['annualized_return']:.1%} | 最大回撤 {M['max_drawdown']:.1%} | "
           f"夏普 {M['sharpe']:.2f} | 超额 {M['excess_cumulative']:.1%}")
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    _save(fig, path)


def plot_drawdown(result: BacktestResult, path: Path):
    setup_font()
    dd = result.nav / result.nav.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.fill_between(dd.index, dd.values, 0, color="#C73E1D", alpha=0.5)
    ax.set_title(f"策略回撤曲线 — {result.name}（最大回撤 {result.metrics['max_drawdown']:.1%}）")
    ax.set_ylabel("回撤")
    ax.axhline(-cfg.TARGET_MAX_DRAWDOWN, color="gray", ls="--", lw=0.8, label=f"目标线 -{cfg.TARGET_MAX_DRAWDOWN:.0%}")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_monthly_returns(result: BacktestResult, path: Path):
    setup_font()
    m = result.monthly_returns.copy()
    m["month"] = pd.to_datetime(m["fwd_date"]).dt.strftime("%Y-%m")
    x = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - 0.2, m["strategy_ret"], 0.4, label="策略", color="#C73E1D")
    ax.bar(x + 0.2, m["benchmark_ret"], 0.4, label="沪深300", color="#2E86AB")
    ax.set_xticks(x)
    ax.set_xticklabels(m["month"], rotation=30, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_title(f"月度收益对比 — {result.name}")
    ax.set_ylabel("月度收益")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_holdings_heatmap(result: BacktestResult, path: Path):
    setup_font()
    h = result.holdings.copy()
    if h.empty:
        return
    h["month"] = pd.to_datetime(h["signal_date"]).dt.strftime("%Y-%m")
    piv = h.pivot_table(index="industry", columns="month", values="weight", aggfunc="sum").fillna(0)
    piv = piv.reindex(piv.sum(axis=1).sort_values(ascending=False).index)
    ylabels = [name_of(c) for c in piv.index]
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(piv.values, aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=40, ha="right", fontsize=8)
    ax.set_title(f"月度持仓热力图 — {result.name}")
    _save(fig, path)


def plot_robustness_compare(results: dict, path: Path, title: str):
    setup_font()
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#C73E1D", "#2E86AB", "#2A9D8F", "#E76F51", "#8E7DBE"]
    for i, (k, r) in enumerate(results.items()):
        ax.plot(r.nav.index, r.nav.values, label=r.name, color=colors[i % len(colors)], lw=1.5)
    ax.axhline(1.0, color="gray", lw=0.7, ls="--")
    ax.set_title(title)
    ax.set_ylabel("净值（基期=1）")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_risk_spectrum(table: pd.DataFrame, path: Path):
    setup_font()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2E86AB" if (ok_r and ok_d) else "#C73E1D"
              for ok_r, ok_d in zip(table["target_return_ok"], table["target_dd_ok"])]
    ax.scatter(table["max_drawdown"], table["annualized_return"], s=120, c=colors, zorder=3)
    for _, row in table.iterrows():
        ax.annotate(row["config"], (row["max_drawdown"], row["annualized_return"]),
                    fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.axvline(-cfg.TARGET_MAX_DRAWDOWN, color="gray", ls="--", lw=0.8, label=f"回撤上限 -{cfg.TARGET_MAX_DRAWDOWN:.0%}")
    ax.axhspan(cfg.TARGET_ANN_RETURN[0], cfg.TARGET_ANN_RETURN[1], color="#2A9D8F", alpha=0.12, label="目标收益区间 12%-18%")
    ax.set_xlabel("最大回撤")
    ax.set_ylabel("年化收益")
    ax.set_title("风险控制谱系：回撤触发减仓的收益-回撤权衡（绿=双达标）")
    ax.legend(fontsize=8)
    _save(fig, path)
