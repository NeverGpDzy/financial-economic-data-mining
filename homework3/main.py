"""Homework 3 main workflow: Two-Factor Model fitting and PEG backtest."""

from __future__ import annotations

import json
import sys
import io
from pathlib import Path

# Fix Windows GBK encoding for stdout
if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import baostock as bs
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from homework3.backtest import run_peg_backtest
from homework3.config import (
    BACKTEST_END,
    BACKTEST_START,
    COMMISSION_RATE,
    DATA_DIR,
    INITIAL_CAPITAL,
    MARKET_CODE,
    MARKET_NAME,
    OUTPUT_DIR,
    PEG_BUY_THRESHOLD,
    PEG_SELL_THRESHOLD,
    RISK_FREE_DAILY,
    SLIPPAGE_RATE,
    STOCKS,
    TRAIN_END,
    TRAIN_START,
    INDUSTRY,
)
from homework3.data import add_returns, fetch_backtest_data, fetch_close_price, fetch_train_data
from homework3.models import build_model_comparison
from homework3.plots import (
    plot_backtest_curve,
    plot_drawdown,
    plot_model_comparison,
    plot_pe_ttm_trends,
)


def pct(v: float) -> str:
    return f"{v:.2%}"


def main(refresh: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("作业三：二因子模型（CAPM + PE_TTM）")
    print("=" * 60)
    print(f"训练期: {TRAIN_START} ~ {TRAIN_END}")
    print(f"回测期: {BACKTEST_START} ~ {BACKTEST_END}")
    print(f"无风险日利率: {RISK_FREE_DAILY:.8f} (年化 1.5%)")
    print(f"初始资金: {INITIAL_CAPITAL:,.0f} 元")

    # ================================================================
    # Step 1: Data fetching
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 1: 数据获取与预处理")
    print("=" * 60)

    stock_data, growth_rates = fetch_train_data(
        STOCKS, MARKET_CODE, TRAIN_START, TRAIN_END, DATA_DIR, 2018, 2022, refresh,
    )

    # ================================================================
    # Step 2: Model fitting
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 2: CAPM 与二因子模型拟合")
    print("=" * 60)

    comparison = build_model_comparison(
        stock_data, "mkt_return_w", "peTTM_norm", RISK_FREE_DAILY,
    )
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")

    # Print comparison table
    display_cols = [
        "股票",
        "CAPM_α", "CAPM_β(mkt)", "CAPM_α_p值", "CAPM_R²",
        "二因子_α", "二因子_β(mkt)", "二因子_β(pe)", "二因子_α_p值", "二因子_β(pe)_p值", "二因子_R²",
        "α变化", "R²提升",
    ]
    print("\n模型对比结果:")
    fmt = comparison[display_cols].copy()
    for col in fmt.columns:
        if col == "股票":
            continue
        if "p值" in col:
            fmt[col] = fmt[col].apply(lambda x: f"{x:.4f}")
        elif "R²" in col or "变化" in col or "提升" in col:
            fmt[col] = fmt[col].apply(lambda x: f"{x:.4f}")
        elif "β" in col:
            fmt[col] = fmt[col].apply(lambda x: f"{x:.4f}")
        else:
            fmt[col] = fmt[col].apply(lambda x: f"{x:.6f}")
    print(fmt.to_string(index=False))

    # Alpha ranking for backtest
    alpha_rank = dict(zip(comparison["股票"], comparison["alpha_rank"]))

    # ================================================================
    # Step 3: Analysis
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 3: 数据分析")
    print("=" * 60)

    print("\n【1】PE_TTM 因子暴露分析 (β₂):")
    for _, row in comparison.iterrows():
        name = row["股票"]
        beta_pe = row["二因子_β(pe)"]
        p_val = row["二因子_β(pe)_p值"]
        sig = "显著" if p_val < 0.10 else "不显著"
        direction = "正相关" if beta_pe > 0 else "负相关"
        industry = INDUSTRY.get(name, "")
        print(f"  {name}({industry}): β₂={beta_pe:.4f}, p={p_val:.4f} [{sig}] — PE_TTM与收益{direction}")

    print("\n【2】CAPM vs 二因子模型 Alpha 对比:")
    for _, row in comparison.iterrows():
        name = row["股票"]
        a_capm = row["CAPM_α"]
        a_2f = row["二因子_α"]
        delta = row["α变化"]
        r2_capm = row["CAPM_R²"]
        r2_2f = row["二因子_R²"]
        r2_gain = row["R²提升"]
        change = "减少" if abs(a_2f) < abs(a_capm) else "增加"
        print(f"  {name}: CAPM α={a_capm:.6f} → 二因子 α={a_2f:.6f} (α{change}), R² {r2_capm:.4f} → {r2_2f:.4f} (+{r2_gain:.4f})")

    print("\n【3】行业差异分析:")
    for _, row in comparison.iterrows():
        name = row["股票"]
        industry = INDUSTRY.get(name, "")
        beta_pe = row["二因子_β(pe)"]
        print(f"  {name}({industry}): β_pe={beta_pe:.4f}")

    # ================================================================
    # Step 4: PEG Backtest
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 4: PEG 策略回测 (2023-2025)")
    print("=" * 60)

    # Get PE bounds from training data
    pe_bounds = {}
    for name in STOCKS:
        pe_min = float(stock_data[name]["pe_ttm_min"].iloc[0])
        pe_max = float(stock_data[name]["pe_ttm_max"].iloc[0])
        pe_bounds[name] = (pe_min, pe_max)
        print(f"  {name} PE_TTM 训练期范围: [{pe_min:.2f}, {pe_max:.2f}]")

    bt_stock_data, peg_df, _ = fetch_backtest_data(
        STOCKS, MARKET_CODE, BACKTEST_START, BACKTEST_END, DATA_DIR,
        pe_bounds, growth_rates, alpha_rank, refresh,
    )

    # PEG statistics
    print("\nPEG 统计:")
    for name in STOCKS:
        if name in peg_df.columns:
            peg = peg_df[name].dropna()
            if len(peg) > 0:
                print(f"  {name}: 均值={peg.mean():.4f}, 中位数={peg.median():.4f}, "
                      f"<{PEG_BUY_THRESHOLD}天数={(peg < PEG_BUY_THRESHOLD).sum()}, "
                      f">{PEG_SELL_THRESHOLD}天数={(peg > PEG_SELL_THRESHOLD).sum()}")

    # Fetch market backtest data for comparison
    bs.login()
    mkt_raw = fetch_close_price(MARKET_CODE, BACKTEST_START, BACKTEST_END)
    bs.logout()
    mkt_df = add_returns(mkt_raw)

    result, metrics = run_peg_backtest(
        bt_stock_data, peg_df, alpha_rank, mkt_df,
        INITIAL_CAPITAL, COMMISSION_RATE, SLIPPAGE_RATE,
        PEG_BUY_THRESHOLD, PEG_SELL_THRESHOLD,
    )
    result.to_csv(OUTPUT_DIR / "backtest_result.csv", index_label="date", encoding="utf-8-sig")

    # Trade log
    trades = result[result["action"] != "hold"]
    trade_log_path = OUTPUT_DIR / "trade_log.csv"
    trades.to_csv(trade_log_path, index_label="date", encoding="utf-8-sig")

    print(f"\n回测结果:")
    print(f"  策略累计收益率: {pct(metrics['strategy_total_return'])}")
    print(f"  策略年化收益率: {pct(metrics['strategy_annual_return'])}")
    print(f"  策略最大回撤: {pct(metrics['strategy_max_drawdown'])}")
    print(f"  策略卡玛比率: {metrics['strategy_calmar']:.4f}")
    print(f"  市场累计收益率: {pct(metrics['market_total_return'])}")
    print(f"  市场年化收益率: {pct(metrics['market_annual_return'])}")
    print(f"  市场最大回撤: {pct(metrics['market_max_drawdown'])}")
    print(f"  市场卡玛比率: {metrics['market_calmar']:.4f}")
    print(f"  交易次数: {metrics['trade_count']}")
    print(f"  持仓天数: {metrics['holding_days']} ({metrics['holding_pct']:.1f}%)")

    # ================================================================
    # Step 5: Plotting
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 5: 绘图")
    print("=" * 60)

    plot_backtest_curve(result, "PEG策略", "上证指数", OUTPUT_DIR / "backtest_curve.png")
    plot_drawdown(result, OUTPUT_DIR / "drawdown.png")
    plot_model_comparison(comparison, OUTPUT_DIR)
    plot_pe_ttm_trends(stock_data, OUTPUT_DIR)

    print("  图表已保存到 outputs/homework3/")

    # ================================================================
    # Summary
    # ================================================================
    summary = {
        "risk_free_daily": RISK_FREE_DAILY,
        "training_period": f"{TRAIN_START} ~ {TRAIN_END}",
        "backtest_period": f"{BACKTEST_START} ~ {BACKTEST_END}",
        "initial_capital": INITIAL_CAPITAL,
        "commission": COMMISSION_RATE,
        "slippage": SLIPPAGE_RATE,
        "peg_thresholds": {"buy": PEG_BUY_THRESHOLD, "sell": PEG_SELL_THRESHOLD},
        "growth_rates": {k: v for k, v in growth_rates.items()},
        "model_comparison": comparison[display_cols].to_dict(orient="records"),
        "backtest_metrics": metrics,
        "files": {
            "model_comparison": str((OUTPUT_DIR / "model_comparison.csv").relative_to(ROOT)),
            "backtest_result": str((OUTPUT_DIR / "backtest_result.csv").relative_to(ROOT)),
            "trade_log": str((OUTPUT_DIR / "trade_log.csv").relative_to(ROOT)),
            "backtest_curve": str((OUTPUT_DIR / "backtest_curve.png").relative_to(ROOT)),
            "drawdown": str((OUTPUT_DIR / "drawdown.png").relative_to(ROOT)),
            "model_comparison_plot": str((OUTPUT_DIR / "model_comparison.png").relative_to(ROOT)),
            "pe_ttm_trends": str((OUTPUT_DIR / "pe_ttm_trends.png").relative_to(ROOT)),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("全部完成！结果保存在 outputs/homework3/")
    print("=" * 60)


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
