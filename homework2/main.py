"""Homework 2 main workflow: CAPM fitting and high-alpha backtest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from homework2.backtest import backtest_metrics, buy_and_hold_curve, run_horizon_backtests
from homework2.capm import build_capm_table, select_best_alpha
from homework2.config import (
    ALPHA_SIGNIFICANCE_LEVEL,
    BACKTEST_END,
    BACKTEST_START,
    DATA_DIR,
    EXTENDED_BACKTEST_END,
    EXTENDED_BACKTEST_PERIODS,
    INITIAL_CAPITAL,
    MARKET_CODE,
    MARKET_NAME,
    OUTPUT_DIR,
    RISK_FREE_DAILY,
    STOCKS,
    TRAIN_END,
    TRAIN_START,
)
from homework2.data import fetch_many
from homework2.plots import plot_backtest_curve, plot_horizon_comparison


def pct(value: float) -> str:
    return f"{value:.2%}"


def main(refresh: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("===== 作业二：CAPM 拟合与回测 =====")
    print(f"无风险日收益率 Rf_daily = 0.03 / 250 = {RISK_FREE_DAILY:.8f}")

    all_codes = {**STOCKS, MARKET_NAME: MARKET_CODE}

    train_data = fetch_many(all_codes, TRAIN_START, TRAIN_END, DATA_DIR / "train", refresh)
    market_train = train_data.pop(MARKET_NAME)

    capm_table = build_capm_table(
        train_data,
        market_train["return"],
        RISK_FREE_DAILY,
        ALPHA_SIGNIFICANCE_LEVEL,
    )
    capm_csv = OUTPUT_DIR / "capm_results.csv"
    capm_table.to_csv(capm_csv, index=False, encoding="utf-8-sig")

    best = select_best_alpha(capm_table)
    best_stock = str(best["stock"])
    best_code = STOCKS[best_stock]

    print("\n===== CAPM 拟合结果（2020-2022）=====")
    print(
        capm_table[
            [
                "stock",
                "alpha_daily",
                "alpha_annualized",
                "beta",
                "alpha_pvalue",
                "beta_pvalue",
                "r_squared",
                "alpha_significant_10pct",
            ]
        ].to_string(index=False)
    )
    print(
        f"\n显著性阈值 p < {ALPHA_SIGNIFICANCE_LEVEL:.1f}，"
        f"入选股票：{best_stock}，alpha = {best['alpha_daily']:.8f}，"
        f"年化 alpha = {best['alpha_annualized']:.2%}，p = {best['alpha_pvalue']:.4f}"
    )

    backtest_codes = {best_stock: best_code, MARKET_NAME: MARKET_CODE}
    backtest_data = fetch_many(backtest_codes, BACKTEST_START, BACKTEST_END, DATA_DIR / "backtest", refresh)
    selected_df = backtest_data[best_stock]
    market_backtest = backtest_data[MARKET_NAME]

    curve = buy_and_hold_curve(selected_df, market_backtest, INITIAL_CAPITAL)
    metrics = backtest_metrics(curve)

    curve_csv = OUTPUT_DIR / "backtest_curve.csv"
    curve.to_csv(curve_csv, index_label="date", encoding="utf-8-sig")
    plot_path = OUTPUT_DIR / "backtest_curve.png"
    plot_backtest_curve(curve, best_stock, plot_path)

    extended_data = fetch_many(
        backtest_codes,
        BACKTEST_START,
        EXTENDED_BACKTEST_END,
        DATA_DIR / "backtest_extended",
        refresh,
    )
    extended_stock_df = extended_data[best_stock]
    extended_market_df = extended_data[MARKET_NAME]
    extended_table, extended_curves = run_horizon_backtests(
        extended_stock_df,
        extended_market_df,
        INITIAL_CAPITAL,
        EXTENDED_BACKTEST_PERIODS,
    )
    extended_csv = OUTPUT_DIR / "extended_backtest_results.csv"
    extended_table.to_csv(extended_csv, index=False, encoding="utf-8-sig")

    extended_curve = extended_curves[EXTENDED_BACKTEST_PERIODS[-1][0]]
    extended_curve_csv = OUTPUT_DIR / "extended_backtest_curve.csv"
    extended_curve.to_csv(extended_curve_csv, index_label="date", encoding="utf-8-sig")
    extended_plot_path = OUTPUT_DIR / "extended_backtest_curve.png"
    plot_backtest_curve(extended_curve, f"{best_stock} 延长持有", extended_plot_path)
    horizon_plot_path = OUTPUT_DIR / "extended_horizon_comparison.png"
    plot_horizon_comparison(extended_table, best_stock, horizon_plot_path)

    summary = {
        "risk_free_daily": RISK_FREE_DAILY,
        "significance_level": ALPHA_SIGNIFICANCE_LEVEL,
        "selected_stock": best_stock,
        "selected_code": best_code,
        "selected_alpha_daily": float(best["alpha_daily"]),
        "selected_alpha_annualized": float(best["alpha_annualized"]),
        "selected_alpha_pvalue": float(best["alpha_pvalue"]),
        "selected_beta": float(best["beta"]),
        "selected_r_squared": float(best["r_squared"]),
        "backtest_metrics": metrics,
        "extended_backtest_results": extended_table.to_dict(orient="records"),
        "files": {
            "capm_results": str(capm_csv.relative_to(ROOT)),
            "backtest_curve": str(curve_csv.relative_to(ROOT)),
            "backtest_plot": str(plot_path.relative_to(ROOT)),
            "extended_backtest_results": str(extended_csv.relative_to(ROOT)),
            "extended_backtest_curve": str(extended_curve_csv.relative_to(ROOT)),
            "extended_backtest_plot": str(extended_plot_path.relative_to(ROOT)),
            "extended_horizon_plot": str(horizon_plot_path.relative_to(ROOT)),
        },
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 买入持有回测（2023-2024）=====")
    print(f"{best_stock} 累计收益率：{pct(metrics['stock_total_return'])}")
    print(f"{MARKET_NAME} 累计收益率：{pct(metrics['market_total_return'])}")
    print(f"相对沪深300超额收益：{pct(metrics['excess_total_return'])}")
    print(f"{best_stock} 最大回撤：{pct(metrics['stock_max_drawdown'])}")
    print(f"{MARKET_NAME} 最大回撤：{pct(metrics['market_max_drawdown'])}")
    print(f"{best_stock} 卡玛比率：{metrics['stock_calmar']:.4f}")
    print(f"{MARKET_NAME} 卡玛比率：{metrics['market_calmar']:.4f}")

    print("\n===== 补充实验：延长泸州老窖持有期 =====")
    for _, row in extended_table.iterrows():
        print(
            f"{row['period']}（{row['start_date']} ~ {row['end_date']}）："
            f"{best_stock}累计收益={pct(row['stock_total_return'])}，"
            f"{MARKET_NAME}累计收益={pct(row['market_total_return'])}，"
            f"超额收益={pct(row['excess_total_return'])}，"
            f"最大回撤={pct(row['stock_max_drawdown'])}"
        )
    print(f"\n结果已保存到：{OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
