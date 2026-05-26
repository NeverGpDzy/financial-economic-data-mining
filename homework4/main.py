"""Homework 4: Multi-factor quantitative stock selection full pipeline."""

from __future__ import annotations

import json
import sys
import io
from pathlib import Path

# Fix Windows GBK encoding for stdout
if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from homework4.config import (
    DATA_DIR, OUTPUT_DIR,
    TRAIN_START, TRAIN_END, TEST_START, TEST_END,
    INIT_CAPITAL, FEE, TOP_N, MAX_SINGLE_WEIGHT,
    FACTORS, FACTOR_NAMES,
    IC_MEAN_THRESHOLD, IC_EXCELLENT_THRESHOLD, IR_THRESHOLD,
    RF_MONTHLY, MARKET_NAME, MARKET_CODE,
)
from homework4.data import get_data
from homework4.models import (
    capm_screening,
    single_factor_test,
    factor_standardize,
    compute_ic_ir,
    multi_factor_regression,
)
from homework4.backtest import standardize_test_factors, select_top_stocks, run_backtest
from homework4.plots import (
    plot_nav_vs_market,
    plot_drawdown,
    plot_ic_series,
    plot_factor_weights,
    plot_ic_ir_summary,
    plot_metrics_table,
)


def pct(v: float) -> str:
    return f"{v:.2%}"


def main(refresh: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("作业四：量化全流程 — 因子挖掘与多因子选股回测")
    print("=" * 60)
    print(f"训练期: {TRAIN_START} ~ {TRAIN_END}")
    print(f"回测期: {TEST_START} ~ {TEST_END}")
    print(f"无风险利率(月): {RF_MONTHLY:.4f}")
    print(f"初始资金: {INIT_CAPITAL:,.0f} 元")
    print(f"交易成本(单边): {FEE:.1%}")
    print(f"选股数量: Top {TOP_N}")
    print(f"因子池: {', '.join(FACTORS)}")
    print(f"  SMB  = 规模因子 (总市值)")
    print(f"  PE_inv = 价值因子 (1/PE_TTM)")
    print(f"  Quality = 销售净利率 × (经营现金流/净利润) × 净利润增速")

    # ================================================================
    # Module 1: Data fetching & preprocessing
    # ================================================================
    print("\n" + "=" * 60)
    print("模块1：数据获取与预处理")
    print("=" * 60)

    train_df, test_df, mkt_df = get_data(refresh=refresh)

    print(f"\n训练集: {train_df['date'].min().date()} ~ {train_df['date'].max().date()}")
    print(f"  样本数: {len(train_df)}, 股票数: {train_df['code'].nunique()}")
    print(f"回测集: {test_df['date'].min().date()} ~ {test_df['date'].max().date()}")
    print(f"  样本数: {len(test_df)}, 股票数: {test_df['code'].nunique()}")

    # ================================================================
    # Module 2: CAPM screening + single factor test
    # ================================================================
    print("\n" + "=" * 60)
    print("模块2：单因子有效性检验")
    print("=" * 60)

    # Step 1: CAPM individual stock screening
    print("\n--- 步骤1: CAPM个股筛选 ---")
    capm_results = capm_screening(train_df, mkt_df)
    valid_stocks = capm_results[capm_results["beta_significant"]]
    n_valid = len(valid_stocks)
    n_total = len(capm_results)
    print(f"CAPM β显著(stock): {n_valid}/{n_total}")
    print(f"  均通过 → 全部保留用于因子检验")

    # Step 2-3: Single factor cross-sectional regression
    print("\n--- 步骤2-3: 单因子横截面回归 ---")
    factor_test_results = single_factor_test(train_df, FACTORS)
    valid_factors = factor_test_results["valid_factors"]
    factor_details = factor_test_results["factor_details"]

    for f in FACTORS:
        info = factor_details[f]
        status = "✓ 有效" if info["valid"] else "✗ 无效"
        fm_t_str = f"{info['fm_t']:.4f}" if not pd.isna(info.get('fm_t', np.nan)) else "N/A"
        print(f"  {FACTOR_NAMES.get(f, f)}: β均值={info['beta_mean']:.6f}, "
              f"FM_t={fm_t_str}, p={info.get('fm_p', np.nan):.4f} [{status}]")

    print(f"\n有效因子: {valid_factors}")
    if not valid_factors:
        print("⚠ 无有效因子，将所有因子纳入后续分析")
        valid_factors = FACTORS.copy()

    # ================================================================
    # Module 3: IC/IR factor quality check
    # ================================================================
    print("\n" + "=" * 60)
    print("模块3：因子质检 — IC/IR计算")
    print("=" * 60)

    # Standardize
    train_std = factor_standardize(train_df, valid_factors)
    test_std = standardize_test_factors(test_df, train_df, valid_factors)

    # IC/IR
    ic_results = compute_ic_ir(train_std, valid_factors)
    ic_summary_rows = []
    for f in valid_factors:
        info = ic_results[f]
        ir_status = "✓ IR通过" if abs(info["IR"]) > IR_THRESHOLD else "✗ IR不足"
        ic_summary_rows.append({
            "因子": FACTOR_NAMES.get(f, f),
            "IC均值": info["IC_mean"],
            "IC标准差": info["IC_std"],
            "IR": info["IR"],
            "评级": info["grade"],
            "IR判定": ir_status,
        })
    ic_df = pd.DataFrame(ic_summary_rows)
    print(ic_df.to_string(index=False))

    # ================================================================
    # Module 4: Multi-factor regression static weighting
    # ================================================================
    print("\n" + "=" * 60)
    print("模块4：多因子线性回归静态赋权")
    print("=" * 60)

    weights, score_func = multi_factor_regression(train_std, valid_factors)

    print(f"\n回归结果 (n={weights['nobs']}, R²={weights['r2']:.4f}):")
    print(f"  截距 α = {weights['alpha']:.6f} (p={weights['alpha_pvalue']:.4f})")
    for f in valid_factors:
        p_key = f"{f}_pvalue"
        sig = "显著" if weights.get(p_key, 1.0) < 0.05 else "不显著"
        print(f"  {FACTOR_NAMES.get(f, f)}: w = {weights[f]:.6f} (p={weights.get(p_key, np.nan):.4f}) [{sig}]")

    print(f"\n综合得分公式:")
    terms = []
    for f in valid_factors:
        terms.append(f"({weights[f]:.6f})×{f}_std")
    print(f"  Score = {' + '.join(terms)}")

    # ================================================================
    # Module 5: Cross-sectional stock selection + backtest
    # ================================================================
    print("\n" + "=" * 60)
    print("模块5：截面选股 + 样本外回测")
    print("=" * 60)

    result, metrics, trade_records = run_backtest(
        test_std, mkt_df, score_func, valid_factors,
        init_capital=INIT_CAPITAL, fee=FEE, top_n=TOP_N,
        max_single_weight=MAX_SINGLE_WEIGHT,
    )

    # Print first/last few months of scoring
    print("\n--- 截面打分选股示例 (前3月) ---")
    first_months = sorted(test_std["date"].unique())[:3]
    for month in first_months:
        sub = test_std[test_std["date"] == month].copy()
        ranked = select_top_stocks(sub, score_func, top_n=TOP_N)
        print(f"\n  {str(month)[:7]}:")
        for _, row in ranked.head(TOP_N).iterrows():
            print(f"    {row['rank']}. {row['code']} {row['name']} 得分={row['score']:.6f}")

    # Monthly rebalancing records
    print("\n--- 月度调仓记录 ---")
    trade_df = pd.DataFrame(trade_records)
    trade_df.to_csv(OUTPUT_DIR / "trade_log.csv", index=False, encoding="utf-8-sig")
    print(trade_df.to_string(index=False))

    # Backtest metrics
    print("\n--- 回测指标 ---")
    metric_rows = [
        ["策略累计收益率", pct(metrics["累计收益率"])],
        ["策略年化收益率", pct(metrics["年化收益率"])],
        ["策略最大回撤", pct(metrics["最大回撤"])],
        ["月度胜率", pct(metrics["月度胜率"])],
        ["超额收益(vs上证指数)", pct(metrics["超额收益(vs上证指数)"])],
        ["上证指数累计收益", pct(metrics["上证指数累计收益"])],
        ["上证指数年化收益", pct(metrics["上证指数年化收益"])],
        ["上证指数最大回撤", pct(metrics["上证指数最大回撤"])],
        ["夏普比率", f"{metrics['夏普比率']:.4f}"],
        ["卡玛比率", f"{metrics['卡玛比率']:.4f}"],
        ["调仓次数", str(metrics["调仓次数"])],
        ["回测月数", str(metrics["回测月数"])],
    ]
    for name, val in metric_rows:
        print(f"  {name}: {val}")

    # Save backtest result
    result.to_csv(OUTPUT_DIR / "backtest_result.csv", index_label="date", encoding="utf-8-sig")

    # ================================================================
    # Plots
    # ================================================================
    print("\n" + "=" * 60)
    print("绘图与输出")
    print("=" * 60)

    plot_nav_vs_market(result, OUTPUT_DIR / "nav_vs_market.png")
    plot_drawdown(result, OUTPUT_DIR / "drawdown.png")
    if ic_results:
        plot_ic_series(ic_results, OUTPUT_DIR / "ic_series.png")
        plot_ic_ir_summary(ic_results, OUTPUT_DIR / "ic_ir_summary.png")
    plot_factor_weights(weights, valid_factors, OUTPUT_DIR / "factor_weights.png")
    plot_metrics_table(metrics, OUTPUT_DIR / "metrics_table.png")

    # ================================================================
    # Summary JSON
    # ================================================================
    summary = {
        "训练期": f"{TRAIN_START} ~ {TRAIN_END}",
        "回测期": f"{TEST_START} ~ {TEST_END}",
        "因子池": FACTORS,
        "有效因子": valid_factors,
        "单因子检验": {f: {k: v for k, v in d.items() if k not in ("beta_series", "p_series")}
                     for f, d in factor_details.items()},
        "IC_IR": {
            f: {k: v for k, v in d.items() if k != "ic_series"}
            for f, d in ic_results.items()
        },
        "因子权重": {f: weights[f] for f in valid_factors},
        "回归截距": weights["alpha"],
        "回归R2": weights["r2"],
        "回测指标": {k: v for k, v in metrics.items()},
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )

    print("  图表已保存到 outputs/homework4/")
    print("\n" + "=" * 60)
    print("作业四完成！")
    print("=" * 60)


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
