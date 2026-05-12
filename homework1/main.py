"""作业1 主流程：使用机器学习进行技术分析。

流程：
1. 从 baostock 获取股票数据
2. 构建滚动时序特征
3. 2024 年数据划分训练/测试集，训练评估三个模型
4. 用全部 2024 年数据训练最终模型
5. 对 2025 年数据进行量化回测
6. 可视化结果
"""

import sys
from pathlib import Path

# 将项目根目录加入路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data_fetcher import fetch_all_stocks
from homework1.config import (
    STOCK_CODES, START_DATE, END_DATE, WINDOW,
    INITIAL_CAPITAL, COMMISSION, BUY_THRESHOLD, SELL_THRESHOLD,
)
from homework1.features import build_features
from homework1.models import train_and_evaluate, get_models
from homework1.backtest import run_backtest
from homework1.plots import plot_prediction_vs_actual, plot_backtest_curves, print_backtest_summary

import numpy as np
import pandas as pd


def main():
    # ======== 1. 数据获取 ========
    print("=" * 50)
    print("任务1：数据获取")
    print("=" * 50)

    data_dir = ROOT / "data" / "homework1"
    stock_data = fetch_all_stocks(STOCK_CODES, START_DATE, END_DATE, save_dir=str(data_dir))

    selected_code = STOCK_CODES[0]
    print(f"\n选定股票: {selected_code}")
    df = stock_data[selected_code]
    print(f"数据量: {len(df)} 条 ({df.index[0].date()} ~ {df.index[-1].date()})")

    # ======== 2. 特征构建 ========
    print("\n" + "=" * 50)
    print("任务2：时序特征构建")
    print("=" * 50)

    df_feat, feature_cols = build_features(df, window=WINDOW)
    print(f"特征数: {len(feature_cols)}")
    print(f"样本数: {len(df_feat)}")

    # ======== 3. 数据划分 ========
    print("\n" + "=" * 50)
    print("任务3：数据划分")
    print("=" * 50)

    mask_2024 = df_feat.index < "2025-01-01"
    df_2024 = df_feat[mask_2024]
    X_2024 = df_2024[feature_cols]
    y_2024 = df_2024["label"]

    split_idx = int(len(df_2024) * 0.8)
    X_train, X_test = X_2024.iloc[:split_idx], X_2024.iloc[split_idx:]
    y_train, y_test = y_2024.iloc[:split_idx], y_2024.iloc[split_idx:]
    print(f"2024 训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条")

    mask_2025 = df_feat.index >= "2025-01-01"
    df_2025 = df_feat[mask_2025]
    X_2025 = df_2025[feature_cols]
    y_2025 = df_2025["label"]
    print(f"2025 回测集: {len(X_2025)} 条")

    # ======== 4. 模型训练与评估 ========
    print("\n" + "=" * 50)
    print("任务4：模型训练与评估（2024年数据）")
    print("=" * 50)

    results_2024 = train_and_evaluate(X_train, y_train, X_test, y_test)

    print("\n使用全部 2024 年数据训练最终模型...")
    final_models = get_models()
    for name, model in final_models.items():
        model.fit(X_2024, y_2024)
    print("最终模型训练完成。")

    # ======== 5. 量化回测 ========
    print("\n" + "=" * 50)
    print("任务5：2025年量化回测")
    print("=" * 50)

    backtest_results = {}
    pred_dict = {}

    for name, model in final_models.items():
        y_pred_2025 = model.predict(X_2025)
        pred_dict[name] = y_pred_2025

        bt = run_backtest(
            df_2025, y_pred_2025,
            initial_capital=INITIAL_CAPITAL,
            commission=COMMISSION,
            buy_threshold=BUY_THRESHOLD,
            sell_threshold=SELL_THRESHOLD,
        )
        backtest_results[name] = bt
        print(f"  {name}: 累计收益={bt['total_return']:.2%}, "
              f"最大回撤={bt['max_drawdown']:.2%}, 胜率={bt['win_rate']:.2%}")

    # ======== 6. 结果可视化 ========
    print("\n" + "=" * 50)
    print("任务6：结果可视化")
    print("=" * 50)

    output_dir = ROOT / "outputs" / "homework1" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_prediction_vs_actual(
        y_2025.values, pred_dict,
        title=f"{selected_code} 2025年模型预测值 vs 真实值",
        save_path=str(output_dir / "prediction_vs_actual.png"),
    )

    plot_backtest_curves(
        backtest_results,
        initial_capital=INITIAL_CAPITAL,
        title=f"{selected_code} 2025年回测累计收益曲线",
        save_path=str(output_dir / "backtest_curves.png"),
    )

    print_backtest_summary(backtest_results)
    print(f"\n所有结果已保存至 {output_dir}")


if __name__ == "__main__":
    main()
