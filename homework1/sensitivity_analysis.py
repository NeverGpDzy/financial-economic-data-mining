"""作业1 敏感性分析脚本。

支持两种敏感性分析：
1. 改变滑动窗口大小（默认10天 vs 20天）
2. 改变统计周期（训练-回测年份）
"""

import sys
from pathlib import Path

# 将项目根目录加入路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data_fetcher import fetch_all_stocks
from homework1.features import build_features
from homework1.models import train_and_evaluate, get_models
from homework1.backtest import run_backtest
from homework1.plots import plot_prediction_vs_actual, plot_backtest_curves, print_backtest_summary

import numpy as np
import pandas as pd
import json


def run_single_experiment(
    stock_codes: list[str],
    start_date: str,
    end_date: str,
    window: int,
    initial_capital: float = 1_000_000,
    commission: float = 0.0003,
    buy_threshold: float = 0.005,
    sell_threshold: float = 0.005,
    data_dir: str = None,
) -> dict:
    """运行单次实验。

    Args:
        stock_codes: 股票代码列表
        start_date: 数据起始日期
        end_date: 数据结束日期
        window: 滑动窗口大小
        initial_capital: 初始资金
        commission: 手续费率
        buy_threshold: 买入阈值
        sell_threshold: 卖出阈值
        data_dir: 数据缓存目录

    Returns:
        实验结果字典
    """
    print(f"\n{'='*60}")
    print(f"实验配置: 窗口={window}, 周期={start_date}~{end_date}")
    print(f"{'='*60}")

    # 1. 数据获取
    print("\n[1/6] 数据获取...")
    stock_data = fetch_all_stocks(stock_codes, start_date, end_date, save_dir=data_dir)

    selected_code = stock_codes[0]
    df = stock_data[selected_code]
    print(f"选定股票: {selected_code}, 数据量: {len(df)} 条")

    # 2. 特征构建
    print("\n[2/6] 特征构建...")
    df_feat, feature_cols = build_features(df, window=window)
    print(f"特征数: {len(feature_cols)}, 样本数: {len(df_feat)}")

    # 3. 数据划分 - 训练集和回测集各占一半
    print("\n[3/6] 数据划分...")
    total_len = len(df_feat)
    split_idx = total_len // 2

    df_train = df_feat.iloc[:split_idx]
    df_test = df_feat.iloc[split_idx:]

    X_train = df_train[feature_cols]
    y_train = df_train["label"]
    X_test = df_test[feature_cols]
    y_test = df_test["label"]

    train_start = df_train.index[0].strftime("%Y-%m-%d")
    train_end = df_train.index[-1].strftime("%Y-%m-%d")
    test_start = df_test.index[0].strftime("%Y-%m-%d")
    test_end = df_test.index[-1].strftime("%Y-%m-%d")

    print(f"训练集: {len(X_train)} 条 ({train_start} ~ {train_end})")
    print(f"回测集: {len(X_test)} 条 ({test_start} ~ {test_end})")

    # 4. 模型训练与评估
    print("\n[4/6] 模型训练与评估...")
    results_train = train_and_evaluate(X_train, y_train, X_test, y_test)

    # 训练最终模型
    final_models = get_models()
    for name, model in final_models.items():
        model.fit(X_train, y_train)

    # 5. 量化回测
    print("\n[5/6] 量化回测...")
    backtest_results = {}
    pred_dict = {}

    for name, model in final_models.items():
        y_pred = model.predict(X_test)
        pred_dict[name] = y_pred

        bt = run_backtest(
            df_test, y_pred,
            initial_capital=initial_capital,
            commission=commission,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
        backtest_results[name] = bt
        print(f"  {name}: 累计收益={bt['total_return']:.2%}, "
              f"最大回撤={bt['max_drawdown']:.2%}, 胜率={bt['win_rate']:.2%}")

    # 6. 汇总结果
    print("\n[6/6] 汇总结果...")
    summary = {
        "config": {
            "window": window,
            "start_date": start_date,
            "end_date": end_date,
            "train_period": f"{train_start}~{train_end}",
            "test_period": f"{test_start}~{test_end}",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        },
        "backtest_results": {}
    }

    for name, bt in backtest_results.items():
        summary["backtest_results"][name] = {
            "total_return": bt["total_return"],
            "max_drawdown": bt["max_drawdown"],
            "win_rate": bt["win_rate"],
            "trades": bt["trades"],
        }

    return summary, backtest_results, pred_dict, y_test, df_test


def save_experiment_results(
    summary: dict,
    backtest_results: dict,
    pred_dict: dict,
    y_test: pd.Series,
    df_test: pd.DataFrame,
    output_dir: Path,
    initial_capital: float = 1_000_000,
):
    """保存实验结果。

    Args:
        summary: 实验摘要
        backtest_results: 回测结果
        pred_dict: 预测结果
        y_test: 真实标签
        df_test: 测试数据
        output_dir: 输出目录
        initial_capital: 初始资金
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存JSON摘要
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {output_dir / 'summary.json'}")

    # 保存预测值vs真实值图
    plot_prediction_vs_actual(
        y_test.values, pred_dict,
        title=f"窗口={summary['config']['window']} 周期={summary['config']['start_date']}~{summary['config']['end_date']}\n模型预测值 vs 真实值",
        save_path=str(output_dir / "prediction_vs_actual.png"),
    )

    # 保存回测曲线图
    plot_backtest_curves(
        backtest_results,
        initial_capital=initial_capital,
        title=f"窗口={summary['config']['window']} 周期={summary['config']['start_date']}~{summary['config']['end_date']}\n回测累计收益曲线",
        save_path=str(output_dir / "backtest_curves.png"),
    )

    # 保存回测汇总表
    print_backtest_summary(backtest_results)


def run_window_sensitivity(stock_codes: list[str], data_dir: str, output_base: Path):
    """运行滑动窗口敏感性分析。

    Args:
        stock_codes: 股票代码列表
        data_dir: 数据缓存目录
        output_base: 输出基础目录
    """
    print("\n" + "=" * 70)
    print("敏感性分析1: 滑动窗口大小 (10天 vs 20天)")
    print("=" * 70)

    # 基准实验: 窗口=10
    summary_10, bt_10, pred_10, y_test_10, df_test_10 = run_single_experiment(
        stock_codes=stock_codes,
        start_date="2024-01-01",
        end_date="2025-12-31",
        window=10,
        data_dir=data_dir,
    )
    save_experiment_results(
        summary_10, bt_10, pred_10, y_test_10, df_test_10,
        output_dir=output_base / "window_10",
    )

    # 对比实验: 窗口=20
    summary_20, bt_20, pred_20, y_test_20, df_test_20 = run_single_experiment(
        stock_codes=stock_codes,
        start_date="2024-01-01",
        end_date="2025-12-31",
        window=20,
        data_dir=data_dir,
    )
    save_experiment_results(
        summary_20, bt_20, pred_20, y_test_20, df_test_20,
        output_dir=output_base / "window_20",
    )

    # 生成对比图
    _plot_comparison(
        results_list=[summary_10, summary_20],
        labels=["窗口=10", "窗口=20"],
        title="滑动窗口敏感性分析对比",
        save_path=output_base / "window_comparison.png",
    )

    return [summary_10, summary_20]


def run_period_sensitivity(stock_codes: list[str], data_dir: str, output_base: Path):
    """运行统计周期敏感性分析。

    Args:
        stock_codes: 股票代码列表
        data_dir: 数据缓存目录
        output_base: 输出基础目录
    """
    print("\n" + "=" * 70)
    print("敏感性分析2: 统计周期 (2023-2024 vs 2022-2023)")
    print("=" * 70)

    # 基准实验: 2023训练-2024回测
    summary_23_24, bt_23_24, pred_23_24, y_test_23_24, df_test_23_24 = run_single_experiment(
        stock_codes=stock_codes,
        start_date="2023-01-01",
        end_date="2024-12-31",
        window=10,
        data_dir=data_dir,
    )
    save_experiment_results(
        summary_23_24, bt_23_24, pred_23_24, y_test_23_24, df_test_23_24,
        output_dir=output_base / "period_2023_2024",
    )

    # 对比实验: 2022训练-2023回测
    summary_22_23, bt_22_23, pred_22_23, y_test_22_23, df_test_22_23 = run_single_experiment(
        stock_codes=stock_codes,
        start_date="2022-01-01",
        end_date="2023-12-31",
        window=10,
        data_dir=data_dir,
    )
    save_experiment_results(
        summary_22_23, bt_22_23, pred_22_23, y_test_22_23, df_test_22_23,
        output_dir=output_base / "period_2022_2023",
    )

    # 生成对比图
    _plot_comparison(
        results_list=[summary_23_24, summary_22_23],
        labels=["2023训练-2024回测", "2022训练-2023回测"],
        title="统计周期敏感性分析对比",
        save_path=output_base / "period_comparison.png",
    )

    return [summary_23_24, summary_22_23]


def _plot_comparison(
    results_list: list[dict],
    labels: list[str],
    title: str,
    save_path: Path,
):
    """生成对比图。

    Args:
        results_list: 实验结果列表
        labels: 标签列表
        title: 图表标题
        save_path: 保存路径
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    models = ["LinearRegression", "RandomForest", "LightGBM"]
    metrics = ["total_return", "max_drawdown", "win_rate", "trades"]
    metric_names = ["累计收益", "最大回撤", "胜率", "交易次数"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    x = np.arange(len(models))
    width = 0.35

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        values_list = []
        for summary in results_list:
            values = [summary["backtest_results"][m][metric] for m in models]
            values_list.append(values)

        for i, (values, label) in enumerate(zip(values_list, labels)):
            bars = ax.bar(x + i * width, values, width, label=label)
            # 添加数值标签
            for bar, val in zip(bars, values):
                if metric in ["total_return", "max_drawdown", "win_rate"]:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                           f'{val:.2%}', ha='center', va='bottom', fontsize=8)
                else:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                           f'{val}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel("模型")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  已保存对比图: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    STOCK_CODES = [
        "sh.600519",  # 贵州茅台
    ]

    output_base = ROOT / "outputs" / "homework1" / "sensitivity"
    data_dir = str(ROOT / "data" / "homework1")

    # 运行滑动窗口敏感性分析
    window_results = run_window_sensitivity(STOCK_CODES, data_dir, output_base)

    # 运行统计周期敏感性分析
    period_results = run_period_sensitivity(STOCK_CODES, data_dir, output_base)

    print("\n" + "=" * 70)
    print("所有敏感性分析完成！")
    print("=" * 70)
    print(f"结果保存在: {output_base}")
