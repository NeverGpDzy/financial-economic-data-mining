"""Homework 8 executable pipeline."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from . import config
from .analysis import build_best_pair_detail, scan_all_pairs
from .data import build_close_matrix, load_all_assets
from .plots import plot_best_pair_prices, plot_pair_pvalues, plot_spread, plot_zscore


def _save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _json_safe(obj):
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def write_report(pairs: pd.DataFrame, best: dict, pair_df: pd.DataFrame, output_dir: Path) -> None:
    significant = int(pairs["is_cointegrated"].sum())
    total = int(len(pairs))
    top_rows = pairs.head(10)

    lines = [
        "# 作业8：统计套利之协整检验与配对交易模型报告",
        "",
        f"学生：{config.STUDENT_NAME}  ",
        f"学号：{config.STUDENT_ID}",
        "",
        "## 1. 实验目标",
        "",
        "本作业基于EG两步法遍历标的池内所有配对，寻找残差价差最显著平稳的股票对，并围绕最优配对完成线性回归、价差平稳性检验、z-score择时信号标注，形成统计套利配对交易模型。",
        "",
        "## 2. 数据与方法",
        "",
        f"- 数据来源：Baostock日度收盘价，区间为{config.START_DATE}至{config.END_DATE}。",
        "- 标的池：作业基础信息列出的中国石油、贵州茅台、泸州老窖、兴蓉环境、招商银行、工商银行、长江电力、上证50指数。",
        "- EG两步法：先用OLS拟合 `Y = alpha + beta * X + epsilon`，再对残差 `epsilon` 做ADF单位根检验。",
        "- 协整判定：残差ADF p值 < 0.05 且ADF统计量 < 5%临界值，认为该配对协整显著。",
        "- 交易信号：价差z-score > 2时做空Y、做多X；z-score < -2时做多Y、做空X；z-score回归0附近时平仓观察。",
        "",
        "## 3. 全部配对协整检验",
        "",
        f"- 共检验{total}组配对，其中{significant}组通过5%显著性协整检验。",
        f"- 最优配对：{best['y_asset']} 与 {best['x_asset']}，残差ADF p值为{best['p_value']:.6f}。",
        "",
        "| 排名 | Y资产 | X资产 | alpha | beta | R² | ADF统计量 | p值 | 5%临界值 | 结论 |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in top_rows.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['y_asset']} | {row['x_asset']} | "
            f"{row['alpha']:.4f} | {row['beta']:.4f} | {row['r_squared']:.4f} | "
            f"{row['adf_stat']:.4f} | {row['p_value']:.6f} | {row['critical_5pct']:.4f} | {row['conclusion']} |"
        )

    lines += [
        "",
        "## 4. 最优配对建模结果",
        "",
        f"- 回归方程：`Y = {best['ols_alpha']:.4f} + {best['ols_beta']:.4f} * X + epsilon`。",
        f"- 回归R²：{best['ols_r_squared']:.4f}。",
        f"- 残差ADF统计量：{best['adf_stat']:.4f}，p值：{best['p_value']:.6f}，5%临界值：{best['critical_5pct']:.4f}。",
        f"- z-score > 2信号次数：{best['trade_signal_count_short_y']}；z-score < -2信号次数：{best['trade_signal_count_long_y']}；接近0平仓观察次数：{best['exit_zone_count']}。",
        "",
        "## 5. 深度问答题",
        "",
        "1. 为什么会出现配对现象？  ",
        "   配对现象通常来自共同基本面、同一行业或指数、相似资金定价逻辑、共同风险因子暴露。两只资产价格水平可以各自非平稳，但若它们长期受同一经济变量驱动，线性组合后的价差就可能围绕均值波动。",
        "",
        "2. 这个现象在未来一定会重现吗？  ",
        "   不一定。协整关系是历史样本中的统计关系，不是无条件的经济定律。未来基本面、监管、流动性、投资者结构和公司经营状态变化，都可能使历史价差均值回复规律失效。",
        "",
        "3. 哪些因素可能导致配对失效？  ",
        "   行业逻辑变化、公司重大事件、财务结构分化、指数成分调整、交易拥挤、流动性冲击、极端市场环境、样本过短或过拟合，都会导致配对关系断裂。",
        "",
        "4. 你打算如何维护模型？  ",
        "   应定期滚动重估协整关系和对冲系数，设置止损与最大持仓期限，监控残差ADF检验、z-score分布、交易成本和滑点；当p值恶化、价差均值漂移或基本面逻辑破坏时，暂停或更换配对。",
        "",
        "## 6. AI辅助过程记录",
        "",
        "- 第1步：导入Baostock、pandas、numpy、matplotlib、statsmodels并设置中文绘图。",
        "- 第2步：定义标的池与2015-01-01至2018-01-01时间范围，批量获取并对齐收盘价。",
        "- 第3步：基于EG两步法编写协整检验函数。",
        "- 第4步：遍历所有两两配对，按残差ADF p值排序。",
        "- 第5步：绘制最优配对标准化价格走势。",
        "- 第6步：输出最优配对OLS系数、截距与R²，计算残差价差。",
        "- 第7步：对价差做ADF检验并绘制价差图。",
        "- 第8步：计算z-score并标注交易信号。",
        "- 第9步：整合调试为一键运行脚本，输出表格、图表、报告和PPT。",
    ]

    text = "\n".join(lines)
    (output_dir / "homework8_report.md").write_text(text, encoding="utf-8")
    (config.REPORT_DIR / "homework8_report.md").write_text(text, encoding="utf-8")


def write_ai_files(output_dir: Path) -> None:
    interaction = """# 作业8 AI交互记录

1. 环境准备：导入协整检验、配对交易所需库：baostock、pandas、numpy、matplotlib、statsmodels，并设置中文绘图正常显示。
2. 数据参数：定义实验标的池和2015-01-01至2018-01-01时间范围，编写函数批量获取收盘价并做时间对齐、删除缺失值。
3. 协整检验函数：基于EG两步法，先做线性回归，再对残差做ADF检验，输出p值和协整结论。
4. 遍历所有配对：批量计算所有两两组合的协整关系，生成按显著性排序的汇总表，筛选最优配对。
5. 价格对比图：选择最优配对，绘制标准化价格时序对比图。
6. 线性拟合：对最优配对做OLS回归，输出回归系数、截距、R²，并计算残差价差。
7. 价差检验：对残差做ADF检验并绘制价差时序图。
8. z-score信号：计算价差z-score，按±2阈值标注开仓信号，回归0附近作为平仓观察。
9. 整合调试：将全部功能整合为一键运行代码，输出表格、图表、检验值、报告和PPT。
"""
    audit = """# 作业8 AI代码审查与修复表

| 编号 | 审查发现 | 影响 | 修复动作 | 验证 |
| --- | --- | --- | --- | --- |
| 1 | 作业基础信息与分步指令标的池不完全一致。 | 少做标的会降低遍历完整性。 | 保留基础信息中列出的8个标的，覆盖分步指令中的6个标的。 | `cointegration_pairs.csv`包含8选2共28组配对。 |
| 2 | 协整检验不能直接检验两只价格相关性。 | 相关但不协整会造成伪套利。 | 使用EG两步法：OLS回归后对残差做ADF检验。 | 输出残差ADF统计量、p值和5%临界值。 |
| 3 | 所有资产交易日期必须对齐。 | 日期错位会污染回归残差。 | 先透视为收盘价矩阵，再删除任一资产缺失的日期。 | `aligned_close_prices.csv`为统一日期索引。 |
| 4 | z-score信号需要明确多空方向。 | PPT和报告中可能无法解释交易动作。 | 按Y-alpha-beta*X残差定义，z>2做空Y做多X，z<-2做多Y做空X。 | `best_pair_spread_zscore.csv`包含信号列。 |
"""
    (output_dir / "AI交互记录.md").write_text(interaction, encoding="utf-8")
    (output_dir / "AI代码审查与修复表.md").write_text(audit, encoding="utf-8")


def build_ppt_if_requested() -> None:
    sys.path.insert(0, str(config.ROOT))
    from report.build_homework8_ppt import build

    build()


def run(refresh: bool = False) -> dict:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("作业8：协整检验与配对交易模型")
    print(f"数据区间：{config.START_DATE} ~ {config.END_DATE}")

    prices = load_all_assets(refresh=refresh)
    close_matrix = build_close_matrix(prices)
    pairs = scan_all_pairs(close_matrix)
    best_detail, pair_df = build_best_pair_detail(close_matrix, pairs)

    _save_df(prices, config.OUTPUT_DIR / "asset_prices_long.csv")
    _save_df(close_matrix.reset_index(), config.OUTPUT_DIR / "aligned_close_prices.csv")
    _save_df(pairs, config.OUTPUT_DIR / "cointegration_pairs.csv")
    _save_df(pair_df, config.OUTPUT_DIR / "best_pair_spread_zscore.csv")

    plots = {
        "pair_pvalues": str(plot_pair_pvalues(pairs, config.OUTPUT_DIR / "cointegration_pvalues_top12.png")),
        "best_pair_prices": str(plot_best_pair_prices(best_detail, pair_df, config.OUTPUT_DIR / "best_pair_standardized_prices.png")),
        "spread": str(plot_spread(best_detail, pair_df, config.OUTPUT_DIR / "best_pair_spread.png")),
        "zscore": str(plot_zscore(best_detail, pair_df, config.OUTPUT_DIR / "best_pair_zscore_signals.png")),
    }

    write_report(pairs, best_detail, pair_df, config.OUTPUT_DIR)
    write_ai_files(config.OUTPUT_DIR)

    summary = {
        "student": {"name": config.STUDENT_NAME, "id": config.STUDENT_ID},
        "period": {"start": config.START_DATE, "end": config.END_DATE},
        "asset_count": int(close_matrix.shape[1]),
        "aligned_rows": int(close_matrix.shape[0]),
        "pair_count": int(len(pairs)),
        "significant_pair_count": int(pairs["is_cointegrated"].sum()),
        "best_pair": best_detail,
        "plots": plots,
    }
    (config.OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_safe),
        encoding="utf-8",
    )
    print(f"结果已保存到：{config.OUTPUT_DIR}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="作业8 协整检验与配对交易模型")
    parser.add_argument("--refresh", action="store_true", help="忽略本地缓存，重新从Baostock获取行情")
    parser.add_argument("--build-ppt", action="store_true", help="分析完成后同步生成PPT")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(refresh=args.refresh)
    if args.build_ppt:
        build_ppt_if_requested()


if __name__ == "__main__":
    main()

