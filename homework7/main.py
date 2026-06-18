"""Homework 7 executable pipeline."""

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
from .analysis import analyze_stationarity
from .data import load_all_prices
from .plots import plot_stationarity_summary, plot_stock_series


def _save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _json_safe(obj):
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def write_report(summary_df: pd.DataFrame, output_dir: Path) -> None:
    price_stationary = int((summary_df["price_stationarity"] == "平稳").sum())
    return_stationary = int((summary_df["return_stationarity"] == "平稳").sum())

    lines = [
        "# 作业7：统计套利之平稳性检验报告",
        "",
        f"学生：{config.STUDENT_NAME}  ",
        f"学号：{config.STUDENT_ID}",
        "",
        "## 1. 实验目标",
        "",
        "本作业对中国石油、贵州茅台、兴蓉环境、招商银行、工商银行五只股票在2024年的日度收盘价和对数收益率分别进行ADF单位根检验，验证金融时间序列中“价格通常非平稳、收益率更接近平稳”的经验规律，并为后续协整套利和配对交易建模提供前提检验。",
        "",
        "## 2. 方法说明",
        "",
        "- 数据来源：Baostock日度行情，时间范围为2024-01-01至2024-12-31。",
        "- 对数收益率：`r_t = ln(P_t) - ln(P_{t-1})`。",
        "- ADF原假设：序列存在单位根，即非平稳。",
        "- 判定规则：p值 < 0.05 且ADF统计量 < 5%临界值时，判为平稳。",
        "",
        "## 3. ADF检验结果汇总",
        "",
        "| 股票 | 代码 | 价格样本 | 收盘价均值 | 收盘价方差 | 收盘价ADF | 收盘价p值 | 价格结论 | 收益率均值 | 收益率方差 | 收益率ADF | 收益率p值 | 收益率结论 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['stock_name']} | {row['code']} | {int(row['price_obs'])} | "
            f"{row['price_mean']:.4f} | {row['price_variance']:.4f} | "
            f"{row['price_adf_stat']:.4f} | {row['price_p_value']:.4f} | {row['price_stationarity']} | "
            f"{row['return_mean']:.6f} | {row['return_variance']:.8f} | "
            f"{row['return_adf_stat']:.4f} | {row['return_p_value']:.4f} | {row['return_stationarity']} |"
        )

    lines += [
        "",
        "## 4. 结果规律",
        "",
        f"- 五只股票中，收盘价序列判为平稳的数量为{price_stationary}只，收益率序列判为平稳的数量为{return_stationary}只。",
        "- 从检验逻辑看，收盘价包含趋势、估值重定价和随机游走成分，均值与方差容易随时间改变；对数收益率消除了价格水平趋势，更接近围绕零均值波动，因此更容易通过ADF平稳性检验。",
        "- 若个别股票在单一年份内收盘价也通过ADF检验，应理解为短样本区间内价格呈现均值回复或区间震荡，不代表长期价格序列一定平稳。",
        "",
        "## 5. 理论思考题",
        "",
        "1. 为什么股票收盘价序列通常非平稳，而收益率序列通常平稳？  ",
        "   收盘价是价格水平变量，会持续吸收基本面、风险偏好、流动性和市场情绪信息，常表现为随机游走或带趋势过程；收益率是一阶差分后的相对变化，去除了价格水平趋势，波动通常围绕零均值展开，因此更接近平稳。",
        "",
        "2. 统计套利模型为什么一般用收益率而不是直接用价格建模？  ",
        "   非平稳价格直接回归容易产生伪回归，使相关性和显著性被夸大；收益率更接近平稳，均值、方差和协方差相对稳定，更适合做风险估计、信号检验和策略回测。若使用价格，则通常必须先验证协整关系。",
        "",
        "3. 什么样的资产更适合做统计套利？  ",
        "   更适合统计套利的资产应具有稳定、可解释的长期关系，收益率或价差序列平稳，流动性充足，交易成本较低，并且关系不容易被基本面突变破坏。高度同质、同产业链、同指数成分或同类金融工具更容易形成可交易的均值回复关系。",
        "",
        "## 6. AI辅助过程记录",
        "",
        "- 指令1：生成导入Baostock、pandas、numpy、matplotlib、statsmodels.adfuller的环境准备代码。",
        "- 指令2：定义五只股票代码映射与2024年统一检验周期。",
        "- 指令3：编写数据获取、对数收益率计算、统计量计算、ADF检验和时序图绘制函数。",
        "- 指令4：批量运行五只股票的价格与收益率平稳性检验。",
        "- 指令5：生成汇总对比表，展示收盘价与收益率平稳性差异。",
        "- 指令6：整合调试为一键可运行脚本，并输出结果表、图表、报告和PPT。",
    ]

    text = "\n".join(lines)
    (output_dir / "homework7_report.md").write_text(text, encoding="utf-8")
    (config.REPORT_DIR / "homework7_report.md").write_text(text, encoding="utf-8")


def write_ai_files(output_dir: Path) -> None:
    interaction = """# 作业7 AI交互记录

1. 环境准备：请生成Python代码，导入baostock、pandas、numpy、matplotlib、statsmodels.adfuller库，用于股票数据获取、收益率计算和时间序列平稳性检验。
2. 参数定义：请定义5只股票名称与代码映射：中国石油601857.SH、贵州茅台600519.SH、兴蓉环境000598.SZ、招商银行600036.SH、工商银行601398.SH；设置统一时间周期2024-01-01至2024-12-31。
3. 核心函数：请编写平稳性检验函数，实现Baostock获取日度收盘价、计算对数收益率、计算均值方差、分别执行ADF检验并绘制收盘价和收益率时序图。
4. 批量检验：调用上述函数，批量对5只股票做收盘价和收益率平稳性检验，输出每只股票的统计量、ADF结果和平稳性结论。
5. 对比表格：生成5只股票收盘价平稳性与收益率平稳性汇总表，清晰展示差异。
6. 代码调试：检查并修复代码错误，生成可直接运行的完整代码，运行后输出所有结果、图表和对比表。
"""
    audit = """# 作业7 AI代码审查与修复表

| 编号 | 审查发现 | 影响 | 修复动作 | 验证 |
| --- | --- | --- | --- | --- |
| 1 | 股票代码需要转换为Baostock格式，如601857.SH对应sh.601857。 | 代码格式错误会导致无法获取数据。 | 在配置中直接使用Baostock格式，并在输出中转换为展示格式。 | 主程序成功获取5只股票数据。 |
| 2 | ADF判定不能只看p值，还需比较ADF统计量与5%临界值。 | 平稳性结论可能不严谨。 | `adf_test()`同时检查p值和5%临界值。 | `stationarity_summary.csv`含p值、统计量、临界值和结论。 |
| 3 | 收益率首日为空值。 | 直接检验会引入缺失值。 | 计算后在ADF检验前清理NaN和无穷值。 | 每只股票收益率样本数为价格样本数减1。 |
| 4 | 结果需要可提交材料而不只是终端输出。 | 学习通提交缺少报告和展示材料。 | 输出CSV、PNG、Markdown报告、AI记录、代码审查表和PPT。 | `outputs/homework7/`生成完整文件。 |
"""
    (output_dir / "AI交互记录.md").write_text(interaction, encoding="utf-8")
    (output_dir / "AI代码审查与修复表.md").write_text(audit, encoding="utf-8")


def build_ppt_if_requested() -> None:
    sys.path.insert(0, str(config.ROOT))
    from report.build_homework7_ppt import build

    build()


def run(refresh: bool = False) -> dict:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("作业7：统计套利之平稳性检验")
    print(f"数据区间：{config.START_DATE} ~ {config.END_DATE}")

    prices = load_all_prices(refresh=refresh)
    panel, summary_df = analyze_stationarity(prices)

    _save_df(panel, config.OUTPUT_DIR / "stationarity_panel.csv")
    _save_df(summary_df, config.OUTPUT_DIR / "stationarity_summary.csv")
    plot_paths = plot_stock_series(panel, config.OUTPUT_DIR)
    pvalue_plot = plot_stationarity_summary(summary_df, config.OUTPUT_DIR / "adf_pvalue_comparison.png")
    write_report(summary_df, config.OUTPUT_DIR)
    write_ai_files(config.OUTPUT_DIR)

    summary = {
        "student": {"name": config.STUDENT_NAME, "id": config.STUDENT_ID},
        "period": {"start": config.START_DATE, "end": config.END_DATE},
        "stock_count": int(summary_df.shape[0]),
        "price_stationary_count": int((summary_df["price_stationarity"] == "平稳").sum()),
        "return_stationary_count": int((summary_df["return_stationarity"] == "平稳").sum()),
        "plot_paths": {k: str(v) for k, v in plot_paths.items()},
        "pvalue_plot": str(pvalue_plot),
    }
    (config.OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_safe),
        encoding="utf-8",
    )
    print(f"结果已保存到：{config.OUTPUT_DIR}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="作业7 平稳性检验")
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

