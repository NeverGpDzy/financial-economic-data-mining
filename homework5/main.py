"""Homework 5 executable pipeline: CAPM alpha persistence."""

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
from .analysis import (
    PRIMARY_TEST_PERIOD,
    ROBUST_TEST_PERIOD,
    TRAIN_PERIOD,
    capm_alpha_by_stock,
    compare_alpha_periods,
    format_pct,
)
from .data import add_stock_returns, data_audit, load_or_fetch_market, load_stock_prices
from .plots import plot_all


def _save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _json_safe(obj):
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def write_report(summary: dict, period_results: dict, output_dir: Path) -> None:
    primary = period_results[config.PRIMARY_TEST_LABEL]["summary"]
    robust = period_results[config.ROBUST_TEST_LABEL]["summary"]
    train_top = period_results[config.PRIMARY_TEST_LABEL]["train_top"].head(20)
    overlap_codes = ", ".join(primary["top20_retention_codes"]) or "无"

    lines = [
        "# 作业5：Alpha的持续性分析报告",
        "",
        f"学生：{config.STUDENT_NAME}  ",
        f"学号：{config.STUDENT_ID}",
        "",
        "## 1. 任务理解",
        "",
        "本作业研究“历史高 Alpha 股票在未来是否还能保持高 Alpha”。核心方法是在两个时间区间分别对每只上证50成分股做 CAPM 时序回归，按训练期 Alpha 排序取 Top20，再观察这些股票在未来区间的 Alpha、排名和 Top20 重合度。",
        "",
        "题目存在一个口径差异：数据说明写样本外为 2022-2023，但 AI 指令第4条写 2022-2024。本报告以 2022-2024 为主结果，并额外输出 2022-2023 稳健性结果。",
        "",
        "## 2. 数据与模型",
        "",
        f"- 股票数据：教师提供 CSV，{summary['data_audit']['stock_count']} 只上证50成分股，{summary['data_audit']['stock_date_min']} 至 {summary['data_audit']['stock_date_max']}，共 {summary['data_audit']['stock_rows']} 行。",
        f"- 市场基准：{summary['data_audit']['market_source']}，用于计算市场超额收益。",
        f"- 训练期：{config.TRAIN_START} 至 {config.TRAIN_END}。",
        f"- 主检验期：{config.PRIMARY_TEST_START} 至 {config.PRIMARY_TEST_END}。",
        f"- 稳健性检验期：{config.ROBUST_TEST_START} 至 {config.ROBUST_TEST_END}。",
        f"- 无风险利率：年化 {format_pct(config.RISK_FREE_ANNUAL)}，按 {config.TRADING_DAYS} 个交易日折算为日度 {config.RISK_FREE_DAILY:.8f}。",
        "",
        "CAPM 回归式：",
        "",
        "$$R_{i,t}-R_f=\\alpha_i+\\beta_i(R_{m,t}-R_f)+\\epsilon_{i,t}$$",
        "",
        "其中 Alpha 使用日度截距乘以 252 转换为年化 Alpha，用于排序和比较。",
        "",
        "## 3. 主结果：2022-2024",
        "",
        f"- 前期 Top20 与后期 Top20 重合数量：{primary['overlap_count']} / {primary['top_n']}。",
        f"- 重合度：{format_pct(primary['overlap_ratio'])}。",
        f"- 前期 Top20 训练期平均年化 Alpha：{format_pct(primary['train_top_alpha_mean'])}。",
        f"- 同一批股票在后期平均年化 Alpha：{format_pct(primary['train_top_future_alpha_mean'])}。",
        f"- 平均 Alpha 变化：{format_pct(primary['alpha_mean_change'])}，结论为{primary['alpha_trend']}。",
        f"- 全股票 Alpha 前后 Spearman 排名相关：{primary['spearman_corr']:.4f}。",
        f"- 仍留在后期 Top20 的股票：{overlap_codes}。",
        "",
        "训练期 Top20 股票的前后 Alpha 对比表：",
        "",
        "| 代码 | 训练期排名 | 后期排名 | 训练期Alpha | 后期Alpha | Alpha变化 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in train_top.iterrows():
        lines.append(
            f"| {row['code']} | {int(row['alpha_rank_train'])} | {int(row['alpha_rank_test'])} | "
            f"{format_pct(row['alpha_annual_train'])} | {format_pct(row['alpha_annual_test'])} | {format_pct(row['alpha_change'])} |"
        )

    lines += [
        "",
        "## 4. 稳健性结果：2022-2023",
        "",
        f"- 重合度：{robust['overlap_count']} / {robust['top_n']} = {format_pct(robust['overlap_ratio'])}。",
        f"- 前期 Top20 后期平均年化 Alpha：{format_pct(robust['train_top_future_alpha_mean'])}。",
        f"- 平均 Alpha 变化：{format_pct(robust['alpha_mean_change'])}。",
        f"- Spearman 排名相关：{robust['spearman_corr']:.4f}。",
        "",
        "## 5. 思考题回答",
        "",
        f"1. 前期高 Alpha 组合在后期 Alpha 是否上升/下降/不变？  ",
        f"   主口径下，前期 Top20 平均年化 Alpha 从 {format_pct(primary['train_top_alpha_mean'])} 变为 {format_pct(primary['train_top_future_alpha_mean'])}，整体{primary['alpha_trend']}。",
        "",
        "2. Alpha 是否具有持续性？  ",
        f"   持续性较弱。主口径 Top20 重合度只有 {format_pct(primary['overlap_ratio'])}，排名相关为 {primary['spearman_corr']:.4f}，说明历史高 Alpha 并不能稳定预测未来高 Alpha。",
        "",
        "3. 为什么会出现这种现象？如何避免？  ",
        "   金融市场存在均值回归、风格轮动、基本面变化和交易拥挤。历史高 Alpha 可能来自短期误定价或特定行情环境，一旦市场定价修正或资金追逐同类策略，Alpha 就会衰减。应对方法包括滚动窗口重估 Alpha、加入交易成本和风险约束、分行业/风格中性、避免只看单一期历史表现，并用样本外和稳健性检验确认策略。",
        "",
        "## 6. AI辅助代码关键修改点",
        "",
        "- 直接读取教师提供的本地 CSV，避免重复下载上证50成分股。",
        "- 按题目要求补充沪深300指数收益；联网失败时使用上证50等权代理并在结果中标注。",
        "- 将题目中 2022-2023 与 2022-2024 的口径差异显式处理为主结果与稳健性结果。",
        "- 对每只股票分别执行 CAPM 回归，输出 Alpha、Beta、显著性、R2、样本数和总收益。",
        "- 输出 Top20 重合度、Alpha 均值变化、排名相关和分组持续性图表。",
    ]
    (output_dir / "homework5_report.md").write_text("\n".join(lines), encoding="utf-8")
    (config.ROOT / "report" / "homework5_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_ai_note(output_dir: Path) -> None:
    text = """# 作业5 AI辅助代码说明与关键修改点

本次作业按“AI辅助生成代码，人工理解并修改参数/口径”的要求整理为可直接运行的 Python 工程。

关键修改点：

1. 数据读取优先使用 `data/homework5/raw/上证50_日度行情_2019_2025.csv`；若 raw CSV 缺失，会自动从 `data/homework5/original/上证50_日度行情_2019_2025.zip` 解压。
2. 市场基准按题目指定使用沪深300，程序会联网缓存 `data/homework5/raw/沪深300指数_2019_2024.csv`，并用 meta 文件记录来源。
3. 如果联网失败，程序不会静默报错，而是明确使用“上证50成分股等权收益代理”并写入 summary。
4. CAPM 使用日度超额收益回归，Alpha 年化口径为日度截距乘以 252。
5. 题目中“2022-2023”和“2022-2024”存在不一致，代码同时输出主结果 2022-2024 与稳健性结果 2022-2023。
6. 可通过 `python -m homework5.main --build-ppt` 在分析完成后同步重建 PPT，避免展示材料停留在旧结果。
"""
    (output_dir / "AI辅助代码说明.md").write_text(text, encoding="utf-8")


def write_audit_table(output_dir: Path) -> None:
    text = """# 作业5 AI代码审查与修复表

| 编号 | 严重性 | 审查发现 | 影响 | 修复动作 | 验证方式 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 高 | `load_stock_prices()` 只读取 raw CSV，但 raw CSV 被 `.gitignore` 忽略。 | 新环境只有原始 zip 时，主流程会 `FileNotFoundError`，不满足代码可直接运行。 | 新增 `ensure_stock_price_csv()`，raw CSV 缺失时从 `data/homework5/original/上证50_日度行情_2019_2025.zip` 安全解压。 | 临时移走 raw CSV 后调用 `ensure_stock_price_csv()`，确认自动恢复 CSV 且文件大小正确。 | 已修复 |
| 2 | 高 | `.gitignore` 中作业五 PPTX 例外被后续全局 `*.pptx` 覆盖。 | 最终 PPT 不会被 Git 收录，提交材料可能缺失。 | 在全局 `*.pptx` 规则之后补充 `!outputs/homework5/**/*.pptx` 和 `!outputs/homework5/**/*.pdf`。 | `git status --untracked-files=all outputs/homework5` 已显示 PPTX/PDF 为可跟踪文件。 | 已修复 |
| 3 | 中 | 指定 `--market-source eastmoney/baostock` 时，只要缓存存在就直接读缓存，不校验缓存来源。 | 参数语义不可靠，可能用错市场基准来源或旧缓存。 | 新增缓存 meta 来源校验；显式指定来源时，仅当缓存来源匹配才复用，否则重新获取。 | 读取 meta 并用 `python -m homework5.main --market-source baostock` 验证缓存来源为 baostock 时可复用。 | 已修复 |
| 4 | 中 | 主流程不生成 PPT，需手动运行 `report/build_homework5_ppt.py`。 | 重跑分析后 PPT 可能保留旧结果。 | 新增 `--build-ppt` 参数，主流程结束后可同步重建 PPT。 | 运行 `python -m homework5.main --build-ppt`，确认分析和 PPT 均成功生成。 | 已修复 |
"""
    (output_dir / "AI代码审查与修复表.md").write_text(text, encoding="utf-8")


def build_ppt_if_requested() -> None:
    """Build the homework PPT from the freshly generated outputs."""
    sys.path.insert(0, str(config.ROOT))
    from report.build_homework5_ppt import build

    build()


def run(refresh_market: bool = False, market_source: str = "auto") -> dict:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    (config.ROOT / "report").mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("作业5：Alpha的持续性 - AI辅助的上证50指数股数据分析")
    print("=" * 64)
    print(f"训练期: {config.TRAIN_START} ~ {config.TRAIN_END}")
    print(f"主检验期: {config.PRIMARY_TEST_START} ~ {config.PRIMARY_TEST_END}")
    print(f"稳健性检验期: {config.ROBUST_TEST_START} ~ {config.ROBUST_TEST_END}")
    print(f"无风险利率(日): {config.RISK_FREE_DAILY:.8f}")

    print("\n[1/5] 读取股票行情数据")
    stock_prices = load_stock_prices()
    stock_returns = add_stock_returns(stock_prices)
    print(f"股票数: {stock_prices['code'].nunique()}，行情行数: {len(stock_prices)}")
    print(f"日期范围: {stock_prices['date'].min().date()} ~ {stock_prices['date'].max().date()}")

    print("\n[2/5] 读取或联网缓存沪深300市场基准")
    market_returns, market_source_name = load_or_fetch_market(
        stock_returns, refresh=refresh_market, source=market_source
    )
    print(f"市场基准: {market_source_name}")
    print(f"市场日期: {market_returns['date'].min().date()} ~ {market_returns['date'].max().date()}")

    audit = data_audit(stock_prices, market_returns, market_source_name)

    print("\n[3/5] CAPM回归估计Alpha")
    train_alpha = capm_alpha_by_stock(stock_returns, market_returns, TRAIN_PERIOD)
    primary_alpha = capm_alpha_by_stock(stock_returns, market_returns, PRIMARY_TEST_PERIOD)
    robust_alpha = capm_alpha_by_stock(stock_returns, market_returns, ROBUST_TEST_PERIOD)
    print(f"训练期有效股票: {len(train_alpha)}")
    print(f"主检验期有效股票: {len(primary_alpha)}")
    print(f"稳健性检验期有效股票: {len(robust_alpha)}")

    print("\n[4/5] Alpha持续性比较")
    period_results = {
        config.PRIMARY_TEST_LABEL: compare_alpha_periods(
            train_alpha, primary_alpha, config.PRIMARY_TEST_LABEL
        ),
        config.ROBUST_TEST_LABEL: compare_alpha_periods(
            train_alpha, robust_alpha, config.ROBUST_TEST_LABEL
        ),
    }

    _save_df(train_alpha, config.OUTPUT_DIR / "alpha_train_2019_2021.csv")
    _save_df(primary_alpha, config.OUTPUT_DIR / "alpha_test_2022_2024.csv")
    _save_df(robust_alpha, config.OUTPUT_DIR / "alpha_test_2022_2023.csv")

    for label, result in period_results.items():
        suffix = label.replace("-", "_")
        _save_df(result["comparison"], config.OUTPUT_DIR / f"alpha_comparison_{suffix}.csv")
        _save_df(result["train_top"], config.OUTPUT_DIR / f"top20_train_alpha_{suffix}.csv")
        _save_df(result["future_top"], config.OUTPUT_DIR / f"top20_future_alpha_{suffix}.csv")
        _save_df(result["overlap"], config.OUTPUT_DIR / f"top20_overlap_{suffix}.csv")
        _save_df(result["group_persistence"], config.OUTPUT_DIR / f"group_persistence_{suffix}.csv")

        s = result["summary"]
        print(
            f"{label}: 重合度 {s['overlap_count']}/{s['top_n']}={format_pct(s['overlap_ratio'])}，"
            f"Top20后期Alpha均值 {format_pct(s['train_top_future_alpha_mean'])}，"
            f"Spearman {s['spearman_corr']:.4f}"
        )

    print("\n[5/5] 生成图表、报告与摘要")
    plot_paths = plot_all(period_results, config.OUTPUT_DIR)
    summary = {
        "student": {"name": config.STUDENT_NAME, "id": config.STUDENT_ID},
        "train_period": {"label": TRAIN_PERIOD.label, "start": TRAIN_PERIOD.start, "end": TRAIN_PERIOD.end},
        "primary_test_period": {
            "label": PRIMARY_TEST_PERIOD.label,
            "start": PRIMARY_TEST_PERIOD.start,
            "end": PRIMARY_TEST_PERIOD.end,
        },
        "robust_test_period": {
            "label": ROBUST_TEST_PERIOD.label,
            "start": ROBUST_TEST_PERIOD.start,
            "end": ROBUST_TEST_PERIOD.end,
        },
        "risk_free_annual": config.RISK_FREE_ANNUAL,
        "risk_free_daily": config.RISK_FREE_DAILY,
        "data_audit": audit,
        "period_results": {label: result["summary"] for label, result in period_results.items()},
        "plot_paths": {k: str(v) for k, v in plot_paths.items()},
    }
    (config.OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_safe),
        encoding="utf-8",
    )
    write_report(summary, period_results, config.OUTPUT_DIR)
    write_ai_note(config.OUTPUT_DIR)
    write_audit_table(config.OUTPUT_DIR)
    print(f"结果已保存到: {config.OUTPUT_DIR}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="作业5 Alpha持续性分析")
    parser.add_argument("--refresh-market", action="store_true", help="重新联网获取沪深300指数缓存")
    parser.add_argument(
        "--market-source",
        choices=["auto", "eastmoney", "baostock", "proxy"],
        default="auto",
        help="市场基准来源，默认auto",
    )
    parser.add_argument("--build-ppt", action="store_true", help="分析完成后同步重建作业5 PPT")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(refresh_market=args.refresh_market, market_source=args.market_source)
    if args.build_ppt:
        build_ppt_if_requested()


if __name__ == "__main__":
    main()
