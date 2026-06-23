"""Homework 9 executable pipeline."""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from . import config
from .analysis import analyze
from .data import build_close_matrix, load_all_prices
from .plots import (
    plot_drawdown_comparison,
    plot_dynamic_thresholds,
    plot_nav_comparison,
    plot_price_and_spread,
    plot_static_thresholds,
    plot_window_pvalues,
)


def _save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _json_safe(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(results: dict, output_dir: Path) -> None:
    calibration = results["initial_calibration"]
    metrics = results["metrics"].copy()
    dynamic_windows = results["dynamic_instruction_windows"].copy()
    causal_windows = results["dynamic_causal_windows"].copy()

    a_row = metrics.loc[metrics["strategy"] == "方案A：静态中枢无风控"].iloc[0]
    b1_row = metrics.loc[metrics["strategy"] == "方案B1：半年窗口ADF风控（教师指令版）"].iloc[0]
    b2_row = metrics.loc[metrics["strategy"] == "方案B2：滚动ADF风控（防未来函数）"].iloc[0]
    pass_windows = int(dynamic_windows["can_trade"].sum())
    total_windows = int(len(dynamic_windows))
    max_drift = float(dynamic_windows["abs_mu_drift_vs_static"].max())
    causal_pass_windows = int(causal_windows["can_trade"].sum())
    causal_total_windows = int(len(causal_windows))

    lines = [
        "# 作业9：协整套利模型维护与风控对比实验报告",
        "",
        f"学生：{config.STUDENT_NAME}",
        f"学号：{config.STUDENT_ID}",
        "",
        "## 1. 实验目标",
        "",
        "本作业围绕贵州茅台与泸州老窖的对数价差，比较静态中枢、半年ADF动态风控、滚动防未来函数风控和长期持有基准在2018-2024区间的表现。方案B1严格按教师指令在每个半年窗口内做ADF检验并交易；方案B2作为代码审查后的稳健性补充，用窗口开始日前最近三年数据估计参数，避免未来函数。核心观察累计收益、最大回撤、交易次数、空仓/失效时长和中枢漂移。",
        "",
        "## 2. 数据与方法",
        "",
        f"- 数据来源：Baostock日度收盘价，区间为{config.DATA_START}至{config.BACKTEST_END}。",
        f"- 标的：{config.MAOTAI}（600519.SH）与{config.LAOJIAO}（000568.SZ）。",
        "- 对数价差：`spread_t = ln(P_茅台,t) - ln(P_老窖,t)`。",
        "- ADF判定：p值 < 0.05 且ADF统计量小于5%临界值时，认为价差平稳。",
        f"- 交易阈值：`μ ± {config.THRESHOLD_MULTIPLIER}σ`；价差高于上轨做空价差，低于下轨做多价差，回归中枢平仓。",
        f"- 回测口径：日收盘信号用于下一交易日持仓；配对交易使用等权多空，总资金口径为`{config.PAIR_LEG_WEIGHT:.1f} × (茅台收益 - 老窖收益)`，不计手续费、滑点和做空限制。",
        "",
        "## 3. 期初协整价差检验",
        "",
        f"- 建模样本：{config.DATA_START}至{config.INITIAL_TRAIN_END}。",
        f"- 期初价差中枢 μ：{calibration.mu:.6f}。",
        f"- 期初价差标准差 σ：{calibration.sigma:.6f}。",
        f"- ADF统计量：{calibration.adf_stat:.4f}；p值：{calibration.p_value:.6f}；5%临界值：{calibration.critical_5pct:.4f}。",
        f"- 期初结论：{'价差平稳，协整关系成立' if calibration.is_stationary else '价差未通过5%平稳性检验，静态策略存在失效风险'}。",
        "",
        "## 4. 策略指标对比",
        "",
        "| 策略 | 累计收益 | 年化收益 | 最大回撤 | 年化波动 | Sharpe | 调仓次数 | 持仓天数 | 空仓天数 | 风控空仓天数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['strategy']} | {_pct(row['cumulative_return'])} | {_pct(row['annual_return'])} | "
            f"{_pct(row['max_drawdown'])} | {_pct(row['annual_volatility'])} | {row['sharpe']:.3f} | "
            f"{int(row['trade_count'])} | {int(row['active_days'])} | {int(row['cash_days'])} | {int(row['risk_off_days'])} |"
        )

    lines += [
        "",
        "## 5. 风控窗口与中枢漂移",
        "",
        f"- B1教师指令版共划分{total_windows}个半年交易窗口，其中{pass_windows}个窗口通过ADF检验并允许交易，{total_windows - pass_windows}个窗口触发空仓风控。",
        f"- B1窗口中枢相对期初静态中枢的最大绝对漂移为{max_drift:.6f}。",
        f"- B2防未来函数版共划分{causal_total_windows}个半年交易窗口，其中{causal_pass_windows}个窗口通过ADF检验并允许交易。",
        f"- 方案A最终累计收益为{_pct(a_row['cumulative_return'])}，最大回撤为{_pct(a_row['max_drawdown'])}；B1最终累计收益为{_pct(b1_row['cumulative_return'])}，最大回撤为{_pct(b1_row['max_drawdown'])}；B2最终累计收益为{_pct(b2_row['cumulative_return'])}，最大回撤为{_pct(b2_row['max_drawdown'])}。",
        "",
        "| 交易窗口 | 校准样本 | ADF p值 | μ | σ | 中枢漂移 | 是否交易 | 说明 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for _, row in dynamic_windows.iterrows():
        sample = f"{row['calibration_start']}~{row['calibration_end']}"
        lines.append(
            f"| {row['trade_window']} | {sample} | {row['p_value']:.6f} | {row['mu']:.6f} | "
            f"{row['sigma']:.6f} | {row['mu_drift_vs_static']:.6f} | "
            f"{'是' if row['can_trade'] else '否'} | {row['reason']} |"
        )

    lines += [
        "",
        "## 6. 结果解释",
        "",
        "- 静态中枢方案的问题在于：2015-2017得到的价差均值和波动区间被固定用于2018-2024，全程不检查协整关系是否恶化。当白酒股基本面、估值偏好或资金结构发生变化时，价差中枢会漂移，旧阈值可能把结构性偏离误判为均值回复机会。",
        "- B1严格贴合教师指令：每半年窗口先做ADF检验，检验通过才按该窗口μ和σ交易，检验失败则空仓。该口径便于课堂展示动态风控差异，但窗口内估计和窗口内交易存在事后检验色彩。",
        "- B2是代码审查后的稳健性补充：每半年开始前只使用已知历史数据估计ADF和阈值，牺牲交易频率，换取更符合实盘模型维护的因果口径。",
        "- 长期持有方案不依赖协整关系，收益主要来自单边价格趋势，适合作为价值投资基准。它可能在牛市中胜出，但回撤暴露和择时逻辑与配对套利不同。",
        "",
        "## 7. 展示材料说明",
        "",
        "- `aligned_close_prices.csv`：两只股票对齐后的收盘价。",
        "- `spread_panel.csv`：对数价格、对数价差、日收益和配对收益。",
        "- `strategy_metrics.csv`：各策略绩效汇总。",
        "- `dynamic_windows.csv`：B1教师指令版半年ADF风控窗口明细。",
        "- `dynamic_causal_windows.csv`：B2防未来函数版半年ADF风控窗口明细。",
        "- `strategy_nav_comparison.png`、`drawdown_comparison.png`、`dynamic_adf_pvalues.png`等图表用于PPT展示。",
        "",
        "## 8. AI辅助过程记录",
        "",
        "- 指令1：获取贵州茅台、泸州老窖2015-2024日度收盘价，清洗并按交易日对齐。",
        "- 指令2：对收盘价取自然对数，构造`ln(茅台)-ln(老窖)`价差，完成期初ADF平稳性检验。",
        "- 指令3：实现方案A静态中枢回测，固定2015-2017的μ和σ。",
        "- 指令4：实现方案B1半年窗口ADF风控，检验通过才使用动态阈值交易，失败则空仓。",
        "- 审查修正：补充方案B2防未来函数版，使用窗口开始日前最近三年数据估计参数。",
        "- 指令5：实现方案C长期持有，并输出累计收益、最大回撤和净值曲线。",
        "- 指令6：生成指标表、窗口诊断表、图表、Markdown报告和PPT。",
    ]

    text = "\n".join(lines)
    (output_dir / "homework9_report.md").write_text(text, encoding="utf-8")
    (config.REPORT_DIR / "homework9_report.md").write_text(text, encoding="utf-8")


def write_ai_files(output_dir: Path) -> None:
    interaction = """# 作业9 AI交互记录

1. 数据获取+预处理：编写Python代码，用Baostock获取贵州茅台600519.SH、泸州老窖000568.SZ在2015-01-01至2024-12-31的日度收盘价；对齐交易日、删除缺失值。
2. ADF平稳性检验：基于2015-01-01至2018-01-01的对数价差序列，输出ADF统计量、p值、临界值和协整关系判断。
3. 方案A：用2015-2017期初样本计算唯一μ和σ，2018-2024固定阈值交易，不做风控调整。
4. 方案B1：严格按教师指令，每半年窗口内做ADF检验；p<0.05才计算阈值并交易，否则空仓。
5. 审查修正：补充方案B2，使用窗口开始日前最近三年价差做ADF检验和阈值估计，避免未来函数。
6. 方案C：2018首个交易日买入贵州茅台或泸州老窖并长期持有到2024-12-31，输出收益和回撤。
7. 结果整合：汇总累计收益率、最大回撤、交易次数、空仓时间段、中枢漂移幅度，并生成净值、回撤、价差阈值和ADF窗口图。
"""
    audit = """# 作业9 AI代码审查与修复表

| 编号 | 审查发现 | 影响 | 修复动作 | 验证 |
| --- | --- | --- | --- | --- |
| 1 | 原始文档目录叫作业九，但Word标题写作业10/项目10。 | 输出命名可能混乱。 | 仓库按连续作业编号使用`homework9`和“作业9”，报告中保留原始项目10标题来源。 | `data/homework9/original/`保存原件，`homework9/assignment.md`保留原文。 |
| 2 | 方案B若直接用当前半年数据估计并交易，会产生未来函数。 | 回测收益会被高估，不符合实盘模型维护逻辑。 | 保留教师指令版B1作为主提交口径，同时新增B2防未来函数版作为稳健性对照。 | `dynamic_windows.csv`和`dynamic_causal_windows.csv`分别列出两版窗口。 |
| 3 | 配对交易收益若直接用`茅台收益-老窖收益`，等价于200%毛敞口。 | 收益、波动和回撤被放大，与长期持有基准不可比。 | 改为等权多空总资金口径：`0.5 × (茅台收益-老窖收益)`。 | `spread_panel.csv`同时保留原始200%毛敞口差值和总资金口径收益。 |
| 4 | 配对交易信号和收益方向容易写反。 | 做多/做空价差的净值会错误。 | 价差定义为`ln(茅台)-ln(老窖)`；上轨做空价差，下轨做多价差，收益使用等权价差收益乘以前一日持仓。 | `static_backtest.csv`、`dynamic_instruction_backtest.csv`和`dynamic_causal_backtest.csv`包含持仓、信号和策略收益列。 |
| 5 | 只输出终端结果不满足学习通提交。 | 缺少PPT、图表、过程材料。 | 输出CSV、PNG、Markdown报告、AI记录、审查表和PPT。 | `outputs/homework9/`生成完整提交材料。 |
"""
    (output_dir / "AI交互记录.md").write_text(interaction, encoding="utf-8")
    (output_dir / "AI代码审查与修复表.md").write_text(audit, encoding="utf-8")


def build_ppt_if_requested() -> None:
    sys.path.insert(0, str(config.ROOT))
    from report.build_homework9_ppt import build

    build()


def run(refresh: bool = False) -> dict:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("作业9：协整套利模型维护与风控对比实验")
    print(f"数据区间：{config.DATA_START} ~ {config.BACKTEST_END}")

    prices = load_all_prices(refresh=refresh)
    close_matrix = build_close_matrix(prices)
    results = analyze(close_matrix)

    _save_df(prices, config.OUTPUT_DIR / "asset_prices_long.csv")
    _save_df(close_matrix.reset_index(), config.OUTPUT_DIR / "aligned_close_prices.csv")
    _save_df(results["panel"], config.OUTPUT_DIR / "spread_panel.csv")
    _save_df(results["static_regime"], config.OUTPUT_DIR / "static_regime.csv")
    _save_df(results["dynamic_instruction_regime"], config.OUTPUT_DIR / "dynamic_regime.csv")
    _save_df(results["dynamic_instruction_regime"], config.OUTPUT_DIR / "dynamic_instruction_regime.csv")
    _save_df(results["dynamic_instruction_windows"], config.OUTPUT_DIR / "dynamic_windows.csv")
    _save_df(results["dynamic_instruction_windows"], config.OUTPUT_DIR / "dynamic_instruction_windows.csv")
    _save_df(results["dynamic_causal_regime"], config.OUTPUT_DIR / "dynamic_causal_regime.csv")
    _save_df(results["dynamic_causal_windows"], config.OUTPUT_DIR / "dynamic_causal_windows.csv")
    _save_df(results["static_backtest"], config.OUTPUT_DIR / "static_backtest.csv")
    _save_df(results["dynamic_instruction_backtest"], config.OUTPUT_DIR / "dynamic_backtest.csv")
    _save_df(results["dynamic_instruction_backtest"], config.OUTPUT_DIR / "dynamic_instruction_backtest.csv")
    _save_df(results["dynamic_causal_backtest"], config.OUTPUT_DIR / "dynamic_causal_backtest.csv")
    _save_df(results["hold_maotai"], config.OUTPUT_DIR / "hold_maotai_backtest.csv")
    _save_df(results["hold_laojiao"], config.OUTPUT_DIR / "hold_laojiao_backtest.csv")
    _save_df(results["metrics"], config.OUTPUT_DIR / "strategy_metrics.csv")

    plots = {
        "price_and_spread": str(plot_price_and_spread(results["panel"], config.OUTPUT_DIR / "price_and_spread.png")),
        "static_thresholds": str(plot_static_thresholds(results["static_backtest"], config.OUTPUT_DIR / "static_thresholds.png")),
        "dynamic_thresholds": str(plot_dynamic_thresholds(results["dynamic_instruction_backtest"], config.OUTPUT_DIR / "dynamic_thresholds.png")),
        "dynamic_causal_thresholds": str(plot_dynamic_thresholds(results["dynamic_causal_backtest"], config.OUTPUT_DIR / "dynamic_causal_thresholds.png")),
        "nav_comparison": str(
            plot_nav_comparison(
                {
                    "静态无风控": results["static_backtest"],
                    "B1教师指令风控": results["dynamic_instruction_backtest"],
                    "B2防未来函数": results["dynamic_causal_backtest"],
                    "茅台持有": results["hold_maotai"],
                    "老窖持有": results["hold_laojiao"],
                },
                config.OUTPUT_DIR / "strategy_nav_comparison.png",
            )
        ),
        "drawdown_comparison": str(
            plot_drawdown_comparison(
                {
                    "静态无风控": results["static_backtest"],
                    "B1教师指令风控": results["dynamic_instruction_backtest"],
                    "B2防未来函数": results["dynamic_causal_backtest"],
                    "茅台持有": results["hold_maotai"],
                    "老窖持有": results["hold_laojiao"],
                },
                config.OUTPUT_DIR / "drawdown_comparison.png",
            )
        ),
        "dynamic_adf_pvalues": str(plot_window_pvalues(results["dynamic_instruction_windows"], config.OUTPUT_DIR / "dynamic_adf_pvalues.png")),
        "dynamic_causal_adf_pvalues": str(plot_window_pvalues(results["dynamic_causal_windows"], config.OUTPUT_DIR / "dynamic_causal_adf_pvalues.png")),
    }

    write_report(results, config.OUTPUT_DIR)
    write_ai_files(config.OUTPUT_DIR)

    metrics_records = results["metrics"].to_dict(orient="records")
    summary = {
        "student": {"name": config.STUDENT_NAME, "id": config.STUDENT_ID},
        "period": {
            "data_start": config.DATA_START,
            "initial_train_end": config.INITIAL_TRAIN_END,
            "backtest_end": config.BACKTEST_END,
        },
        "initial_calibration": asdict(results["initial_calibration"]),
        "metrics": metrics_records,
        "dynamic_window_count": int(len(results["dynamic_instruction_windows"])),
        "dynamic_pass_window_count": int(results["dynamic_instruction_windows"]["can_trade"].sum()),
        "dynamic_causal_window_count": int(len(results["dynamic_causal_windows"])),
        "dynamic_causal_pass_window_count": int(results["dynamic_causal_windows"]["can_trade"].sum()),
        "plots": plots,
    }
    (config.OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_safe),
        encoding="utf-8",
    )
    print(f"结果已保存到：{config.OUTPUT_DIR}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="作业9 协整套利模型维护与风控对比实验")
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
