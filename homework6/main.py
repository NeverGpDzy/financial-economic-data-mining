"""Homework 6 executable pipeline: FCFF value investing dual-strategy analysis."""

from __future__ import annotations

import argparse
import io
import json
import sys

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from . import config
from .backtest import (
    build_strategy_groups,
    fcff_group_backtest,
    price_group_backtest,
    save_backtest_outputs,
)
from .data import (
    build_annual_panel,
    data_audit,
    load_price_data,
    load_raw_tables,
    save_data_description,
    save_panel,
)
from .factors import (
    add_traditional_scores,
    compute_ic_ir,
    compute_vif,
    preprocess_factors,
    select_model_features,
    single_factor_ols,
)
from .models import run_modeling
from .plots import generate_all_plots


def _save_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(config.OUTPUT_DIR / name, index=False, encoding="utf-8-sig")


def _pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.2%}"


def _json_safe(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return str(obj)


def write_ai_records() -> None:
    interaction = """# 作业6 AI交互记录

本记录按老师要求整理为五个任务的AI指令和人工修改要点，便于提交时截图或转写。

## 任务1：数据加载、因子计算与预处理

AI指令：读取上证50成分股年度/季度财务数据、分红数据、复权日行情和沪深300指数，按年度截面计算F1-F11价值财务因子、FCFF及未来1年/3年FCFF标签；反向因子做正向化，按年度截面3σ缩尾、缺失值中位数填充并Z标准化；输出训练集和测试集。

人工修改：发现教师文件实际覆盖2014-2024，因此将可复现口径改为2014-2017训练、2018-2024样本外；模型训练标签只使用2014-2016，避免2018标签泄露；FCFF增速受接近0分母影响较大，对标签也按年度截面做3σ缩尾，保留原始标签字段备查。

## 任务2：IC/IR与VIF因子质检

AI指令：基于年度因子面板逐年计算Spearman IC，汇总IC均值、IC标准差和IR；全因子做VIF共线性检验，VIF>10迭代剔除，输出因子质检表。

人工修改：保留IC有效性和VIF结果两套字段，LGBM使用VIF保留因子，p值待定因子允许进入非线性模型。

## 任务3：单因子年度截面OLS

AI指令：对每个标准化因子逐年度做截面OLS，因变量为未来1年FCFF增速，输出β均值、p值均值、t统计量均值和线性有效/待定判定。

人工修改：对小截面和常数因子做跳过处理，防止回归奇异。

## 任务4：LGBM机器学习非线性赋权

AI指令：使用过VIF质检的标准化因子训练LightGBM回归模型，标签为未来1年FCFF增速；限制树深不超过3，使用时间序列交叉验证，输出特征重要度和综合价值得分。

人工修改：因实际训练年度较少，交叉验证自动退化为可用的扩展窗口年度验证；模型参数加入min_data_in_leaf和L2正则控制过拟合。

## 任务5：双策略FCFF与股价收益回测

AI指令：构建方案A固定阈值传统价值规则和方案B LGBM价值得分，按年度截面分为A/B/C三组；统计未来3年FCFF年化增速分层结果，并用同一分组做年调仓等权股价回测，初始资金100万、单边手续费0.1%、基准沪深300。

人工修改：价格回测使用上一年得分持有下一年，严格年度调仓；样本外股价回测截止到2024，3年FCFF标签截止到2021。
"""
    audit = """# 作业6 AI代码审查与修复表

| 编号 | 严重性 | 审查发现 | 影响 | 修复动作 | 验证方式 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 高 | 题目写2006-2025，但教师数据实际只有2014-2024。 | 直接照题目年份会生成空样本或虚假结果。 | 按原始文件日期改为2014-2017训练、2018-2024样本外，并在报告说明口径差异。 | `data_audit.json` 记录实际日期范围。 | 已修复 |
| 2 | 高 | 未来1年FCFF标签会让2017样本使用2018信息。 | 如果用2017标签训练再回测2018，会产生样本外泄露。 | LGBM训练标签截至2016，2017只作为2018持仓打分起点。 | `summary.json` 写明 `model_train_end_year=2016`。 | 已修复 |
| 3 | 中 | F4固资/总资产没有直接总资产字段。 | 因子无法按原定义直接计算。 | 用 `bps * total_share * assets_to_eqt` 估计总资产，保留公式说明。 | `annual_factor_panel.csv` 输出F4结果。 | 已修复 |
| 4 | 中 | 评分标准提到11个因子，但正文列F1-F10。 | 交付可能被认为少一个长期现金流因子。 | 在F10股息率后补充F11 FCFF收益率，并在报告解释该补充。 | `factor_quality_summary.csv` 包含F1-F11。 | 已修复 |
| 5 | 中 | 小样本LightGBM容易过拟合。 | 训练误差好但样本外分层失真。 | 限制 `max_depth=3`、`num_leaves=7`、`min_data_in_leaf=5` 并使用时间序列CV。 | `lgbm_cv_results.csv` 输出验证MSE。 | 已修复 |
| 6 | 中 | FCFF增速在FCFF接近0或由负转正时存在极端值。 | IC、OLS和LGBM会被少数异常标签主导。 | 新增 `target_fcff_growth_1y` 和 `target_fcff_growth_3y_ann`，按年度截面3σ缩尾后用于检验、训练和分层统计，原始标签保留。 | `annual_factor_panel.csv` 同时包含原始标签和缩尾标签。 | 已修复 |
"""
    (config.OUTPUT_DIR / "AI交互记录.md").write_text(interaction, encoding="utf-8")
    (config.OUTPUT_DIR / "AI代码审查与修复表.md").write_text(audit, encoding="utf-8")


def write_report(summary: dict, ic_ir: pd.DataFrame, importance: pd.DataFrame, fcff_summary: pd.DataFrame, price_metrics: pd.DataFrame) -> None:
    top_ic = ic_ir.sort_values("IC均值", ascending=False).head(5)
    top_imp = importance.head(5)
    price_a = price_metrics[price_metrics["group"].eq("A")].copy()
    ml_a = price_a[price_a["scheme"].str.contains("LGBM", na=False)]
    rule_a = price_a[price_a["scheme"].str.contains("传统", na=False)]
    ml_metric = ml_a.iloc[0] if not ml_a.empty else None
    rule_metric = rule_a.iloc[0] if not rule_a.empty else None

    lines = [
        "# 作业6B：基于FCFF的价值投资实验报告",
        "",
        f"学生：{config.STUDENT_NAME}  ",
        f"学号：{config.STUDENT_ID}",
        "",
        "## 1. 任务理解与可复现口径",
        "",
        "本作业将巴菲特-芒格“三好价值投资”拆解为财务价值因子，并按因子预处理、IC/IR、VIF、单因子OLS、LGBM非线性赋权、双策略回测的全链路完成实证。两套方案分别是：方案A固定阈值传统价值规则，方案B基于LGBM的财务因子综合打分。",
        "",
        f"题目正文写明数据范围为2006-2025，但教师提供文件实际覆盖 {summary['data_audit']['trade_date_min']} 至 {summary['data_audit']['trade_date_max']}。因此本报告采用可复现口径：{config.TRAIN_START_YEAR}-{config.TRAIN_END_YEAR}为训练研究窗口，{config.TEST_START_YEAR}-{config.TEST_END_YEAR}为样本外价格回测窗口；未来3年FCFF标签因数据截止2024，样本外可验证到{config.FCFF_3Y_TEST_END_YEAR}年截面。",
        "",
        "评分表提到11个财务因子，而正文列到F10。为避免长期现金流维度不足，本次在F10股息率后补充F11 FCFF收益率，所有因子均按年度截面正向化、3σ缩尾和Z标准化。FCFF增速标签容易受接近0分母影响，因此IC、OLS、LGBM和FCFF分层统计使用年度截面3σ缩尾后的标签，同时在面板中保留原始标签备查。",
        "",
        "## 2. 数据与因子",
        "",
        f"- 年度面板：{summary['data_audit']['panel_rows']}行，{summary['data_audit']['stock_count']}只股票。",
        "- 标签Y：未来1年FCFF增速，年度截面缩尾后用于IC、OLS和LGBM训练。",
        "- FCFF分层回测标签：未来3年FCFF年化增速，年度截面缩尾后统计。",
        "- FCFF计算：经营现金流净额约等于 `ocfps * total_share`，固定资产投入用固定资产正增量代理。",
        "- 股价回测：上一年年末分组，下一年等权持有，单边手续费0.1%，沪深300为基准。",
        "",
        "## 3. 因子质检结论",
        "",
        "| 因子 | IC均值 | IR | IC有效 | IR达标 |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for _, row in top_ic.iterrows():
        lines.append(
            f"| {row['因子']} | {row['IC均值']:.4f} | {row['IR']:.4f} | {row['IC有效']} | {row['IR达标']} |"
        )

    lines += [
        "",
        "VIF检验采用阈值10迭代剔除高共线因子；单因子OLS用于识别线性有效因子，p值不显著的因子仍允许进入LGBM捕捉非线性关系。",
        "",
        "## 4. LGBM重要因子",
        "",
        "| 排名 | 因子 | 重要性占比 |",
        "| ---: | --- | ---: |",
    ]
    for i, (_, row) in enumerate(top_imp.iterrows(), start=1):
        lines.append(f"| {i} | {row['因子']} | {_pct(row['importance_share'])} |")

    lines += [
        "",
        "## 5. 双策略回测表现",
        "",
        "### FCFF分层",
        "",
        "| 方案 | 样本 | A组 | B组 | C组 | A-C差值 | 单调性 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    diff = fcff_summary.drop_duplicates(["scheme", "样本"])[
        ["scheme", "样本", "A组3年FCFF增速均值", "B组3年FCFF增速均值", "C组3年FCFF增速均值", "A-C差值", "是否单调A>B>C"]
    ]
    for _, row in diff.iterrows():
        lines.append(
            f"| {row['scheme']} | {row['样本']} | {_pct(row['A组3年FCFF增速均值'])} | "
            f"{_pct(row['B组3年FCFF增速均值'])} | {_pct(row['C组3年FCFF增速均值'])} | "
            f"{_pct(row['A-C差值'])} | {row['是否单调A>B>C']} |"
        )

    lines += ["", "### 股价收益（样本外A组）", ""]
    if rule_metric is not None and len(rule_metric) > 0:
        lines.append(
            f"- 方案A传统规则A组：累计收益率 {_pct(rule_metric['累计收益率'])}，年化收益率 {_pct(rule_metric['年化收益率'])}，超额收益 {_pct(rule_metric['超额收益'])}，最大回撤 {_pct(rule_metric['最大回撤'])}。"
        )
    if ml_metric is not None and len(ml_metric) > 0:
        lines.append(
            f"- 方案B LGBM A组：累计收益率 {_pct(ml_metric['累计收益率'])}，年化收益率 {_pct(ml_metric['年化收益率'])}，超额收益 {_pct(ml_metric['超额收益'])}，最大回撤 {_pct(ml_metric['最大回撤'])}。"
        )

    lines += [
        "",
        "## 6. 思考题回答",
        "",
        "1. 芒格把经济护城河和商业模式置于首位，是因为长期复利来自企业持续创造现金流的能力，而不是价格短期波动。技术分析或短线趋势跟踪主要捕捉交易行为和价格惯性，价值投资首先判断企业能否长期以高资本回报率再投资。",
        "",
        "2. 巴菲特强调自由现金流，是因为净利润可能受应收账款、存货、折旧政策和一次性项目影响；自由现金流更接近股东真实可分配资金。高利润低现金流企业常见于大量赊销、资本开支重或利润确认提前的行业。",
        "",
        "3. 从IC/IR和重要度看，核心有效因子应同时满足经济含义和样本外分层能力。若部分分红或利润率因子无效，可能来自上证50行业结构偏金融、样本股票少、2014以后风格切换以及财务指标低频滞后。",
        "",
        "4. 固定阈值策略透明、可解释、抗过拟合，但阈值僵硬，容易漏掉边际优秀公司；LGBM能捕捉非线性和交互关系，但小样本下更依赖正则和样本外验证。二者收益和回撤差异主要来自选股集中度、行业暴露和对现金流因子的权重处理。",
        "",
        "5. 本价值多因子模型是低频、基本面、年度调仓，IC周期以年度FCFF为核心；4B短线中频模型是日频或短周期行情/流动性因子，标签是未来1-2日收益，持仓周期短，建模重点是交易信号和快速调仓。两者的因子逻辑、检验周期和风险来源完全不同。",
        "",
        "## 7. 文件索引",
        "",
        "- `outputs/homework6/factor_quality_summary.csv`：IC/IR/VIF/OLS汇总。",
        "- `outputs/homework6/feature_importance.csv`：LGBM特征重要度。",
        "- `outputs/homework6/strategy_yearly_groups.csv`：两套策略年度分组与入选股票。",
        "- `outputs/homework6/fcff_group_summary.csv`：FCFF分层回测指标。",
        "- `outputs/homework6/price_metrics.csv`：股价回测指标。",
        "- `outputs/homework6/*.png`：核心图表。",
        "- `outputs/homework6/202331060205_丁致宇_作业6.pptx`：课堂汇报PPT。",
    ]
    text = "\n".join(lines)
    (config.OUTPUT_DIR / "homework6_report.md").write_text(text, encoding="utf-8")
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "homework6_report.md").write_text(text, encoding="utf-8")


def build_ppt_if_requested() -> None:
    sys.path.insert(0, str(config.ROOT))
    from report.build_homework6_ppt import build

    build()


def run(build_ppt: bool = False) -> dict:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("作业6B：基于FCFF的价值投资 - 双策略全链路实证")
    print("=" * 72)

    print("\n[1/7] 加载教师配套数据并构建年度因子面板")
    tables = load_raw_tables()
    panel = build_annual_panel(tables)
    audit = data_audit(panel, tables)
    save_data_description(audit)
    print(f"年度面板: {len(panel)}行，{panel['ts_code'].nunique()}只股票")
    print(audit["assignment_period_note"])

    print("\n[2/7] 因子正向化、缩尾、标准化与传统规则打分")
    panel = preprocess_factors(panel)
    panel = add_traditional_scores(panel)
    save_panel(panel, audit)

    print("\n[3/7] IC/IR、VIF和单因子OLS质检")
    ic_ir, yearly_ic = compute_ic_ir(panel)
    vif_df, keep_features = compute_vif(panel)
    ols_df = single_factor_ols(panel)
    model_features = select_model_features(ic_ir, keep_features)
    quality = (
        ic_ir.merge(vif_df[["factor", "VIF", "判定"]], on="factor", how="left")
        .merge(ols_df[["factor", "β均值", "p值均值", "t统计量均值", "有效截面数", "判定"]], on="factor", how="left", suffixes=("_VIF", "_OLS"))
    )
    _save_csv(ic_ir, "ic_ir_summary.csv")
    _save_csv(yearly_ic, "ic_yearly.csv")
    _save_csv(vif_df, "vif_results.csv")
    _save_csv(ols_df, "single_factor_ols.csv")
    _save_csv(quality, "factor_quality_summary.csv")
    print("LGBM特征:", ", ".join(model_features))

    print("\n[4/7] 训练LGBM并生成综合价值得分")
    scored, model, cv_results, importance = run_modeling(panel, model_features)

    print("\n[5/7] 双策略FCFF分层回测与股价收益回测")
    strategy_panel = build_strategy_groups(scored)
    fcff_yearly, fcff_summary = fcff_group_backtest(strategy_panel)
    stock_daily, hs300 = load_price_data(tables)
    price_nav, annual_returns, price_metrics = price_group_backtest(strategy_panel, stock_daily, hs300)
    save_backtest_outputs(strategy_panel, fcff_yearly, fcff_summary, price_nav, annual_returns, price_metrics)

    print("\n[6/7] 生成图表、报告和AI记录")
    plot_paths = generate_all_plots(
        ic_ir,
        importance,
        fcff_summary,
        price_nav,
        annual_returns,
        strategy_panel,
        price_metrics,
    )
    summary = {
        "student": {"name": config.STUDENT_NAME, "id": config.STUDENT_ID},
        "data_audit": audit,
        "train_years": [config.TRAIN_START_YEAR, config.TRAIN_END_YEAR],
        "model_train_end_year": config.MODEL_TRAIN_END_YEAR,
        "test_years": [config.TEST_START_YEAR, config.TEST_END_YEAR],
        "model_features": model_features,
        "plot_paths": plot_paths,
        "top_ic_factors": ic_ir.head(5).to_dict(orient="records"),
        "top_importance": importance.head(5).to_dict(orient="records"),
        "price_metrics_A_group": price_metrics[price_metrics["group"].eq("A")].to_dict(orient="records"),
    }
    (config.OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_safe),
        encoding="utf-8",
    )
    write_ai_records()
    write_report(summary, ic_ir, importance, fcff_summary, price_metrics)

    if build_ppt:
        print("\n[7/7] 构建PPT")
        build_ppt_if_requested()
    else:
        print("\n[7/7] 已跳过PPT构建，可使用 --build-ppt 生成")

    print(f"\n作业6完成，输出目录: {config.OUTPUT_DIR}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="作业6 FCFF价值投资分析")
    parser.add_argument("--build-ppt", action="store_true", help="分析完成后同步生成作业6 PPT")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(build_ppt=args.build_ppt)


if __name__ == "__main__":
    main()
