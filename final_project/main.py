"""期末大作业主入口：行业轮动多因子策略全流程编排。

用法（仓库根目录）：
    python -m final_project.main                # 运行全流程，产出图表/数据/报告
    python -m final_project.main --build-docx   # 额外生成 Word 报告
    python -m final_project.main --refresh      # 强制重算（默认即重算）
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from . import config as cfg
from . import viz
from .backtest import backtest
from .data_loader import load_market
from .factors import build_factor_panel, filter_panel
from .model import composite_score, ic_directions, train_lgbm
from .quality import run_quality_check
from .report import build_ai_interaction_md, build_ai_review_md, build_docx, build_report_md
from .robustness import run_all

warnings.filterwarnings("ignore")  # 屏蔽 lightgbm 特征名提示等无关警告


def _save_csv(df: pd.DataFrame, path: Path, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, encoding="utf-8-sig", **kw)
    print(f"  · {path.name}")


def main(with_docx: bool = False):
    out_dir = cfg.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录：{out_dir}")

    # 1. 数据与因子
    print("[1/7] 加载数据与构建因子面板...")
    market = load_market()
    panel = build_factor_panel(market)
    train = filter_panel(panel, cfg.IN_SAMPLE_START, cfg.IN_SAMPLE_END, by="fwd_date")
    print(f"  交易日 {len(market.dates)}，行业 {market.n_industries}，因子面板 {panel.shape}，"
          f"训练集 {len(train)} 条 / {train.signal_date.nunique()} 月")

    # 2. 因子双质检
    print("[2/7] 因子双质检（IC/IR + 共线性）...")
    qc = run_quality_check(train)
    out_oos = filter_panel(panel, cfg.OUT_SAMPLE_START, cfg.OUT_SAMPLE_END, by="fwd_date")
    qc_oos = run_quality_check(out_oos)
    _save_csv(qc["ic_summary"].reset_index().rename(columns={"index": "factor"}), out_dir / "ic_summary.csv")
    _save_csv(qc["ic_series"], out_dir / "ic_series.csv")
    _save_csv(qc["corr_matrix"].reset_index().rename(columns={"index": "factor"}), out_dir / "corr_matrix.csv")
    _save_csv(qc["advisory"].reset_index().rename(columns={"index": "factor"}), out_dir / "factor_advisory.csv")

    # 3. LGBM 赋权
    print("[3/7] LGBM 因子赋权与打分...")
    model = train_lgbm(train)
    directions = ic_directions(train)
    _save_csv(model["importances"].reset_index().rename(columns={"index": "factor"}), out_dir / "lgbm_importances.csv")
    _save_csv(model["cv_metrics"], out_dir / "lgbm_cv_metrics.csv")
    _save_csv((model["importances"] * directions).reset_index().rename(columns={"index": "factor", 0: "signed_weight"}),
              out_dir / "lgbm_signed_weights.csv")
    panel = panel.copy()
    panel["score"] = composite_score(panel, model["importances"], directions)
    _save_csv(panel, out_dir / "factor_panel.csv", index=False)

    # 4. 主回测（2025 样本外）
    print("[4/7] 回测（2025 样本外 + 2023-2025 全周期）...")
    b25 = filter_panel(panel, cfg.OUT_SAMPLE_START, cfg.OUT_SAMPLE_END, by="fwd_date")
    primary = backtest(b25, market, top_n=cfg.TOP_N_DEFAULT, name="LGBM综合得分_Top5_2025样本外")
    _save_csv(primary.nav.rename("nav").to_frame(), out_dir / "nav_primary.csv")
    _save_csv(primary.benchmark_nav.rename("benchmark_nav").to_frame(), out_dir / "benchmark_nav_primary.csv")
    _save_csv(primary.monthly_returns, out_dir / "monthly_primary.csv", index=False)
    _save_csv(primary.holdings, out_dir / "holdings_primary.csv", index=False)
    _save_csv(pd.DataFrame([primary.metrics]).T.rename(columns={0: "value"}), out_dir / "metrics_primary.csv")
    bf = filter_panel(panel, cfg.FULL_START, cfg.FULL_END, by="fwd_date")
    full_bt = backtest(bf, market, top_n=cfg.TOP_N_DEFAULT, name="全周期_2023-2025")

    # 5. 鲁棒性与风险控制
    print("[5/7] 鲁棒性测试与风险控制谱系...")
    rob = run_all(panel, market, panel.drop(columns=["score"]), model, directions, primary)
    _save_csv(rob["parameter"]["table"], out_dir / "robustness_parameter.csv", index=False)
    _save_csv(rob["time"]["table"], out_dir / "robustness_time.csv", index=False)
    _save_csv(rob["method"]["table"], out_dir / "robustness_method.csv", index=False)
    _save_csv(rob["risk_control"]["table"], out_dir / "risk_control_spectrum.csv", index=False)

    # 6. 图表
    print("[6/7] 生成图表...")
    viz.plot_ic_ir(qc, out_dir / "ic_ir.png")
    viz.plot_collinearity(qc["corr_matrix"], out_dir / "collinearity.png")
    viz.plot_lgbm_importances(model["importances"], out_dir / "lgbm_importances.png")
    viz.plot_nav(primary, out_dir / "nav_primary.png")
    viz.plot_drawdown(primary, out_dir / "drawdown_primary.png")
    viz.plot_monthly_returns(primary, out_dir / "monthly_primary.png")
    viz.plot_holdings_heatmap(primary, out_dir / "holdings_primary.png")
    viz.plot_robustness_compare(rob["parameter"]["results"], out_dir / "robustness_parameter.png",
                                "参数敏感性：Top3 vs Top5（2025样本外）")
    viz.plot_robustness_compare(rob["time"]["results"], out_dir / "robustness_time.png",
                                "时间敏感性：2024(样本内) vs 2025(样本外)")
    viz.plot_risk_spectrum(rob["risk_control"]["table"], out_dir / "risk_spectrum.png")
    print("  · 全部图表已生成")

    # 7. 报告
    print("[7/7] 生成报告...")
    ctx = {"qc": qc, "qc_oos": qc_oos, "model_result": model, "directions": directions,
           "primary": primary, "full_bt": full_bt, "robustness": rob, "output_dir": out_dir}
    report_md = build_report_md(ctx)
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")
    (out_dir / "AI代码审核表.md").write_text(build_ai_review_md(), encoding="utf-8")
    (out_dir / "AI交互记录.md").write_text(build_ai_interaction_md(), encoding="utf-8")
    print("  · report.md / AI代码审核表.md / AI交互记录.md")

    if with_docx:
        docx_path = out_dir / f"{cfg.STUDENT_ID}_{cfg.STUDENT_NAME}_期末大作业.docx"
        build_docx(ctx, docx_path)
        print(f"  · {docx_path.name}")

    # 控制台摘要
    print("\n" + "=" * 70)
    print("主回测（2025 样本外，LGBM综合得分 Top5）")
    M = primary.metrics
    print(f"  累计收益 {M['cumulative_return']:.2%} | 年化 {M['annualized_return']:.2%} | "
          f"最大回撤 {M['max_drawdown']:.2%} | 夏普 {M['sharpe']:.3f}")
    print(f"  沪深300年化 {M['benchmark_annualized']:.2%} | 超额 {M['excess_cumulative']:.2%}")
    print(f"  达标[年化12-18%]={M['target_return_ok']}  达标[回撤≤10%]={M['target_dd_ok']}")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="期末大作业：行业轮动多因子策略")
    ap.add_argument("--build-docx", action="store_true", help="生成 Word 报告")
    ap.add_argument("--refresh", action="store_true", help="强制重算（默认即重算）")
    args = ap.parse_args()
    main(with_docx=args.build_docx)
