"""Build the three LaTeX experiment reports from existing outputs.

The reports intentionally share one structure so each independent experiment
submission satisfies the unified checklist required by the assignment.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from string import Template

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
REPORT_DIR = ROOT / "report"
STUDENT_NAME = "丁致宇"
STUDENT_ID = "202331060205"


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def pct(value: float) -> str:
    return f"{value * 100:.2f}\\%"


def f6(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return tex_escape(value)


def tex_path(path: str) -> str:
    return rf"\path{{{path}}}"


def cn_file(name: str) -> str:
    return tex_escape(name)


def table_rows(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    view = df.loc[:, columns].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    rows: list[str] = []
    for _, row in view.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, pd.Timestamp):
                cells.append(value.strftime("%Y-%m-%d"))
            elif isinstance(value, float):
                cells.append(f"{value:.6f}")
            else:
                cells.append(tex_escape(value))
        rows.append(" & ".join(cells) + r" \\")
    return "\n        ".join(rows)


def copy_style(report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    source = REPORT_DIR / "experiment1_latex" / "thesis-style.sty"
    target = report_dir / "thesis-style.sty"
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    (report_dir / "latexmkrc").write_text(
        "$pdf_mode = 5;\n$xelatex = 'xelatex -interaction=nonstopmode -synctex=1 %O %S';\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def postprocess_ai_audit_latex(path: Path) -> None:
    """Keep the DOCX table structure, but make it fit A4 portrait pages."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    def table_group(widths: list[float]) -> list[str] | None:
        if len(widths) == 7:
            column_spec = (
                r"@{}L{0.045\textwidth}L{0.190\textwidth}L{0.140\textwidth}"
                r"L{0.190\textwidth}L{0.060\textwidth}L{0.060\textwidth}"
                r"L{0.205\textwidth}@{}"
            )
            return [
                r"\begingroup",
                r"\tiny",
                r"\setlength{\tabcolsep}{1pt}",
                r"\renewcommand{\arraystretch}{1.16}",
                rf"\begin{{longtable}}{{{column_spec}}}",
            ]
        if len(widths) == 2:
            if widths[0] > 0.45:
                size = r"\scriptsize"
                column_spec = r"@{}L{0.28\textwidth}L{0.66\textwidth}@{}"
                tabcolsep = "2pt"
            else:
                size = r"\small"
                column_spec = r"@{}L{0.24\textwidth}L{0.70\textwidth}@{}"
                tabcolsep = "3pt"
            return [
                r"\begingroup",
                size,
                rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
                r"\renewcommand{\arraystretch}{1.16}",
                rf"\begin{{longtable}}{{{column_spec}}}",
            ]
        return None

    output: list[str] = []
    in_grouped_table = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == r"\begin{longtable}[]{@{}":
            widths: list[float] = []
            j = i + 1
            while j < len(lines):
                widths.extend(float(value) for value in re.findall(r"\\real\{([^}]+)\}", lines[j]))
                if lines[j].strip().endswith(r"@{}}"):
                    break
                j += 1
            group = table_group(widths)
            if group is not None:
                output.extend(group)
                in_grouped_table = True
                i = j + 1
                continue

        output.append(line)
        if in_grouped_table and line.strip() == r"\end{longtable}":
            output.append(r"\endgroup")
            in_grouped_table = False
        i += 1
    text = "\n".join(output)
    if path.read_text(encoding="utf-8").endswith("\n"):
        text += "\n"
    text = text.replace("/", r"/\allowbreak{}")
    text = text.replace(r"\_", r"\_\allowbreak{}")
    text = text.replace(":", r":\allowbreak{}")
    text = text.replace("; ", r";\allowbreak{} ")
    path.write_text(text, encoding="utf-8")


def build_submission_checklist(exp_name: str, submit_dir: Path) -> str:
    audit_doc = "AI代码审查与修复表.docx"
    chat_doc = "AI交互记录.docx"
    code_zip = "代码附件.zip"
    charts_zip = "结果图表表格.zip"
    code_md = "代码附录.md"
    submit_text = tex_escape(str(submit_dir.relative_to(PARENT)).replace("\\", "/"))
    return rf"""
\section{{提交材料与作业要求对照}}

本报告为{exp_name}独立报告。为避免老师只阅读本报告时遗漏附件，本节逐条对照最初作业要求列出证据位置。AI 代码审核表、AI 交互记录截图、代码附件和图表附件均已放在父目录“{submit_text}”的提交文件中。

{{\small
\begin{{longtable}}{{@{{}}L{{0.07\textwidth}}L{{0.30\textwidth}}L{{0.50\textwidth}}L{{0.08\textwidth}}@{{}}}}
\caption{{{exp_name}作业要求满足情况自查表}}\\
\toprule
序号 & 要求 & 本报告与附件中的对应证据 & 状态 \\
\midrule
\endfirsthead
\toprule
序号 & 要求 & 本报告与附件中的对应证据 & 状态 \\
\midrule
\endhead
1 & 提交 AI 代码审核表，记录生成、校验与修改过程 & 本报告附录单独列出“AI 代码审核表”；“AI 交互记录”附录并入交互原文与截图；父目录提交文件为“{cn_file(audit_doc)}”和“{cn_file(chat_doc)}” & 已满足 \\
2 & 提交全套可运行代码，覆盖数据对齐、特征工程、时序划分、LGBM、残差、SHAP、可视化 & 提交目录中的“{cn_file(code_zip)}”与“{cn_file(code_md)}”；本报告附录列出核心源码 & 已满足 \\
3 & 附数据集说明：区间、样本量、特征、标签 & 正文“数据集说明”与“特征工程”章节 & 已满足 \\
4 & 附 LGBM 测试集预测指标表 & 正文“LGBM 样本外预测结果”章节 & 已满足 \\
5 & 附特征重要性排序，说明羊群效应贡献和最优滞后阶数 & 正文“特征重要性与羊群效应贡献”章节 & 已满足 \\
6 & 附 SHAP 依赖图并解释非线性关系与线性假设失效 & 正文“SHAP 非线性解释”章节 & 已满足 \\
7 & 附双向建模结果对比，分析情绪与收益传导强度和周期 & 正文“双向传导与金融反身性”章节 & 已满足 \\
8 & 提交收益率与羊群效应对比图、残差图、SHAP 图 & 图表嵌入正文；原图打包在“{cn_file(charts_zip)}” & 已满足 \\
9 & 实验总结回答四个核心问题 & 正文“实验总结”逐条回答 & 已满足 \\
10 & 截取运行结果与图表，附简要解读与完整代码 & 正文含表格、截图型图表与解读；完整代码见附录和 {tex_path(code_zip)} & 已满足 \\
\bottomrule
\end{{longtable}}
}}
"""


def common_preamble(title: str, subtitle: str, header: str, graphic_paths: list[str]) -> str:
    paths = "".join("{" + p + "/}" for p in graphic_paths)
    return rf"""\documentclass[UTF8,a4paper,12pt]{{ctexart}}
\usepackage{{thesis-style}}
\usepackage{{amsmath}}
\usepackage{{calc}}
\providecommand{{\tightlist}}{{\setlength{{\itemsep}}{{0pt}}\setlength{{\parskip}}{{0pt}}}}

\graphicspath{{{paths}}}
\headermark{{{header}}}
\reporttitle{{《金融经济数据挖掘》实验报告}}
\reportsubtitle{{{subtitle}}}
\author{{{STUDENT_NAME}}}
\studentid{{{STUDENT_ID}}}
\major{{数据科学与大数据技术}}
\phone{{18765788600}}
\date{{2026年6月}}

\begin{{document}}

\maketitle

\begin{{ustcabstract}}
{title}
\end{{ustcabstract}}
"""


def prepare_ai_assets(exp_num: int) -> None:
    """Extract AI audit text/tables and chat screenshots from submitted DOCX files."""
    cn = exp_cn(exp_num)
    report_dir = REPORT_DIR / f"experiment{exp_num}_latex"
    report_dir.mkdir(parents=True, exist_ok=True)
    submit_dir = PARENT / f"实验{cn}" / "提交内容"
    audit_doc = submit_dir / f"{STUDENT_ID}_{STUDENT_NAME}_实验{cn}_AI代码审查与修复表.docx"
    chat_doc = submit_dir / f"{STUDENT_ID}_{STUDENT_NAME}_实验{cn}_AI交互记录.docx"
    if not audit_doc.exists():
        raise FileNotFoundError(f"缺少 AI 代码审查表：{audit_doc}")
    if not chat_doc.exists():
        raise FileNotFoundError(f"缺少 AI 交互记录：{chat_doc}")

    subprocess.run(
        [
            "pandoc",
            "--track-changes=all",
            str(audit_doc),
            "-t",
            "latex",
            "--wrap=none",
            "-o",
            str(report_dir / "ai_audit_from_docx.tex"),
            f"--extract-media={report_dir / 'ai_audit_media'}",
        ],
        check=True,
        cwd=ROOT,
    )
    postprocess_ai_audit_latex(report_dir / "ai_audit_from_docx.tex")
    subprocess.run(
        [
            "pandoc",
            "--track-changes=all",
            str(audit_doc),
            "-t",
            "markdown",
            "--wrap=none",
            "-o",
            str(report_dir / "ai_audit_from_docx.md"),
        ],
        check=True,
        cwd=ROOT,
    )

    chat_md = report_dir / "ai_chat_from_docx.md"
    subprocess.run(
        [
            "pandoc",
            "--track-changes=all",
            str(chat_doc),
            "-t",
            "markdown",
            "--wrap=none",
            "-o",
            str(chat_md),
            f"--extract-media={report_dir / 'ai_chat_media'}",
        ],
        check=True,
        cwd=ROOT,
    )

    lines = chat_md.read_text(encoding="utf-8").splitlines()
    text_only: list[str] = []
    for line in lines:
        if line.startswith("![]"):
            break
        text_only.append(line)
    text_md = report_dir / "ai_chat_text_only.md"
    text_md.write_text("\n".join(text_only).strip() + "\n", encoding="utf-8")
    subprocess.run(
        [
            "pandoc",
            str(text_md),
            "-t",
            "latex",
            "--wrap=none",
            "--shift-heading-level-by=2",
            "-o",
            str(report_dir / "ai_chat_text_from_docx.tex"),
        ],
        check=True,
        cwd=ROOT,
    )


def ai_process_appendix(exp_num: int, exp_name: str) -> str:
    report_dir = REPORT_DIR / f"experiment{exp_num}_latex"
    image_dir = report_dir / "ai_chat_media" / "media"
    images = sorted(image_dir.glob("image*.png"), key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0))
    figure_blocks = []
    for idx, image in enumerate(images, start=1):
        rel = image.relative_to(report_dir).as_posix()
        figure_blocks.append(
            rf"""
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth,height=0.72\textheight,keepaspectratio]{{{rel}}}
\caption{{{exp_name} AI 交互记录截图 {idx}}}
\end{{figure}}
"""
        )
    figures_tex = "\n".join(figure_blocks) if figure_blocks else "未在 AI 交互记录 docx 中检测到图片。"
    return rf"""

\clearpage
\section{{AI 代码审核表}}

本节直接从父目录对应提交文件夹中的 AI 代码审核表 docx 抽取内容，单独作为报告附录列出，用于记录代码生成、校验与修改的完整过程。该内容同时保留在提交目录的原始 docx 文件中，便于核验。

\input{{ai_audit_from_docx.tex}}

\clearpage
\section{{AI 交互记录}}

本节直接从父目录对应提交文件夹中的 AI 交互记录 docx 抽取内容，将交互记录原文与 docx 中的截图原图并入本 LaTeX/PDF 报告。它与上一节“AI 代码审核表”共同构成第一条要求所需的生成、校验、修改和人工审查证据链。

\subsection{{AI 交互记录原文}}
{{\small
\input{{ai_chat_text_from_docx.tex}}
}}

\subsection{{AI 交互记录截图}}
{figures_tex}
"""


def report_tail(code_inputs: list[tuple[str, str]], exp_name: str, exp_num: int) -> str:
    module_map = {"实验一": "experiment1", "实验二": "experiment2", "实验三": "experiment3"}
    module_name = module_map.get(exp_name, "experiment")
    parts = [
        r"\clearpage",
        r"\appendix",
        r"\section{完整代码附录与复现入口}",
        (
            "完整可运行代码已随提交目录中的代码附件 zip 和代码附录 Markdown 一并提交。"
            "本 LaTeX 附录列出所有核心源码文件、职责和运行入口；实际完整代码以"
            rf"“{cn_file(f'{STUDENT_ID}_{STUDENT_NAME}_{exp_name}_代码附件.zip')}”"
            rf"和“{cn_file(f'{STUDENT_ID}_{STUDENT_NAME}_{exp_name}_代码附录.md')}”为准。"
            "这种组织方式可以避免报告因超长源码而难以编译，同时保证老师能够按附件直接复现实验。"
        ),
        "",
        r"\begin{longtable}{p{0.32\textwidth}p{0.58\textwidth}}",
        rf"\caption{{{exp_name}核心源码文件清单}}\\",
        r"\toprule",
        r"文件 & 作用 \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"文件 & 作用 \\",
        r"\midrule",
        r"\endhead",
    ]
    for caption, path in code_inputs:
        desc = {
            "config.py": "路径、日期范围、模型参数和学生信息配置。",
            "data.py": "原始数据读取、清洗、交易日过滤、周度聚合和数据库写入。",
            "sentiment.py": "FinBERT 情绪标注与词典回退逻辑。",
            "analysis.py": "羊群效应指标、特征工程、LightGBM、SHAP、残差和双向建模。",
            "plots.py": "收益率、羊群效应、特征重要性、残差、SHAP 等核心可视化。",
            "main.py": "实验主入口，串联全流程并生成 CSV、SQLite、图表与报告材料。",
        }.get(Path(caption).name, "实验核心源码。")
        parts.append(rf"\path{{{caption}}} & {desc} \\")
    parts.extend(
        [
            r"\bottomrule",
            r"\end{longtable}",
            "",
            r"\subsection{复现命令}",
            rf"\begin{{CodeBlock}}",
            rf"pip install -r requirements.txt",
            rf"python -m {module_name}.main",
            rf"\end{{CodeBlock}}",
        ]
    )
    parts.append(ai_process_appendix(exp_num, exp_name))
    parts.append(r"\end{document}")
    return "\n\n".join(parts)


def experiment1_tex() -> str:
    summary = read_json(ROOT / "outputs" / "experiment1" / "summary.json")
    gain = pd.read_csv(ROOT / "outputs" / "experiment1" / "lgbm_feature_importance.csv").head(8)
    shap = pd.read_csv(ROOT / "outputs" / "experiment1" / "shap_importance.csv").head(8)
    bidirectional = pd.DataFrame(summary["bidirectional_comparison"])
    submit_dir = PARENT / "实验一" / "提交内容"
    checklist = build_submission_checklist("实验一", submit_dir)

    forward = summary["forward_metrics"]
    backward = summary["backward_metrics"]
    audit = summary["audit"]
    best_feature = summary["best_shap_feature"]
    gain_rows = table_rows(gain, ["feature", "importance"])
    shap_rows = table_rows(shap, ["feature", "importance"])
    bi_rows = table_rows(bidirectional, ["direction", "mse", "mae", "r2", "ic", "direction_acc", "best_lag"])
    h3_share = sum(
        float(row["importance"]) for row in gain.to_dict("records") if str(row["feature"]).startswith("H3")
    ) / float(pd.read_csv(ROOT / "outputs" / "experiment1" / "lgbm_feature_importance.csv")["importance"].sum())

    body = Template(r"""
\keywords{金融文本挖掘；BERT；市场情绪；羊群效应；LightGBM；SHAP；金融反身性}

\clearpage
\tableofcontents
\clearpage

$checklist

\section{实验目标与完整链路}

实验一以金融新闻非结构化文本为起点，完成新闻清洗、BERT 情绪标注、5 个交易日周度聚合，并在同一可运行代码链路中继续生成羊群效应指标、沪深300周收益率、LGBM 非线性预测、残差诊断、SHAP 分析和双向传导验证。这样安排的原因是：实验一输出的周度情绪表是后续两个实验的基础；同时，为满足统一提交要求，本报告将完整实证链路一并呈现。

全流程入口为：
\begin{CodeBlock}
python -m experiment1.main
\end{CodeBlock}

\section{数据集说明}

\begin{table}[H]
\centering
\caption{实验一数据集与样本审计}
\begin{tabular}{lr}
\toprule
项目 & 数值 \\
\midrule
原始新闻行数 & $raw_rows \\
日期范围与正文非空过滤后行数 & $date_rows \\
正文重复删除行数 & $duplicate_rows \\
剔除非交易日或交易日历外新闻行数 & $non_trading_rows \\
进入情绪标注的交易日新闻行数 & $labeled_rows \\
新闻样本区间 & $date_start 至 $date_end \\
情绪标注方法 & BERT 本地推理 \\
周度情绪表行数 & $weekly_rows \\
建模对齐数据行数 & $modeling_rows \\
特征工程有效样本行数 & $feature_rows \\
\bottomrule
\end{tabular}
\end{table}

标签定义分两层：情绪标注阶段的文本标签为正面、中性、负面三分类；预测建模阶段的标签为沪深300周收益率 \(Ret_t\)。核心特征包括 \(H3_{t-1}\) 至 \(H3_{t-5}\)、3 周和 5 周滚动均值与标准差、月份和季度特征。所有预测特征均来自历史周，避免使用目标周同期信息。

\section{数据清洗、情绪标注与羊群效应构建}

程序先将新闻发布时间转为标准日期，保留 2014-10-01 至 2015-10-31 区间内正文非空的样本，再按正文去重，并用沪深300交易日历剔除非交易日新闻。情绪标注使用 \path{yiyanghkust/finbert-tone-chinese} 完成正面、中性、负面三分类；当模型文件不可用时，代码提供金融情绪词典回退路径以保证可运行性。

周度情绪占比定义为：
\begin{equation}
P_t=\frac{WeekPositive_t}{WeekPositive_t+WeekNegative_t}.
\end{equation}

羊群效应指标按如下方式构造：
\begin{equation}
H1_t=P_t-E(P_t),\quad
H2_t=1-\frac{|WeekPositive_t-WeekNegative_t|}{WeekPositive_t+WeekNegative_t},
\end{equation}
\begin{equation}
H3_t=Norm(|H1_t|)\times(1-Norm(H2_t)).
\end{equation}
其中 \(H2_t\) 表示分歧度，越低代表观点越单边，因此在 \(H3_t\) 合成时取 \(1-Norm(H2_t)\) 表示一致性强度。

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{weekly_sentiment_counts.png}
\caption{周度情绪数量统计图}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{herd_index_timeseries.png}
\caption{收益率建模前的羊群效应指数时序图}
\end{figure}

\section{数据对齐、特征工程与时序划分}

沪深300周收益率使用每 5 个交易日一组后的最后收盘价计算：
\begin{equation}
Ret_t=\frac{Close_t-Close_{t-1}}{Close_{t-1}}.
\end{equation}
程序将 \(H3_t\) 与沪深300周收益率按交易周末日期对齐，并生成 \path{modeling_dataset.csv} 与 \path{feature_engineering.csv}。特征工程包含 \(H3\) 滞后 1 至 5 期、历史滚动均值、历史滚动标准差以及月份、季度变量。训练测试划分严格遵循时间顺序：前 80\% 为训练集，后 20\% 为测试集，测试集样本数为 9。

\section{LGBM 样本外预测结果}

\begin{table}[H]
\centering
\caption{实验一完整链路 LGBM 测试集预测指标}
\begin{tabular}{lr}
\toprule
指标 & 数值 \\
\midrule
MSE & $mse \\
MAE & $mae \\
\(R^2\) & $r2 \\
IC（Spearman 秩相关） & $ic \\
方向准确率 & $direction_acc \\
测试集样本数 & $n_test \\
\bottomrule
\end{tabular}
\end{table}

正向模型 \(R^2\) 为 $r2，说明当前小样本下预测误差仍高于均值基准；IC 为 $ic，表示排序关系存在弱信号，但方向准确率仅为 $direction_acc，不能把羊群效应解释为稳定的单向择时因子。

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{herd_vs_hs300_return.png}
\caption{沪深300周收益率与羊群效应时序对比图}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.82\textwidth]{residual_timeseries.png}
\caption{LGBM 测试集残差时序图}
\end{figure}

\section{特征重要性与羊群效应贡献}

LightGBM 特征重要性前 8 项如下。按 Gain 排序，最优羊群效应预测滞后为 \texttt{H3\_lag5}；在当前模型中，\(H3\) 相关特征贡献占比约为 $h3_share。

\begin{table}[H]
\centering
\caption{LightGBM 特征重要性排序}
\begin{tabular}{lr}
\toprule
特征 & 重要性 \\
\midrule
$gain_rows
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.78\textwidth]{feature_importance.png}
\caption{LightGBM 特征重要性图}
\end{figure}

\section{SHAP 非线性解释}

\begin{table}[H]
\centering
\caption{SHAP 平均绝对贡献排序}
\begin{tabular}{lr}
\toprule
特征 & 平均绝对 SHAP \\
\midrule
$shap_rows
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.78\textwidth]{shap_dependence.png}
\caption{SHAP 依赖图：$best_feature}
\end{figure}

SHAP 依赖图显示，当 \texttt{H3\_lag5} 较低时，模型倾向于给收益预测正向贡献；当 \texttt{H3\_lag5} 升高到中高区间后，SHAP 值明显转为负值。这种“低位正贡献、高位负贡献”的分段形态说明羊群效应与指数收益之间不是固定线性斜率关系，而更像存在阈值和反转机制。线性模型只能给出单一系数，难以同时表达这种区间差异，因此线性假设在本任务中容易失效。

\section{双向传导与金融反身性}

\begin{table}[H]
\centering
\scriptsize
\caption{双向建模结果对比}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrr}
\toprule
方向 & MSE & MAE & \(R^2\) & IC & 方向准确率 & 最优滞后 \\
\midrule
$bi_rows
\bottomrule
\end{tabular}}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.78\textwidth]{bidirectional_comparison.png}
\caption{情绪与收益双向传导对比图}
\end{figure}

反向模型 \(R^2=$backward_r2\)，高于正向模型 \(R^2=$r2\)；反向 IC 为 $backward_ic，方向准确率为 $backward_acc。这说明当前样本中收益对后续羊群情绪的反馈强于情绪对收益的预测，符合金融反身性理论中“市场价格改变投资者认知，认知再影响后续市场行为”的反馈链条。

\section{实验总结}

\textbf{第一，羊群效应与指数收益是否存在显著预测关联？} 样本外结果显示存在可检验但较弱的非线性关联，正向模型 \(R^2\) 为负，说明整体预测能力有限；IC 为正，提示排序层面有弱信号。

\textbf{第二，最优预测周期和非线性特征是什么？} 实验一完整链路下，最优羊群效应滞后为 5 周。SHAP 图显示低 \(H3\) 区间贡献偏正、中高 \(H3\) 区间贡献偏负，呈现分段和阈值特征。

\textbf{第三，情绪与收益存在怎样的双向传导？} 反向模型强于正向模型，说明收益变化更容易影响后续舆情羊群强度。市场下跌或上涨改变新闻叙事与投资者预期，进而形成反馈循环。

\textbf{第四，线性假设与非线性建模如何取舍？} 金融市场更适合非线性分析框架，因为情绪变量影响收益时常表现为阈值、反转、滞后和反馈，而不是稳定单一斜率。

""").substitute(
        checklist=checklist,
        raw_rows=audit["raw_news_rows"],
        date_rows=audit["rows_after_date_and_non_null_filter"],
        duplicate_rows=audit["duplicate_content_removed"],
        non_trading_rows=audit["non_trading_or_out_of_calendar_removed"],
        labeled_rows=audit["labeled_trading_day_rows"],
        date_start=audit["date_start"],
        date_end=audit["date_end"],
        weekly_rows=summary["weekly_rows"],
        modeling_rows=summary["modeling_rows"],
        feature_rows=summary["feature_rows"],
        mse=f6(forward["mse"]),
        mae=f6(forward["mae"]),
        r2=f"{forward['r2']:.4f}",
        ic=f"{forward['ic']:.4f}",
        direction_acc=pct(forward["direction_acc"]),
        n_test=int(forward["n_test"]),
        h3_share=pct(h3_share),
        gain_rows=gain_rows,
        shap_rows=shap_rows,
        best_feature=tex_escape(best_feature),
        bi_rows=bi_rows,
        backward_r2=f"{backward['r2']:.4f}",
        backward_ic=f"{backward['ic']:.4f}",
        backward_acc=pct(backward["direction_acc"]),
    )

    head = common_preamble(
        "本报告为实验一独立报告，围绕金融新闻文本清洗、BERT 情绪标注、周度情绪聚合展开，并按照统一作业要求补充完整的数据对齐、特征工程、LightGBM 样本外预测、残差诊断、SHAP 解释、双向传导和代码审查材料。实验共处理原始新闻 60000 条，清洗后进入交易日情绪标注的新闻为 30623 条，生成 51 行周度情绪指标和 45 行机器学习特征样本。正向模型测试集 MSE 为 0.006118，MAE 为 0.053541，\\(R^2=-0.2759\\)；双向建模显示收益率对后续羊群效应的反馈更强，体现出金融反身性。",
        "实验一：金融非结构化数据预处理、情绪量化与完整预测链路",
        "《金融经济数据挖掘》实验一报告",
        ["../../outputs/experiment1", "assets"],
    )
    tail = report_tail(
        [
            ("experiment1/config.py", "../../experiment1/config.py"),
            ("experiment1/data.py", "../../experiment1/data.py"),
            ("experiment1/sentiment.py", "../../experiment1/sentiment.py"),
            ("experiment1/analysis.py", "../../experiment1/analysis.py"),
            ("experiment1/plots.py", "../../experiment1/plots.py"),
            ("experiment1/main.py", "../../experiment1/main.py"),
        ],
        "实验一",
        1,
    )
    return head + body + tail


def experiment2_tex() -> str:
    exp2_summary = read_json(ROOT / "outputs" / "experiment2" / "summary.json")
    exp3_summary = read_json(ROOT / "outputs" / "experiment3" / "summary.json")
    top = pd.read_csv(ROOT / "outputs" / "experiment2" / "top_herd_weeks.csv").head(10)
    gain = pd.read_csv(ROOT / "outputs" / "experiment3" / "feature_importance_gain.csv").head(8)
    shap = pd.read_csv(ROOT / "outputs" / "experiment3" / "shap_importance.csv").head(8)
    comparison = pd.read_csv(ROOT / "outputs" / "experiment3" / "bidirectional_comparison.csv")
    submit_dir = PARENT / "实验二" / "提交内容"
    checklist = build_submission_checklist("实验二", submit_dir)

    audit = exp2_summary["audit"]
    forward = exp3_summary["forward_metrics"]
    backward = exp3_summary["backward_metrics"]
    contrib = exp3_summary["contribution"]
    top_rows = table_rows(top, ["rank", "trade_date", "P_t", "E_P_t", "H1t", "H2t", "H3t", "NewsCount"])
    gain_rows = table_rows(gain, ["feature", "gain", "split", "gain_share"])
    shap_rows = table_rows(shap, ["feature", "mean_abs_shap", "shap_share"])
    bi_rows = table_rows(comparison, ["direction_short", "mse", "mae", "rmse", "r2", "ic", "direction_acc", "best_lag"])

    body = Template(r"""
\keywords{市场情绪；羊群效应；H3 指数；LightGBM；SHAP；金融反身性}

\clearpage
\tableofcontents
\clearpage

$checklist

\section{实验目标与完整链路定位}

实验二的核心任务是把实验一生成的周度情绪指标转化为羊群效应指数。为满足统一作业要求，本报告不仅说明 \(H1_t\)、\(H2_t\)、\(H3_t\) 的构建过程，还继续引用由实验二输出 \(H3_t\) 接入实验三 LGBM 模块后的数据对齐、特征工程、残差诊断、SHAP 和双向传导结果，使本报告单独阅读时也能覆盖完整实证链路。

实验二本体运行入口为：
\begin{CodeBlock}
python -m experiment2.main
\end{CodeBlock}
完整预测链路运行入口为：
\begin{CodeBlock}
python -m experiment3.main
\end{CodeBlock}

\section{数据集说明}

\begin{table}[H]
\centering
\caption{实验二羊群效应构建样本摘要}
\begin{tabular}{lr}
\toprule
项目 & 数值 \\
\midrule
输入周度情绪样本行数 & $input_rows \\
无正负情绪分母剔除行数 & $zero_rows \\
无历史基准剔除行数 & $baseline_rows \\
3\(\sigma\) 异常值剔除行数 & $outlier_rows \\
最终羊群指标输出行数 & $output_rows \\
羊群指标日期范围 & $date_start 至 $date_end \\
建模对齐样本行数 & $aligned_rows \\
特征工程有效样本行数 & $feature_rows \\
训练集/测试集样本 & $n_train / $n_test \\
\bottomrule
\end{tabular}
\end{table}

实验二输入特征为 \texttt{WeekPositive}、\texttt{WeekNeutral}、\texttt{WeekNegative}、\texttt{NewsCount} 和由正负新闻计算得到的 \(P_t\)。预测建模阶段的标签为沪深300自然周收益率 \(Ret_t\)，建模特征为 \(H3\) 与 \(P_t\) 的历史滞后项、滚动统计量以及时间虚拟变量。

\section{羊群效应指标构建}

周度乐观情绪占比定义为：
\begin{equation}
P_t=\frac{WeekPositive_t}{WeekPositive_t+WeekNegative_t}.
\end{equation}
历史基准情绪采用过去 4 周均值：
\begin{equation}
E(P_t)=\frac{1}{4}\sum_{i=1}^{4}P_{t-i}.
\end{equation}
情绪偏离度为：
\begin{equation}
H1_t=P_t-E(P_t).
\end{equation}
观点分歧度为：
\begin{equation}
H2_t=1-\left|\frac{WeekPositive_t-WeekNegative_t}{WeekPositive_t+WeekNegative_t}\right|.
\end{equation}
综合羊群效应指数为：
\begin{equation}
H3_t=Norm(|H1_t|)\times(1-Norm(H2_t)).
\end{equation}

该定义保证 \(H3_t\) 同时捕捉“情绪反常”和“观点一边倒”。如果只有普通乐观情绪，但没有明显历史偏离，或正负观点高度分歧，\(H3_t\) 都不会被显著抬高。

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{herd_index_timeseries.png}
\caption{实验二羊群效应指标时序图}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{sentiment_vs_herd.png}
\caption{情绪占比与羊群效应对比图}
\end{figure}

\section{高羊群效应周与运行结果截图}

\begin{table}[H]
\centering
\scriptsize
\caption{H3 排名前 10 的交易周}
\resizebox{\textwidth}{!}{%
\begin{tabular}{rlrrrrrr}
\toprule
排名 & 交易日 & \(P_t\) & \(E(P_t)\) & \(H1_t\) & \(H2_t\) & \(H3_t\) & 新闻数 \\
\midrule
$top_rows
\bottomrule
\end{tabular}}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{top_herd_weeks.png}
\caption{H3 最高的 10 个交易周截图}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{weekly_herd_index_table.png}
\caption{羊群效应指标表截图}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.94\textwidth]{core_program_snippet.png}
\caption{核心程序运行与指标计算截图}
\end{figure}

\section{数据对齐、特征工程与时序划分}

为检验羊群效应是否具有预测价值，程序将实验二生成的 \(H3_t\) 与沪深300自然周收益率对齐。正向模型只使用 \(H3_{t-1}\) 至 \(H3_{t-5}\)、历史滚动均值和历史滚动标准差预测 \(Ret_t\)，不使用目标周同期 \(H3_t\)。样本按时间顺序切分，训练集 36 行，测试集 9 行。

\begin{figure}[H]
\centering
\includegraphics[width=0.94\textwidth]{modeling_dataset_table.png}
\caption{实验二输出接入预测链路后的建模数据集截图}
\end{figure}

\section{LGBM 样本外预测结果}

\begin{table}[H]
\centering
\caption{基于实验二 \(H3_t\) 的 LGBM 测试集预测指标}
\begin{tabular}{lr}
\toprule
指标 & 数值 \\
\midrule
MSE & $mse \\
MAE & $mae \\
RMSE & $rmse \\
\(R^2\) & $r2 \\
IC & $ic \\
方向准确率 & $direction_acc \\
\bottomrule
\end{tabular}
\end{table}

正向模型 \(R^2=$r2\)，说明严格样本外收益率预测仍弱于均值基准；但方向准确率为 $direction_acc，IC 为 $ic，显示羊群效应及其辅助情绪变量仍提供弱排序信号。

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{market_herd_timeseries.png}
\caption{收益率与羊群效应自然周时序对比图}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.82\textwidth]{residual_timeseries.png}
\caption{LGBM 测试集残差时序图}
\end{figure}

\section{特征重要性与羊群效应贡献}

\begin{table}[H]
\centering
\scriptsize
\caption{LightGBM Gain 特征重要性前 8}
\begin{tabular}{lrrr}
\toprule
特征 & Gain & Split & Gain 占比 \\
\midrule
$gain_rows
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.82\textwidth]{feature_importance_gain.png}
\caption{LightGBM Gain 特征重要性图}
\end{figure}

\(H3\) 类特征 Gain 总贡献占比为 $h3_share，最优 Gain 滞后为 \texttt{$best_gain_lag}；按 SHAP 排序，最优单一 \(H3\) 滞后项为 \texttt{$best_shap_lag}。这说明羊群效应本身和情绪占比的历史结构都参与了模型分裂和预测。

\section{SHAP 非线性解释}

\begin{table}[H]
\centering
\scriptsize
\caption{SHAP 特征重要性前 8}
\begin{tabular}{lrr}
\toprule
特征 & 平均绝对 SHAP & SHAP 占比 \\
\midrule
$shap_rows
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.78\textwidth]{shap_dependence_best_lag.png}
\caption{SHAP 依赖图：最优 \(H3\) 滞后因子}
\end{figure}

SHAP 依赖图显示 \(H3\) 滞后因子的影响存在区间差异：低位 \(H3\) 对收益预测有正向贡献，中高位 \(H3\) 后贡献趋于负向。这说明情绪羊群效应并非越强越利好，过高的羊群一致性可能意味着拥挤交易、预期透支或反转风险。固定线性系数无法表达这种阈值与反转，因此非线性模型更适合本实验。

\section{双向传导与金融反身性}

\begin{table}[H]
\centering
\scriptsize
\caption{双向建模结果对比}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrl}
\toprule
方向 & MSE & MAE & RMSE & \(R^2\) & IC & 方向准确率 & 最优滞后 \\
\midrule
$bi_rows
\bottomrule
\end{tabular}}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.82\textwidth]{bidirectional_comparison.png}
\caption{情绪与收益双向建模对比图}
\end{figure}

反向模型 \(R^2=$backward_r2\)，高于正向模型，且最优滞后为 \texttt{$best_backward_lag}，说明当前样本中市场收益变化对下一阶段舆情羊群强度的反馈更明显。金融反身性理论认为价格和认知相互塑造，本实验结果更接近“收益改变新闻叙事和投资者情绪，再影响后续行为”的反馈路径。

\section{实验总结}

\textbf{第一，羊群效应与指数收益是否存在显著预测关联？} 存在弱非线性预测关联，但整体预测能力有限，正向 \(R^2\) 为负，说明不能简单把 \(H3_t\) 视为稳定收益预测因子。

\textbf{第二，最优预测周期和非线性特征是什么？} Gain 视角下最优 \(H3\) 滞后为 \texttt{$best_gain_lag}，SHAP 视角下最优单一滞后为 \texttt{$best_shap_lag}。关系呈现低位正贡献、高位负贡献的非线性形态。

\textbf{第三，情绪与收益双向传导如何解释？} 收益到情绪的反向传导更强，说明市场涨跌对新闻叙事和投资者羊群行为具有更明显的反馈作用。

\textbf{第四，线性与非线性框架如何选择？} 金融市场更适合非线性框架，因为羊群效应往往包含阈值、拥挤、反转和反馈链条，线性假设难以完整表达。

""").substitute(
        checklist=checklist,
        input_rows=audit["input_rows"],
        zero_rows=audit["dropped_zero_denominator"],
        baseline_rows=audit["dropped_no_history_baseline"],
        outlier_rows=audit["outlier_removed"],
        output_rows=audit["output_rows"],
        date_start=audit["date_start"],
        date_end=audit["date_end"],
        aligned_rows=exp3_summary["alignment_audit"]["aligned_rows"],
        feature_rows=exp3_summary["feature_rows"],
        n_train=forward["n_train"],
        n_test=forward["n_test"],
        top_rows=top_rows,
        mse=f6(forward["mse"]),
        mae=f6(forward["mae"]),
        rmse=f6(forward["rmse"]),
        r2=f"{forward['r2']:.4f}",
        ic=f"{forward['ic']:.4f}",
        direction_acc=pct(forward["direction_acc"]),
        gain_rows=gain_rows,
        h3_share=pct(contrib["h3_gain_share"]),
        best_gain_lag=tex_escape(exp3_summary["best_gain_lag"]),
        best_shap_lag=tex_escape(exp3_summary["best_shap_lag"]),
        shap_rows=shap_rows,
        bi_rows=bi_rows,
        backward_r2=f"{backward['r2']:.4f}",
        best_backward_lag=tex_escape(exp3_summary["best_backward_lag"]),
    )

    head = common_preamble(
        "本报告为实验二独立报告，重点完成基于周度情绪的羊群效应指数构建，并按照统一作业要求接入后续 LGBM 非线性预测、SHAP 解释、残差诊断和双向传导分析。实验二读取 51 行周度情绪样本，剔除首期历史基准缺失后输出 50 行 \(H1_t\)、\(H2_t\)、\(H3_t\) 指标；最高羊群效应周为 2015-09-16，\(H3_t=0.864687\)。基于该 \(H3_t\) 的后续预测链路显示，收益对情绪的反馈强于情绪对收益的先行预测。",
        "实验二：羊群效应指数构建与非线性预测验证",
        "《金融经济数据挖掘》实验二报告",
        ["../../outputs/experiment2", "../../outputs/experiment3"],
    )
    tail = report_tail(
        [
            ("experiment2/config.py", "../../experiment2/config.py"),
            ("experiment2/analysis.py", "../../experiment2/analysis.py"),
            ("experiment2/plots.py", "../../experiment2/plots.py"),
            ("experiment2/main.py", "../../experiment2/main.py"),
            ("experiment3/analysis.py", "../../experiment3/analysis.py"),
            ("experiment3/main.py", "../../experiment3/main.py"),
        ],
        "实验二",
        2,
    )
    return head + body + tail


def experiment3_tex() -> str:
    summary = read_json(ROOT / "outputs" / "experiment3" / "summary.json")
    gain = pd.read_csv(ROOT / "outputs" / "experiment3" / "feature_importance_gain.csv").head(10)
    shap = pd.read_csv(ROOT / "outputs" / "experiment3" / "shap_importance.csv").head(10)
    comparison = pd.read_csv(ROOT / "outputs" / "experiment3" / "bidirectional_comparison.csv")
    residual = pd.read_csv(ROOT / "outputs" / "experiment3" / "residual_diagnostics.csv")
    submit_dir = PARENT / "实验三" / "提交内容"
    checklist = build_submission_checklist("实验三", submit_dir)

    audit = summary["alignment_audit"]
    forward = summary["forward_metrics"]
    backward = summary["backward_metrics"]
    contrib = summary["contribution"]
    gain_rows = table_rows(gain, ["feature", "gain", "split", "gain_share"])
    shap_rows = table_rows(shap, ["feature", "mean_abs_shap", "shap_share"])
    bi_rows = table_rows(comparison, ["direction_short", "mse", "mae", "rmse", "r2", "ic", "direction_acc", "best_lag"])
    residual_rows = table_rows(residual, ["metric", "value"])

    body = Template(r"""
\keywords{羊群效应；沪深300；LightGBM；SHAP；残差诊断；金融反身性}

\clearpage
\tableofcontents
\clearpage

$checklist

\section{实验目标与建模框架}

实验三以实验二输出的周度羊群效应指数为核心解释变量，检验历史 \(H3_t\) 是否能够预测沪深300周收益率，并进一步通过 SHAP 和双向建模解释情绪与收益之间的非线性反馈。本实验不是同期相关性检验，而是样本外预测任务，因此所有情绪特征均处理为滞后项或历史滚动统计量。

运行入口为：
\begin{CodeBlock}
python -m experiment3.main
\end{CodeBlock}

\section{数据集说明}

\begin{table}[H]
\centering
\caption{实验三数据集与样本审计}
\begin{tabular}{lr}
\toprule
项目 & 数值 \\
\midrule
实验二 \(H3_t\) 周样本 & $herd_rows \\
沪深300自然周价格样本 & $hs300_rows \\
自然周对齐样本 & $aligned_rows \\
对齐日期范围 & $aligned_start 至 $aligned_end \\
对齐后关键字段缺失 & $missing_cells \\
特征工程后可建模样本 & $feature_rows \\
训练集样本 & $n_train \\
测试集样本 & $n_test \\
\bottomrule
\end{tabular}
\end{table}

标签为 \(Ret_t\)，即当周沪深300收益率。正向特征包括 \texttt{H3\_lag1} 至 \texttt{H3\_lag5}、\texttt{H3\_roll\_mean\_3}、\texttt{H3\_roll\_std\_3}、\texttt{H3\_roll\_mean\_5}、\texttt{H3\_roll\_std\_5}、\(P_t\) 的 1 至 3 期滞后、\(P_t\) 的 3 周滚动均值、月份和季度虚拟变量。

\begin{figure}[H]
\centering
\includegraphics[width=0.94\textwidth]{modeling_dataset_table.png}
\caption{建模数据集截图}
\end{figure}

\section{数据对齐、特征工程与时序划分}

程序先将羊群效应指标和沪深300日度价格分别转换为自然周标签，再按自然周结束日做内连接。收益率计算公式为：
\begin{equation}
Ret_t=\frac{Close_t-Close_{t-1}}{Close_{t-1}}.
\end{equation}
特征工程中，所有 \(H3\) 和 \(P_t\) 滚动统计均先 \texttt{shift(1)}，再计算窗口均值或标准差，避免未来信息泄露。训练集为前 80\%，测试集为后 20\%，且参数选择只在训练窗口内部完成。

\section{LGBM 样本外预测结果}

\begin{table}[H]
\centering
\caption{实验三 LGBM 测试集预测指标}
\begin{tabular}{lr}
\toprule
指标 & 数值 \\
\midrule
MSE & $mse \\
MAE & $mae \\
RMSE & $rmse \\
\(R^2\) & $r2 \\
IC & $ic \\
方向准确率 & $direction_acc \\
\bottomrule
\end{tabular}
\end{table}

正向模型测试集 \(R^2=$r2\)，表示严格样本外收益率预测弱于均值基准；方向准确率为 $direction_acc，说明模型仍能在部分样本中捕捉方向信息，但整体预测能力不强。

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{market_herd_timeseries.png}
\caption{收益率与羊群效应时序对比图}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{prediction_vs_actual.png}
\caption{测试集实际收益率与预测收益率}
\end{figure}

\section{特征重要性与羊群效应贡献}

\begin{table}[H]
\centering
\scriptsize
\caption{LightGBM Gain 特征重要性前 10}
\begin{tabular}{lrrr}
\toprule
特征 & Gain & Split & Gain 占比 \\
\midrule
$gain_rows
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.82\textwidth]{feature_importance_gain.png}
\caption{LightGBM Gain 特征重要性图}
\end{figure}

\(H3\) 类特征 Gain 贡献占比为 $h3_share，\(P_t\) 辅助情绪特征贡献占比为 $sentiment_share，时间虚拟变量贡献占比为 $time_share。按 Gain 排序，最优 \(H3\) 滞后为 \texttt{$best_gain_lag}；按 SHAP 排序，最优单一 \(H3\) 滞后为 \texttt{$best_shap_lag}。

\section{SHAP 非线性解释}

\begin{table}[H]
\centering
\scriptsize
\caption{SHAP 特征重要性前 10}
\begin{tabular}{lrr}
\toprule
特征 & 平均绝对 SHAP & SHAP 占比 \\
\midrule
$shap_rows
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.82\textwidth]{shap_feature_importance.png}
\caption{SHAP 特征重要性图}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.76\textwidth]{shap_dependence_best_lag.png}
\caption{SHAP 依赖图}
\end{figure}

SHAP 依赖图显示，低位 \(H3\) 滞后样本对应正 SHAP 值，而中高位 \(H3\) 滞后样本对应负 SHAP 值。这说明羊群效应与指数收益之间不是线性单调关系：适度情绪一致可能带来趋势延续，但过高羊群一致性可能意味着拥挤交易和反转压力。线性回归只能估计一个平均斜率，无法刻画这种阈值和分段效应，因此线性假设在本任务中失效。

\section{残差诊断}

\begin{table}[H]
\centering
\caption{测试集残差描述统计}
\begin{tabular}{lr}
\toprule
指标 & 数值 \\
\midrule
$residual_rows
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{residual_timeseries.png}
\caption{残差时序图}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.66\textwidth]{residual_distribution.png}
\caption{残差分布图}
\end{figure}

残差均值为 -0.032930，说明模型在测试集存在一定高估收益倾向；最小残差为 -0.144697，表明个别极端周的市场波动难以由羊群效应单一因子解释。

\section{双向传导与金融反身性}

\begin{table}[H]
\centering
\scriptsize
\caption{双向建模结果对比}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrl}
\toprule
方向 & MSE & MAE & RMSE & \(R^2\) & IC & 方向准确率 & 最优滞后 \\
\midrule
$bi_rows
\bottomrule
\end{tabular}}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.82\textwidth]{bidirectional_comparison.png}
\caption{情绪与收益双向建模结果对比}
\end{figure}

双向建模显示反向模型 \(R^2=$backward_r2\)，高于正向模型 \(R^2=$r2\)，且反向方向准确率为 $backward_acc。这说明市场收益对后续舆情羊群强度的反馈更强。金融反身性理论认为价格和投资者认知相互影响，本实验更支持“收益变化塑造舆情，再通过情绪影响后续市场行为”的反馈链。

\section{实验总结}

\textbf{第一，羊群效应与指数收益是否存在显著预测关联？} 存在一定非线性预测关联，但正向模型 \(R^2\) 为负，整体预测能力有限，只能说明存在弱排序和方向信号。

\textbf{第二，最优预测周期和非线性特征是什么？} Gain 视角下最优 \(H3\) 滞后为 \texttt{$best_gain_lag}，SHAP 视角下最优单一滞后为 \texttt{$best_shap_lag}。关系不是线性单调，而是低位正贡献、中高位负贡献。

\textbf{第三，情绪与收益双向传导如何解释？} 反向模型更强，说明收益变化对后续情绪羊群效应具有更明显影响，符合金融反身性中的反馈机制。

\textbf{第四，线性假设与非线性建模如何取舍？} 金融市场更适合非线性框架。羊群效应包含阈值、拥挤交易、反转和反馈链条，LightGBM 与 SHAP 能比固定线性系数更细致地刻画这些结构。

""").substitute(
        checklist=checklist,
        herd_rows=audit["herd_rows"],
        hs300_rows=audit["hs300_weekly_rows"],
        aligned_rows=audit["aligned_rows"],
        aligned_start=audit["aligned_start"],
        aligned_end=audit["aligned_end"],
        missing_cells=audit["missing_cells_after_align"],
        feature_rows=summary["feature_rows"],
        n_train=forward["n_train"],
        n_test=forward["n_test"],
        mse=f6(forward["mse"]),
        mae=f6(forward["mae"]),
        rmse=f6(forward["rmse"]),
        r2=f"{forward['r2']:.4f}",
        ic=f"{forward['ic']:.4f}",
        direction_acc=pct(forward["direction_acc"]),
        gain_rows=gain_rows,
        h3_share=pct(contrib["h3_gain_share"]),
        sentiment_share=pct(contrib["sentiment_gain_share"]),
        time_share=pct(contrib["time_gain_share"]),
        best_gain_lag=tex_escape(summary["best_gain_lag"]),
        best_shap_lag=tex_escape(summary["best_shap_lag"]),
        shap_rows=shap_rows,
        residual_rows=residual_rows,
        bi_rows=bi_rows,
        backward_r2=f"{backward['r2']:.4f}",
        backward_acc=pct(backward["direction_acc"]),
    )

    head = common_preamble(
        "本报告为实验三独立报告，在实验二羊群效应指数基础上完成自然周数据对齐、特征工程、时序划分、LightGBM 非线性预测、残差诊断、SHAP 解释、核心可视化和双向传导验证。实验共得到 50 行自然周对齐样本和 45 行可建模样本，训练集 36 行、测试集 9 行。正向模型测试集 MSE 为 0.003503，MAE 为 0.036376，\\(R^2=-0.5398\\)；反向模型 \\(R^2=0.0850\\)，说明收益对后续舆情羊群强度的反馈更明显。",
        "实验三：LGBM 非线性预测、SHAP 解释与双向传导分析",
        "《金融经济数据挖掘》实验三报告",
        ["../../outputs/experiment3"],
    )
    tail = report_tail(
        [
            ("experiment3/config.py", "../../experiment3/config.py"),
            ("experiment3/analysis.py", "../../experiment3/analysis.py"),
            ("experiment3/plots.py", "../../experiment3/plots.py"),
            ("experiment3/main.py", "../../experiment3/main.py"),
        ],
        "实验三",
        3,
    )
    return head + body + tail


def compile_report(report_dir: Path) -> Path | None:
    if shutil.which("xelatex") is None:
        print("未找到 xelatex，跳过 PDF 编译。")
        return None
    for pattern in ("main.aux", "main.toc", "main.out", "main.log", "main.synctex.gz"):
        stale = report_dir / pattern
        if stale.exists():
            stale.unlink()
    for _ in range(2):
        result = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=report_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=240,
        )
        if result.returncode != 0:
            print(result.stdout[-2000:])
            print(result.stderr[-2000:])
            raise RuntimeError(f"LaTeX 编译失败：{report_dir / 'main.tex'}")
    return report_dir / "main.pdf"


def exp_cn(exp_num: int) -> str:
    return {1: "一", 2: "二", 3: "三"}[exp_num]


def write_tex(exp_num: int, tex: str) -> Path:
    report_dir = REPORT_DIR / f"experiment{exp_num}_latex"
    copy_style(report_dir)
    tex_path = report_dir / "main.tex"
    tex_path.write_text(tex, encoding="utf-8")

    submit_dir = PARENT / f"实验{exp_cn(exp_num)}" / "提交内容"
    submit_dir.mkdir(parents=True, exist_ok=True)
    source_name = f"{STUDENT_ID}_{STUDENT_NAME}_实验{exp_cn(exp_num)}_LaTeX报告源码.tex"
    shutil.copy2(tex_path, submit_dir / source_name)
    shutil.copy2(report_dir / "thesis-style.sty", submit_dir / f"{STUDENT_ID}_{STUDENT_NAME}_实验{exp_cn(exp_num)}_thesis-style.sty")
    print(f"已写入 LaTeX：{tex_path}")
    return report_dir


def compile_and_copy(exp_num: int, report_dir: Path, pdf_name: str) -> None:
    pdf_path = compile_report(report_dir)
    submit_dir = PARENT / f"实验{exp_cn(exp_num)}" / "提交内容"
    if pdf_path and pdf_path.exists():
        shutil.copy2(pdf_path, submit_dir / pdf_name)
        print(f"已生成并复制 PDF：{submit_dir / pdf_name}")


def main() -> None:
    for exp_num in (1, 2, 3):
        copy_style(REPORT_DIR / f"experiment{exp_num}_latex")
        prepare_ai_assets(exp_num)
    jobs = [
        (1, experiment1_tex(), f"{STUDENT_ID}_{STUDENT_NAME}_实验一_金融非结构化数据预处理报告.pdf"),
        (2, experiment2_tex(), f"{STUDENT_ID}_{STUDENT_NAME}_实验二_羊群效应指数构建报告.pdf"),
        (3, experiment3_tex(), f"{STUDENT_ID}_{STUDENT_NAME}_实验三_LGBM非线性预测与羊群效应因子分析报告.pdf"),
    ]
    report_dirs = []
    for exp_num, tex, _ in jobs:
        report_dirs.append((exp_num, write_tex(exp_num, tex)))
    for (exp_num, _, pdf_name), (_, report_dir) in zip(jobs, report_dirs):
        compile_and_copy(exp_num, report_dir, pdf_name)


if __name__ == "__main__":
    main()
