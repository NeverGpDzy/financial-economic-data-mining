# 金融与经济数据挖掘课程项目

本仓库保存《金融与经济数据挖掘》课程作业的 Python 代码、报告脚本、图表结果和展示材料。

作者信息：

- 姓名：丁致宇
- 学号：202331060205

## 仓库概览

本项目围绕金融市场数据完成数据获取、特征工程、统计建模、机器学习、量化回测和可视化展示，包含作业 1、作业 2、作业 3、作业 4 和作业 4B。

| 模块 | 主题 | 主要入口 | 说明 |
| --- | --- | --- | --- |
| `homework1/` | 技术分析与机器学习预测 | `python homework1/main.py` | 白酒股日线收益预测、模型对比和交易策略回测 |
| `homework2/` | CAPM 模型 | `python homework2/main.py` | Alpha/Beta 估计、显著性检验和买入持有回测 |
| `homework3/` | 二因子模型与 PEG 策略 | `python homework3/main.py` | 市场因子 + PE_TTM 扩展检验，PEG 策略回测 |
| `homework4/` | 多因子量化选股 | `python homework4/main.py` | SMB、PE 倒数、质量因子、IC/IR、LGBM 和 TopN 回测 |
| `homework4b/` | 中频短线量化全流程 | `python -m homework4b.main` | 4 因子质检、LGBM 赋权、Top5 日频调仓样本外回测 |
| `homework7/` | 统计套利之平稳性检验 | `python -m homework7.main` | 5只股票收盘价与对数收益率ADF检验、图表、报告和PPT |
| `homework8/` | 协整检验与配对交易 | `python -m homework8.main` | EG两步法协整检验、最优配对、价差z-score交易信号 |

## 技术栈

| 类型 | 工具 |
| --- | --- |
| 数据获取 | Baostock、本地 Parquet |
| 数据处理 | pandas, numpy, scipy |
| 统计建模 | statsmodels |
| 机器学习 | scikit-learn, LightGBM |
| 可视化 | matplotlib |
| 报告与展示 | Markdown, python-pptx, PowerPoint/PDF |

## 项目结构

```text
Code/
├── common/                     # 公共数据获取模块
├── data/                       # 本地数据缓存与清洗数据说明
│   ├── homework1/
│   ├── homework2/
│   ├── homework3/
│   ├── homework4/
│   └── homework4b/             # 4B清洗后数据说明；大CSV本地保留不入库
├── homework1/                  # 作业1代码与说明
├── homework2/                  # 作业2代码与说明
├── homework3/                  # 作业3代码
├── homework4/                  # 作业4代码
├── homework4b/                 # 作业4B代码与README
├── outputs/                    # 图表、模型、PPT、PDF、指标表
├── report/                     # 报告与PPT生成脚本
├── requirements.txt
└── README.md
```

说明：

- `data/**/*.csv` 默认不提交，避免把大体积清洗数据塞进 Git；作业 4B 的 `train_data.csv`、`test_data.csv`、`full_data.csv` 会在本地由主程序重新生成。
- `outputs/homework4b/` 的最终提交材料已经作为例外纳入版本控制，包括 PPT、PDF、结果表、图表、AI 审查记录和 AI 交互截图。
- 根目录可能存在未跟踪的历史数据目录，例如 `data/homework4/`，提交前应确认是否与当前任务相关。

## 环境准备

建议使用 Python 3.10 或更高版本。

安装依赖：

```powershell
pip install -r requirements.txt
```

`requirements.txt` 包含：

```text
baostock
pandas
numpy
matplotlib
scikit-learn
lightgbm
statsmodels
python-pptx
scipy
Pillow
python-docx
pyarrow
```

## 运行方法

在仓库根目录执行。

```powershell
# 作业1
python homework1/main.py
python homework1/sensitivity_analysis.py

# 作业2
python homework2/main.py
python homework2/main.py --refresh

# 作业3
python homework3/main.py
python homework3/main.py --refresh

# 作业4
python homework4/main.py
python homework4/main.py --refresh

# 作业4B
python -m homework4b.main

# 作业7
python -m homework7.main --build-ppt

# 作业8
python -m homework8.main --build-ppt
```

作业 4B 依赖教师提供的本地 Parquet 数据，默认路径在 `homework4b/config.py` 中配置为：

```text
../作业四-B/作业4B 配套数据 parquet v1.1/BAKdata
```

## 作业4B最终口径

作业 4B 已完成一次完整审查与修复，最终口径如下：

- 因子范围：严格使用 `Reversal`、`Liquidity`、`MoneyFlow`、`Value` 四个因子。
- 样本隔离：2020-2023 仅用于因子质检、OLS 和 LGBM 训练；2024-2025 仅用于样本外打分与回测。
- 去极值：因子按交易日截面做 3σ 缩尾，避免使用样本外年份估计全样本阈值。
- 交易规则：每日 Top5 等权，初始资金 100 万元，单边交易成本 0.1%，最后交易日只结算不新开仓。
- MoneyFlow 数据限制：本地无个股级 `buy_amount/sell_amount`，使用 5 日滚动签名成交额净流入 / 流通市值作为代理。

修复后关键结果：

| 指标 | 结果 |
| --- | ---: |
| 回测区间 | 2024-01-02 ~ 2025-12-31 |
| 回测天数 | 485 |
| 累计收益率 | 15.37% |
| 年化收益率 | 7.71% |
| 最大回撤 | -40.59% |
| 日胜率 | 44.95% |
| 相对沪深300累计超额 | -21.35% |
| 夏普比率 | 0.3215 |

作业 4B 的详细说明见 [homework4b/README.md](homework4b/README.md)。

## 报告与展示材料

| 文件或目录 | 内容 |
| --- | --- |
| `report/homework1_report.md` | 作业1报告 |
| `report/homework2_report.md` | 作业2报告 |
| `report/homework3_report.md` | 作业3报告 |
| `report/homework4_report.md` | 作业4报告 |
| `report/build_homework*_ppt*.py` | 各作业PPT生成脚本 |
| `outputs/homework4b/202331060205_丁致宇_作业4B.pptx` | 作业4B最终PPT |
| `outputs/homework4b/202331060205_丁致宇_作业4B.pdf` | 作业4B最终PDF |
| `outputs/homework4b/202331060205_丁致宇_AI代码审查与修复表.docx` | 作业4B AI代码审查表 |
| `outputs/homework4b/202331060205_丁致宇_AI交互记录.docx` | 作业4B AI交互记录，含4张截图 |

## 风险提示

本项目仅用于课程学习与方法实验。所有回测结果依赖历史数据、模型参数、交易成本和交易规则，不能代表未来收益，也不构成任何投资建议。
