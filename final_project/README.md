# 期末大作业：行业轮动多因子量化策略

2026 春《金融与经济数据挖掘》期末大作业。以申万一级 31 个行业指数为标的，构建行业轮动多因子量化策略：
四因子计算 → Z-Score 标准化 → IC/IR+共线性双质检 → LGBM 赋权 → 月末截面打分 → 样本外回测 → 风控与鲁棒性。

作业原始要求（原封不动提取）见 [assignment.md](assignment.md)，完整学期报告见 `outputs/final_project/report.md`（及同名 `.docx`）。

## 运行方法

在仓库根目录执行：

```powershell
# 全流程：数据→因子→双质检→LGBM→回测→鲁棒性→风控→图表→报告
python -m final_project.main

# 额外生成 Word 报告（提交用）
python -m final_project.main --build-docx
```

依赖：`pandas numpy scipy scikit-learn lightgbm matplotlib python-docx`（见根目录 `requirements.txt`）。

## 数据来源

教师配套数据（已复制到 `data/final_project/`，无需联网下载）：

| 文件 | 内容 | 区间 |
| --- | --- | --- |
| `hs300_daily.csv` | 沪深300日行情（OHLC+成交额） | 2022-10-10 ~ 2025-12-31，787日 |
| `sw_industry_daily.csv` | 申万一级31行业日行情（收盘价+成交额） | 同上，31×787 |
| `sw_industry_pb.csv` | 申万一级31行业日度PB | 同上，31×787 |

行业代码-名称映射见 `industry_codes.py`（申万2021版一级31行业，801XXX.SI）。

## 模块结构

```text
final_project/
├── assignment.md          # 作业要求原封不动提取
├── README.md              # 本文件
├── __init__.py
├── config.py              # 路径、时间区间、因子参数、回测规则、LGBM超参
├── industry_codes.py      # 31个申万一级行业代码-名称映射
├── data_loader.py         # 三份CSV对齐为日频面板、月末交易日
├── factors.py             # 四因子计算 + 截面Z-Score + 前瞻收益
├── quality.py             # IC/IR有效性检验 + 共线性检验（双质检）
├── model.py               # LGBM训练/CV/赋权 + 综合得分（IC方向定向）
├── backtest.py            # 月度轮动回测引擎 + 回撤减仓风控 + 核心指标
├── robustness.py          # 参数/时间/方法敏感性 + 风控谱系
├── viz.py                 # 因子质检/LGBM/回测/鲁棒性图表
├── report.py              # 学期报告Markdown + Word(.docx) + AI审核表/交互记录
└── main.py                # 全流程编排入口
```

## 方法论要点（与作业要求对应）

- **因子（1.4.2）**：景气度(mom_mid,60日累计涨跌)、估值(pb,月末PB-LF)、动量(mom_short,20日累计涨跌)、资金流(flow,20日成交额占比)；月末跨31行业Z-Score标准化。
- **双质检（1.3.3）**：IC(Spearman)/IR + 共线性(Pearson)，阈值0.85，无因子被剔除，保留全部4因子。
- **赋权打分（1.3.3）**：LGBM回归(预测下月收益)特征重要度为权重，按样本内IC符号定向，综合得分=Σ(权重×方向×z)；月末选Top5等权。
- **回测（1.3.4/1.3.5）**：月末收盘生成信号(行业指数仅收盘价)、等权Top5(单行业≤30%)、单边佣金万分之三按|Δ权重|计入、日频复利、基准沪深300；严格按前瞻收益月份划分样本，无未来函数。
- **样本划分**：样本内2023-02~2024-12（23月713条，训练+质检）；样本外2025全年（12月，达标验证）。
- **风控（模块4）**：回撤触发减仓谱系，证实可在-5%触发/40%暴露下同时满足12%-18%收益与≤10%回撤双目标。
- **鲁棒性（模块5）**：Top3 vs Top5、2024 vs 2025、综合得分 vs LGBM直接预测三组对比。

## 关键结果（2025样本外，LGBM综合得分Top5）

| 指标 | 数值 |
| --- | ---: |
| 累计收益 | 28.24% |
| 年化收益 | 29.43% |
| 最大回撤 | -16.27% |
| 夏普比率 | 1.295 |
| 沪深300年化 | 18.37% |
| 相对超额(累计) | +10.58% |
| 达标[年化12-18%] | ✗（偏高） |
| 达标[回撤≤10%] | ✗（破红线） |

策略显著跑赢沪深300（+10.58%），但回撤超10%红线；叠加回撤减仓风控后可同时满足双目标（详见报告2.5）。

## 输出文件

全部产出在 `outputs/final_project/`：

- **报告**：`report.md`、`202331060205_丁致宇_期末大作业.docx`、`AI代码审核表.md`、`AI交互记录.md`
- **图表**：`ic_ir.png`、`collinearity.png`、`lgbm_importances.png`、`nav_primary.png`、`drawdown_primary.png`、`monthly_primary.png`、`holdings_primary.png`、`robustness_parameter.png`、`robustness_time.png`、`risk_spectrum.png`
- **数据**：`factor_panel.csv`、`ic_summary.csv`、`ic_series.csv`、`corr_matrix.csv`、`lgbm_importances.csv`、`lgbm_cv_metrics.csv`、`nav_primary.csv`、`monthly_primary.csv`、`holdings_primary.csv`、`metrics_primary.csv`、`robustness_*.csv`、`risk_control_spectrum.csv`

## 风险提示

本策略仅用于课程学习与方法实验，回测结果依赖历史数据与模型参数，不代表未来收益，不构成任何投资建议。
