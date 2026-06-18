# 作业7：统计套利之平稳性检验

本目录保存作业7的完整可复现代码。教师提供的原始Word要求已复制到 `data/homework7/original/`，正文已原样提取为 `assignment.md`。

## 运行方法

```powershell
python -m homework7.main
```

如需重新从Baostock获取数据：

```powershell
python -m homework7.main --refresh
```

如需同步生成PPT：

```powershell
python -m homework7.main --build-ppt
```

## 输出文件

核心结果输出到 `outputs/homework7/`：

- `stationarity_summary.csv`：五只股票收盘价与对数收益率ADF检验汇总表。
- `stationarity_panel.csv`：日度收盘价与对数收益率明细。
- `*_price_return.png`：每只股票的收盘价和收益率时序图。
- `adf_pvalue_comparison.png`：收盘价与对数收益率ADF p值对比图。
- `homework7_report.md`：实验报告和理论思考题答案。
- `AI交互记录.md`：分步AI指令记录。
- `AI代码审查与修复表.md`：代码审查与修复说明。
- `202331060205_丁致宇_作业7_CodeBuddy版平稳性检验.pptx`：课堂汇报PPT。

