# 实验三：LGBM非线性预测与羊群效应因子分析

本目录保存实验三的独立可复现实验代码。实验三读取实验二输出的 `weekly_herd_index.csv`，并读取老师配套的沪深300日度价格数据，完成自然周对齐、时序特征工程、LightGBM 样本外预测、SHAP 解释、残差诊断和双向传导验证。

## 运行方法

在仓库根目录执行：

```powershell
python -m experiment3.main --build-pdf
```

如果 `outputs/experiment2/weekly_herd_index.csv` 不存在，请先执行：

```powershell
python -m experiment1.main
python -m experiment2.main
```

## 输入文件

- `outputs/experiment2/weekly_herd_index.csv`：实验二生成的 H1、H2、H3 羊群效应指标。
- `data/experiment2/raw/沪深300日价格指数.xls`：老师提供的沪深300日度价格指数数据。

## 输出文件

结果统一写入 `outputs/experiment3/`：

- `aligned_weekly_dataset.csv`：自然周对齐后的 H3、P_t 与沪深300周收益率。
- `feature_engineering.csv`：无未来信息泄露的滞后、滚动和时间虚拟变量特征表。
- `lgbm_forward_results.csv`：正向模型测试集真实值、预测值和残差。
- `model_metrics.csv`：正向/反向模型样本外预测指标。
- `feature_importance_gain.csv`、`shap_importance.csv`：LGBM增益和SHAP重要性表。
- `bidirectional_comparison.csv`：情绪到收益、收益到情绪的双向建模对比。
- `experiment3_report.md`：实验三 Markdown 报告。
- `202331060205_丁致宇_实验三_LGBM非线性预测与羊群效应因子分析报告.pdf`：可提交 PDF 报告。
- `AI交互记录.md`、`AI代码审查与修复表.md`、`实验三代码附录.md`：AI辅助与代码附录材料。

