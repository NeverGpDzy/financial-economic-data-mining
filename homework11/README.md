# 作业11：银行信用评分模型

消费金融用户信用评分与违约风险预测。实验要求见 `assignment.md`，实验报告见 `实验报告.md`。

## 运行方法

```powershell
python homework11/homework11_credit_scorecard.py
```

## 数据来源

- 训练集：`data/homework11/train.csv`（80万条，47个特征）
- 测试集：`data/homework11/testA.csv`（20万条，46个特征）
- 阿里天池公开数据集：违约贷款数据集（ID:140861）

## 功能模块

1. **数据加载与探索性分析（EDA）**：加载训练集/测试集，统计基本信息、目标变量分布、缺失值分析
2. **特征预处理**：数据清洗、分类特征编码、数值特征标准化
3. **特征相关性分析**：计算特征间相关性，剔除冗余特征
4. **特征工程**：特征分箱 + WOE/IV值计算，筛选有效特征
5. **模型构建**：逻辑回归模型训练，交叉验证
6. **模型评估**：AUC、KS、准确率、精确率、召回率、F1，绘制ROC曲线和KS曲线
7. **信用评分卡生成**：基于PDO和基础好坏比，将模型概率转换为信用评分
8. **测试集预测**：输出测试集用户信用分数、违约概率、信用等级
9. **特征重要性分析**：挖掘核心影响因素，给出信用提升建议

## 输出文件

核心结果输出到 `outputs/homework11/`：

**图表：**
- `target_distribution.png`：目标变量分布图
- `missing_values.png`：缺失值分析图
- `correlation_heatmap.png`：特征相关性热力图
- `feature_iv_values.png`：特征IV值柱状图
- `roc_ks_curves.png`：ROC曲线和KS曲线
- `confusion_matrix.png`：混淆矩阵
- `feature_importance.png`：特征重要性图（系数绝对值 + 系数方向）
- `scorecard_feature_contributions.png`：评分卡特征分数贡献图
- `credit_score_analysis.png`：信用分数分布、等级违约率、分数vs违约概率散点图

**数据：**
- `credit_scorecard.csv`：信用评分卡（特征系数与分数贡献）
- `test_predictions.csv`：测试集预测结果（信用分数、违约概率、信用等级）
- `model_evaluation_results.csv`：模型评估指标汇总
- `feature_importance.csv`：特征重要性数据
- `feature_iv_values.csv`：特征IV值数据
- `credit_grade_statistics.csv`：信用等级统计

## 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| RANDOM_SEED | 42 | 随机种子 |
| TEST_SIZE | 0.2 | 验证集比例 |
| CV_FOLDS | 5 | 交叉验证折数 |
| IV_THRESHOLD | 0.02 | IV值筛选阈值 |
| CORR_THRESHOLD | 0.8 | 相关性剔除阈值 |
| BASE_SCORE | 600 | 信用评分基础分 |
| PDO | 50 | Points to Double the Odds |
| SCORE_MIN / SCORE_MAX | 300 / 850 | 评分范围 |

## 交付物清单

| 交付物 | 文件 |
|--------|------|
| 完整可运行的Python源代码 | `homework11_credit_scorecard.py` |
| 特征重要性可视化图表 | `outputs/homework11/feature_importance.png` |
| 标准化信用评分规则/评分卡 | `outputs/homework11/credit_scorecard.csv` |
| 实验报告 | `实验报告.md` |
