"""
作业11：银行信用评分模型
消费金融用户信用评分与违约风险预测

功能：
1. 数据加载与探索性分析（EDA）
2. 特征预处理：数据清洗、分类特征编码、数值特征标准化
3. 信用特征相关性分析，剔除冗余特征
4. 特征工程：特征分箱 + WOE/IV值计算
5. 模型构建：逻辑回归
6. 模型评估：AUC、KS、准确率、精确率
7. 模型训练与信用评分计算，输出用户信用分数
8. 特征重要性分析，挖掘核心影响因素
9. 基于模型结果，给出用户信用提升建议
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 全局配置参数（可配置，无硬编码魔法数字）
# ============================================================
RANDOM_SEED = 42               # 随机种子，保证结果可复现
TEST_SIZE = 0.2                # 验证集比例
CV_FOLDS = 5                   # 交叉验证折数
IV_THRESHOLD = 0.02            # IV值筛选阈值
CORR_THRESHOLD = 0.8           # 相关性剔除阈值
WOE_BINS = 10                  # WOE分箱数量
BASE_SCORE = 600               # 信用评分基础分
PDO = 50                       # Points to Double the Odds
BASE_ODDS = 50                 # 基础好坏比
SCORE_MIN = 300                # 最低信用分数
SCORE_MAX = 850                # 最高信用分数
DEFAULT_BINS = [299, 450, 550, 650, 750, 850]  # 信用等级分箱
DEFAULT_LABELS = ['D', 'C', 'B', 'A', 'S']     # 信用等级标签

# 输出目录
OUTPUT_DIR = 'outputs/homework11'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子保证可复现
np.random.seed(RANDOM_SEED)

# ============================================================
# 第一部分：数据加载与探索性分析（EDA）
# ============================================================
print("=" * 70)
print("第一部分：数据加载与探索性分析（EDA）")
print("=" * 70)

# 加载数据
print("\n1.1 加载数据...")
train_df = pd.read_csv('data/homework11/train.csv')
test_df = pd.read_csv('data/homework11/testA.csv')

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# 查看数据基本信息
print("\n1.2 数据基本信息:")
print(f"训练集列名: {train_df.columns.tolist()}")

# 目标变量分布
print("\n1.3 目标变量分布 (isDefault):")
default_counts = train_df['isDefault'].value_counts()
print(default_counts)
print(f"违约比例: {default_counts[1] / len(train_df) * 100:.2f}%")

# 可视化目标变量分布
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].pie(default_counts.values, labels=['正常还款', '违约'], autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[0].set_title('目标变量分布')

axes[1].bar(['正常还款', '违约'], default_counts.values, color=['#2ecc71', '#e74c3c'])
axes[1].set_title('目标变量计数')
axes[1].set_ylabel('数量')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: target_distribution.png")

# 缺失值分析
print("\n1.4 缺失值分析:")
missing = train_df.isnull().sum()
missing_pct = (missing / len(train_df) * 100).round(2)
missing_df = pd.DataFrame({'缺失数量': missing, '缺失比例(%)': missing_pct})
missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values('缺失比例(%)', ascending=False)
print(missing_df)

# 可视化缺失值
if len(missing_df) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))
    top_missing = missing_df.head(15)
    ax.barh(top_missing.index, top_missing['缺失比例(%)'], color='#3498db')
    ax.set_xlabel('缺失比例 (%)')
    ax.set_title('Top 15 特征缺失值比例')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/missing_values.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: missing_values.png")

# 数值特征统计
print("\n1.5 数值特征统计描述:")
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('id')
numeric_cols.remove('isDefault')
print(f"数值特征数量: {len(numeric_cols)}")

# ============================================================
# 第二部分：特征预处理
# ============================================================
print("\n" + "=" * 70)
print("第二部分：特征预处理")
print("=" * 70)

# 复制数据
train = train_df.copy()
test = test_df.copy()

# 2.1 处理employmentLength特征
print("\n2.1 处理employmentLength特征...")
def parse_employment_length(x):
    """将工作年限转换为数值"""
    if pd.isna(x):
        return np.nan
    x_str = str(x)
    if '< 1' in x_str:
        return 0.0
    if '10+' in x_str:
        return 10.0
    try:
        return float(x_str.split()[0])
    except (ValueError, IndexError):
        return np.nan

train['employmentLength'] = train['employmentLength'].apply(parse_employment_length)
test['employmentLength'] = test['employmentLength'].apply(parse_employment_length)

# 2.2 处理日期特征
print("2.2 处理日期特征...")
def extract_date_features(df):
    """提取日期特征"""
    df = df.copy()
    df['issueDate'] = pd.to_datetime(df['issueDate'])
    df['issueYear'] = df['issueDate'].dt.year
    df['issueMonth'] = df['issueDate'].dt.month
    df['issueDayOfWeek'] = df['issueDate'].dt.dayofweek
    return df

train = extract_date_features(train)
test = extract_date_features(test)

# 2.3 处理earliesCreditLine特征
print("2.3 处理earliesCreditLine特征...")
def parse_credit_line(x):
    """将最早信用额度日期转换为年份"""
    if pd.isna(x):
        return np.nan
    try:
        return float(str(x)[-4:])
    except (ValueError, IndexError):
        return np.nan

train['earliesCreditLine'] = train['earliesCreditLine'].apply(parse_credit_line)
test['earliesCreditLine'] = test['earliesCreditLine'].apply(parse_credit_line)

# 2.4 分类特征编码
print("2.4 分类特征编码...")
categorical_cols = ['grade', 'subGrade', 'homeOwnership', 'verificationStatus',
                    'purpose', 'initialListStatus', 'applicationType']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le
    print(f"  编码 {col}: {len(le.classes_)} 个类别")

# 2.5 缺失值填充 - 使用训练集的中位数填充训练集和测试集（避免数据泄露）
print("\n2.5 缺失值填充...")
exclude_cols = ['id', 'isDefault', 'issueDate']
numeric_features = [col for col in train.select_dtypes(include=[np.number]).columns
                    if col not in exclude_cols]

# 先计算训练集的中位数
median_values = train[numeric_features].median()

# 用训练集的中位数分别填充训练集和测试集
for col in numeric_features:
    if train[col].isnull().sum() > 0 or test[col].isnull().sum() > 0:
        train[col] = train[col].fillna(median_values[col])
        test[col] = test[col].fillna(median_values[col])

print(f"训练集剩余缺失值: {train.isnull().sum().sum()}")
print(f"测试集剩余缺失值: {test.isnull().sum().sum()}")

# ============================================================
# 第三部分：信用特征相关性分析
# ============================================================
print("\n" + "=" * 70)
print("第三部分：信用特征相关性分析")
print("=" * 70)

# 选择数值特征进行相关性分析（排除已编码的分类特征的原始值）
feature_cols = [col for col in numeric_features if col in train.columns]
corr_matrix = train[feature_cols].corr()

# 绘制相关性热力图
fig, ax = plt.subplots(figsize=(16, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title('特征相关性热力图')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: correlation_heatmap.png")

# 找出高度相关的特征对
print(f"\n3.1 高度相关的特征对 (|r| > {CORR_THRESHOLD}):")
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > CORR_THRESHOLD:
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

for col1, col2, corr in high_corr_pairs:
    print(f"  {col1} - {col2}: {corr:.3f}")

# 剔除高度相关的冗余特征
features_to_drop = set()
for col1, col2, corr in high_corr_pairs:
    corr1 = abs(train[col1].corr(train['isDefault']))
    corr2 = abs(train[col2].corr(train['isDefault']))
    if corr1 < corr2:
        features_to_drop.add(col1)
    else:
        features_to_drop.add(col2)

print(f"\n3.2 剔除的冗余特征: {features_to_drop}")

# ============================================================
# 第四部分：特征工程 - WOE/IV值计算
# ============================================================
print("\n" + "=" * 70)
print("第四部分：特征工程 - WOE/IV值计算")
print("=" * 70)

def calc_woe_iv(df, feature, target, bins=10):
    """
    计算单个特征的WOE和IV值（不修改原始数据框）

    参数:
    df: 数据框
    feature: 特征名
    target: 目标变量名
    bins: 分箱数量

    返回:
    grouped: 每个箱的WOE统计
    iv: IV值
    """
    # 使用临时副本，避免修改原始数据
    temp_df = df[[feature, target]].copy()

    # 分箱
    try:
        temp_df['bin'] = pd.qcut(temp_df[feature], q=bins, duplicates='drop')
    except ValueError:
        temp_df['bin'] = pd.cut(temp_df[feature], bins=bins)

    # 计算每个箱的统计
    grouped = temp_df.groupby('bin', observed=False)[target].agg(['count', 'sum'])
    grouped.columns = ['total', 'bad']
    grouped['good'] = grouped['total'] - grouped['bad']

    # 计算比例
    total_bad = grouped['bad'].sum()
    total_good = grouped['good'].sum()

    if total_bad == 0 or total_good == 0:
        return grouped, 0.0

    grouped['bad_pct'] = grouped['bad'] / total_bad
    grouped['good_pct'] = grouped['good'] / total_good

    # 避免除以0
    grouped['bad_pct'] = grouped['bad_pct'].replace(0, 0.0001)
    grouped['good_pct'] = grouped['good_pct'].replace(0, 0.0001)

    # 计算WOE
    grouped['woe'] = np.log(grouped['good_pct'] / grouped['bad_pct'])

    # 计算IV
    grouped['iv'] = (grouped['good_pct'] - grouped['bad_pct']) * grouped['woe']
    iv = grouped['iv'].sum()

    return grouped, iv

# 计算所有特征的IV值
print(f"\n4.1 计算各特征的IV值...")
iv_results = []
features_for_iv = [col for col in feature_cols if col not in features_to_drop]

for feature in features_for_iv:
    try:
        _, iv = calc_woe_iv(train, feature, 'isDefault', bins=WOE_BINS)
        iv_results.append({'特征': feature, 'IV值': iv})
    except Exception as e:
        print(f"  警告: 计算 {feature} 的IV值时出错: {e}")

iv_df = pd.DataFrame(iv_results).sort_values('IV值', ascending=False)
print("\n特征IV值排序:")
print(iv_df.to_string(index=False))

# IV值解读
print("\n4.2 IV值解读:")
print("  IV < 0.02: 特征无预测能力")
print("  0.02 <= IV < 0.1: 特征预测能力弱")
print("  0.1 <= IV < 0.3: 特征预测能力中等")
print("  IV >= 0.3: 特征预测能力强")

# 选择IV值大于阈值的特征
selected_features = iv_df[iv_df['IV值'] >= IV_THRESHOLD]['特征'].tolist()
print(f"\n4.3 选择的特征 (IV >= {IV_THRESHOLD}): {len(selected_features)} 个")

# 可视化IV值
fig, ax = plt.subplots(figsize=(12, 8))
top_iv = iv_df.head(20)
colors = ['#2ecc71' if iv >= 0.1 else '#f39c12' if iv >= IV_THRESHOLD else '#e74c3c'
          for iv in top_iv['IV值']]
ax.barh(top_iv['特征'], top_iv['IV值'], color=colors)
ax.set_xlabel('IV值')
ax.set_title('Top 20 特征IV值')
ax.axvline(x=IV_THRESHOLD, color='red', linestyle='--', label=f'IV={IV_THRESHOLD}')
ax.axvline(x=0.1, color='green', linestyle='--', label='IV=0.1')
ax.legend()
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/feature_iv_values.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: feature_iv_values.png")

# ============================================================
# 第五部分：模型构建 - 逻辑回归
# ============================================================
print("\n" + "=" * 70)
print("第五部分：模型构建 - 逻辑回归")
print("=" * 70)

# 准备特征
final_features = [f for f in selected_features if f in train.columns]
print(f"\n5.1 使用 {len(final_features)} 个特征进行建模")
print(f"   特征列表: {final_features}")

# 准备数据
X = train[final_features].copy()
y = train['isDefault'].copy()

# 处理无穷大值
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# 划分训练集和验证集（使用分层抽样保证正负样本比例一致）
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)
print(f"训练集大小: {X_train.shape}")
print(f"验证集大小: {X_val.shape}")

# 标准化（仅在训练集上fit，避免数据泄露）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 训练逻辑回归模型
print("\n5.2 训练逻辑回归模型...")
lr_model = LogisticRegression(
    C=1.0, max_iter=1000, random_state=RANDOM_SEED, class_weight='balanced'
)
lr_model.fit(X_train_scaled, y_train)

# 交叉验证
cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=CV_FOLDS, scoring='roc_auc')
print(f"5.3 交叉验证AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================
# 第六部分：模型评估
# ============================================================
print("\n" + "=" * 70)
print("第六部分：模型评估")
print("=" * 70)

# 预测
y_pred_proba = lr_model.predict_proba(X_val_scaled)[:, 1]
y_pred = lr_model.predict(X_val_scaled)

# 计算各项指标
auc_score = roc_auc_score(y_val, y_pred_proba)
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

# 计算KS值
fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
ks_value = max(tpr - fpr)

print("\n6.1 模型评估指标:")
print(f"  AUC: {auc_score:.4f}")
print(f"  KS值: {ks_value:.4f}")
print(f"  准确率: {accuracy:.4f}")
print(f"  精确率: {precision:.4f}")
print(f"  召回率: {recall:.4f}")
print(f"  F1分数: {f1:.4f}")

# 混淆矩阵
print("\n6.2 混淆矩阵:")
cm = confusion_matrix(y_val, y_pred)
print(cm)

# 分类报告
print("\n6.3 分类报告:")
print(classification_report(y_val, y_pred, target_names=['正常还款', '违约']))

# 绘制ROC曲线
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC曲线
axes[0].plot(fpr, tpr, 'b-', label=f'AUC = {auc_score:.4f}')
axes[0].plot([0, 1], [0, 1], 'k--')
axes[0].set_xlabel('假正率 (FPR)')
axes[0].set_ylabel('真正率 (TPR)')
axes[0].set_title('ROC曲线')
axes[0].legend()
axes[0].grid(True)

# KS曲线 - 使用统一长度避免维度不匹配
min_len = min(len(thresholds), len(tpr), len(fpr))
axes[1].plot(thresholds[:min_len], tpr[:min_len], 'b-', label='TPR')
axes[1].plot(thresholds[:min_len], fpr[:min_len], 'r-', label='FPR')
axes[1].plot(thresholds[:min_len], tpr[:min_len] - fpr[:min_len], 'g-', label=f'KS = {ks_value:.4f}')
axes[1].set_xlabel('阈值')
axes[1].set_ylabel('比率')
axes[1].set_title('KS曲线')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/roc_ks_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n已保存: roc_ks_curves.png")

# 混淆矩阵可视化
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['正常还款', '违约'], yticklabels=['正常还款', '违约'])
ax.set_xlabel('预测标签')
ax.set_ylabel('真实标签')
ax.set_title('混淆矩阵')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: confusion_matrix.png")

# ============================================================
# 第七部分：信用评分卡生成
# ============================================================
print("\n" + "=" * 70)
print("第七部分：信用评分卡生成")
print("=" * 70)

def generate_scorecard(model, features, factor, offset):
    """
    生成标准化信用评分卡

    参数:
    model: 训练好的逻辑回归模型
    features: 特征列表
    factor: 评分卡因子
    offset: 评分卡偏移量

    返回:
    scorecard_df: 评分卡数据框
    """
    # 获取模型系数
    coefficients = model.coef_[0]

    # 计算每个特征的分数贡献
    scorecard_data = []
    for i, feature in enumerate(features):
        coef = coefficients[i]
        feature_score = -coef * factor
        scorecard_data.append({
            '特征': feature,
            '系数': coef,
            '分数贡献': feature_score,
            '权重方向': '正向(降低违约风险)' if coef < 0 else '负向(增加违约风险)'
        })

    scorecard_df = pd.DataFrame(scorecard_data)
    scorecard_df = scorecard_df.sort_values('分数贡献', key=abs, ascending=False)
    return scorecard_df

# 计算评分卡参数
factor = PDO / np.log(2)
offset = BASE_SCORE - factor * np.log(BASE_ODDS)

print(f"7.1 评分卡参数:")
print(f"  基础分数: {BASE_SCORE}")
print(f"  PDO (分数翻倍点): {PDO}")
print(f"  Factor: {factor:.2f}")
print(f"  Offset: {offset:.2f}")

# 生成评分卡
scorecard_df = generate_scorecard(lr_model, final_features, factor, offset)

print("\n7.2 信用评分卡特征权重:")
print(scorecard_df.to_string(index=False))

# 保存评分卡
scorecard_df.to_csv(f'{OUTPUT_DIR}/credit_scorecard.csv', index=False, encoding='utf-8-sig')
print("\n已保存: credit_scorecard.csv")

# 可视化特征分数贡献
fig, ax = plt.subplots(figsize=(12, 8))
top_scorecard = scorecard_df.head(15)
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_scorecard['分数贡献']]
ax.barh(top_scorecard['特征'], top_scorecard['分数贡献'], color=colors)
ax.set_xlabel('分数贡献')
ax.set_title('Top 15 特征信用分数贡献')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/scorecard_feature_contributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存: scorecard_feature_contributions.png")

# ============================================================
# 第八部分：特征重要性分析
# ============================================================
print("\n" + "=" * 70)
print("第八部分：特征重要性分析")
print("=" * 70)

# 获取特征重要性（基于模型系数的绝对值）
feature_importance = pd.DataFrame({
    '特征': final_features,
    '重要性': np.abs(lr_model.coef_[0]),
    '系数': lr_model.coef_[0]
}).sort_values('重要性', ascending=False)

print("\n8.1 特征重要性排序:")
print(feature_importance.to_string(index=False))

# 可视化特征重要性
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# 柱状图
top_features = feature_importance.head(15)
axes[0].barh(top_features['特征'], top_features['重要性'], color='#3498db')
axes[0].set_xlabel('重要性 (|系数|)')
axes[0].set_title('Top 15 特征重要性')
axes[0].invert_yaxis()

# 系数方向图
coef_df = feature_importance.sort_values('系数')
colors = ['#2ecc71' if x < 0 else '#e74c3c' for x in coef_df['系数']]
axes[1].barh(coef_df['特征'], coef_df['系数'], color=colors)
axes[1].set_xlabel('系数值')
axes[1].set_title('特征系数 (绿色=降低违约风险, 红色=增加违约风险)')
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n已保存: feature_importance.png")

# ============================================================
# 第九部分：用户信用分数计算与建议
# ============================================================
print("\n" + "=" * 70)
print("第九部分：用户信用分数计算与建议")
print("=" * 70)

def calculate_credit_score(model, scaler, features, factor, offset, df, fill_values):
    """
    计算用户信用分数

    公式: Score = Offset - Factor * ln(p/(1-p))
    其中: odds = (1-p)/p, p为违约概率

    参数:
    model: 训练好的逻辑回归模型
    scaler: 标准化器
    features: 特征列表
    factor: 评分卡因子
    offset: 评分卡偏移量
    df: 待评分数据
    fill_values: 用于填充缺失值的字典（来自训练集）

    返回:
    scores: 信用分数数组
    proba: 违约概率数组
    """
    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # 使用训练集的中位数填充（避免数据泄露）
    for col in features:
        if col in fill_values:
            X[col] = X[col].fillna(fill_values[col])

    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:, 1]

    # 避免log(0)和log(inf)
    proba = np.clip(proba, 0.001, 0.999)

    # 计算分数: Score = Offset - Factor * ln(odds)
    odds = (1 - proba) / proba
    scores = offset - factor * np.log(odds)

    # 限制分数范围
    scores = np.clip(scores, SCORE_MIN, SCORE_MAX)

    return scores, proba

# 保存训练集的中位数用于后续填充
fill_values = X.median().to_dict()

# 计算训练集用户的信用分数
print("\n9.1 计算用户信用分数...")
train_scores, train_proba = calculate_credit_score(
    lr_model, scaler, final_features, factor, offset, train, fill_values
)

train['credit_score'] = train_scores
train['default_probability'] = train_proba

# 统计信用分数分布
print("\n9.2 信用分数分布统计:")
print(f"  平均分: {train_scores.mean():.0f}")
print(f"  中位数: {np.median(train_scores):.0f}")
print(f"  最低分: {train_scores.min():.0f}")
print(f"  最高分: {train_scores.max():.0f}")

# 不同信用等级的违约率
print("\n9.3 不同信用等级的违约率:")
train['credit_grade'] = pd.cut(
    train['credit_score'],
    bins=DEFAULT_BINS,
    labels=DEFAULT_LABELS
)

grade_stats = train.groupby('credit_grade', observed=False).agg({
    'isDefault': ['count', 'sum', 'mean']
}).round(4)
grade_stats.columns = ['用户数', '违约数', '违约率']
print(grade_stats)

# 可视化信用分数分布
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 分数分布直方图
axes[0].hist(train_scores, bins=50, color='#3498db', edgecolor='white')
axes[0].set_xlabel('信用分数')
axes[0].set_ylabel('用户数量')
axes[0].set_title('信用分数分布')
axes[0].axvline(x=train_scores.mean(), color='red', linestyle='--',
                label=f'均值: {train_scores.mean():.0f}')
axes[0].legend()

# 不同等级的违约率
grade_stats['违约率'].plot(
    kind='bar', ax=axes[1],
    color=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60']
)
axes[1].set_xlabel('信用等级')
axes[1].set_ylabel('违约率')
axes[1].set_title('不同信用等级违约率')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

# 分数与违约概率关系（使用固定随机种子采样）
sample_idx = np.random.RandomState(RANDOM_SEED).choice(
    len(train_scores), 10000, replace=False
)
scatter = axes[2].scatter(
    train_scores[sample_idx], train_proba[sample_idx],
    alpha=0.1, s=5, c=train['isDefault'].iloc[sample_idx], cmap='RdYlGn_r'
)
axes[2].set_xlabel('信用分数')
axes[2].set_ylabel('违约概率')
axes[2].set_title('信用分数 vs 违约概率')
plt.colorbar(scatter, ax=axes[2], label='是否违约')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/credit_score_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n已保存: credit_score_analysis.png")

# ============================================================
# 第十部分：用户信用提升建议
# ============================================================
print("\n" + "=" * 70)
print("第十部分：用户信用提升建议")
print("=" * 70)

# 分析特征与信用分数的关系
print("\n10.1 核心影响因素分析:")

# 获取正向和负向影响最大的特征
positive_features = scorecard_df[scorecard_df['分数贡献'] > 0].head(5)
negative_features = scorecard_df[scorecard_df['分数贡献'] < 0].head(5)

print("\n正向影响因素 (提高信用分数):")
for _, row in positive_features.iterrows():
    print(f"  - {row['特征']}: 贡献 +{row['分数贡献']:.2f} 分")

print("\n负向影响因素 (降低信用分数):")
for _, row in negative_features.iterrows():
    print(f"  - {row['特征']}: 贡献 {row['分数贡献']:.2f} 分")

# 生成信用提升建议
print("\n10.2 信用提升建议:")
print("""
基于模型分析，我们提出以下信用提升建议：

【核心建议】

1. **维护良好的还款记录**
   - 按时还款是提升信用分数的最关键因素
   - 避免逾期，特别是连续逾期

2. **控制负债水平**
   - 保持较低的负债收入比(DTI)
   - 合理使用信贷额度，避免过度借贷

3. **建立稳定的信用历史**
   - 保持较长的信用历史
   - 避免频繁申请新信贷

4. **多元化信用组合**
   - 适度持有不同类型的信贷产品
   - 展示良好的多元信用管理能力

5. **保持稳定的财务状况**
   - 维持稳定的工作和收入来源
   - 合理规划个人财务

【具体行动建议】

- 定期查看个人信用报告，及时发现并纠正错误
- 设置自动还款，避免因遗忘导致逾期
- 在申请大额贷款前，提前6个月优化信用状况
- 避免在短时间内多次申请信贷
- 保持信用卡使用率在30%以下
""")

# ============================================================
# 第十一部分：测试集预测与结果保存
# ============================================================
print("\n" + "=" * 70)
print("第十一部分：测试集预测与结果保存")
print("=" * 70)

# 计算测试集用户的信用分数
print("\n11.1 计算测试集用户信用分数...")
test_scores, test_proba = calculate_credit_score(
    lr_model, scaler, final_features, factor, offset, test, fill_values
)

test['credit_score'] = test_scores
test['default_probability'] = test_proba
test['credit_grade'] = pd.cut(
    test['credit_score'],
    bins=DEFAULT_BINS,
    labels=DEFAULT_LABELS
)

# 保存预测结果
prediction_result = test[['id', 'credit_score', 'default_probability', 'credit_grade']].copy()
prediction_result.to_csv(f'{OUTPUT_DIR}/test_predictions.csv', index=False, encoding='utf-8-sig')
print("已保存: test_predictions.csv")

# 测试集统计
print("\n11.2 测试集信用分数分布:")
print(f"  平均分: {test_scores.mean():.0f}")
print(f"  中位数: {np.median(test_scores):.0f}")
print(f"  最低分: {test_scores.min():.0f}")
print(f"  最高分: {test_scores.max():.0f}")

print("\n11.3 测试集信用等级分布:")
print(test['credit_grade'].value_counts().sort_index())

# ============================================================
# 第十二部分：保存完整模型结果
# ============================================================
print("\n" + "=" * 70)
print("第十二部分：保存完整模型结果")
print("=" * 70)

# 保存模型评估结果
model_results = {
    '模型类型': '逻辑回归',
    '特征数量': len(final_features),
    'AUC': auc_score,
    'KS值': ks_value,
    '准确率': accuracy,
    '精确率': precision,
    '召回率': recall,
    'F1分数': f1,
    '交叉验证AUC均值': cv_scores.mean(),
    '交叉验证AUC标准差': cv_scores.std()
}

results_df = pd.DataFrame([model_results])
results_df.to_csv(f'{OUTPUT_DIR}/model_evaluation_results.csv', index=False, encoding='utf-8-sig')
print("已保存: model_evaluation_results.csv")

# 保存特征重要性
feature_importance.to_csv(f'{OUTPUT_DIR}/feature_importance.csv', index=False, encoding='utf-8-sig')
print("已保存: feature_importance.csv")

# 保存IV值结果
iv_df.to_csv(f'{OUTPUT_DIR}/feature_iv_values.csv', index=False, encoding='utf-8-sig')
print("已保存: feature_iv_values.csv")

# 保存信用等级统计
grade_stats.to_csv(f'{OUTPUT_DIR}/credit_grade_statistics.csv', encoding='utf-8-sig')
print("已保存: credit_grade_statistics.csv")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 70)
print("实验总结")
print("=" * 70)

print(f"""
本次银行信用评分模型实验完成，主要成果如下：

1. **数据处理**:
   - 训练集: {train_df.shape[0]} 条记录, {train_df.shape[1]} 个特征
   - 测试集: {test_df.shape[0]} 条记录
   - 处理了缺失值、分类特征编码、日期特征提取等

2. **特征工程**:
   - 计算了 {len(iv_df)} 个特征的IV值
   - 选择了 {len(final_features)} 个IV值>={IV_THRESHOLD}的有效特征
   - 剔除了 {len(features_to_drop)} 个冗余特征

3. **模型性能**:
   - AUC: {auc_score:.4f}
   - KS值: {ks_value:.4f}
   - 准确率: {accuracy:.4f}
   - 精确率: {precision:.4f}

4. **信用评分卡**:
   - 基础分数: {BASE_SCORE}分
   - PDO: {PDO}分
   - 评分范围: {SCORE_MIN}-{SCORE_MAX}分

5. **输出文件**:
   - credit_scorecard.csv: 信用评分卡
   - feature_importance.png: 特征重要性图
   - roc_ks_curves.png: ROC和KS曲线
   - test_predictions.csv: 测试集预测结果
   - 其他分析图表和结果文件
""")

print("\n实验完成！所有结果已保存到 outputs/homework11/ 目录。")
