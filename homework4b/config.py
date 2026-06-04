"""
作业4B 配置文件
中频短线量化全流程：因子挖掘→IC/共线质检→机器学习赋权→1~2日持仓选股回测
"""
import os

# ==================== 路径配置 ====================
# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Parquet数据根目录（教师预下载的配套数据）
PARQUET_ROOT = os.path.join(
    PROJECT_ROOT, '..', '作业四-B', '作业4B 配套数据 parquet v1.1', 'BAKdata'
)

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'homework4b')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据输出目录
DATA_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'homework4b')
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

# ==================== 时间配置 ====================
# 训练集时间范围（因子检验、模型训练）
TRAIN_START = '2020-01-01'
TRAIN_END = '2023-12-31'

# 回测集时间范围（样本外验证）
TEST_START = '2024-01-01'
TEST_END = '2025-12-31'

# ==================== 因子配置 ====================
# 无风险利率（日频）
RISK_FREE_RATE_DAILY = 0.0001  # 日0.01%

# 因子名称列表（4大标准因子 + 4个扩展因子）
FACTOR_NAMES = ['Reversal', 'Liquidity', 'MoneyFlow', 'Value',
                'Momentum', 'Volatility', 'Turnover', 'VolumeChange']

# ==================== 因子质检配置 ====================
# IC筛选阈值
IC_THRESHOLD_EFFECTIVE = 0.02    # |IC| >= 0.02 判定为有效
IC_THRESHOLD_MARGINAL = 0.01    # 0.01 <= |IC| < 0.02 为边缘因子

# VIF共线性阈值
VIF_THRESHOLD = 5.0  # VIF > 5 判定高共线

# OLS显著性水平
OLS_P_THRESHOLD = 0.05  # p < 0.05 判定为线性有效

# ==================== LGBM模型配置 ====================
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}

LGBM_NUM_BOOST_ROUND = 200
LGBM_EARLY_STOPPING = 20
LGBM_K_FOLD = 5  # 时间序列交叉验证折数

# ==================== 回测配置 ====================
INITIAL_CAPITAL = 1_000_000  # 初始资金100万元
COMMISSION_RATE = 0.001      # 单边交易成本0.1%
TOP_N_STOCKS = 5             # 选股Top5
MAX_POSITION_WEIGHT = 0.20   # 单票最大仓位20%
HOLDING_DAYS = 1             # 持仓天数（每日调仓，平均持仓1~2日）

# ==================== 绘图配置 ====================
FIGURE_DPI = 150
FIGURE_SIZE = (14, 8)
