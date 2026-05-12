"""作业1 配置。"""

STOCK_CODES = [
    "sh.600519",  # 贵州茅台
    "sz.000858",  # 五粮液
    "sz.000568",  # 泸州老窖
    "sh.600809",  # 山西汾酒
    "sh.600236",  # 洋河股份
]

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"
WINDOW = 10

INITIAL_CAPITAL = 1_000_000
COMMISSION = 0.0003
BUY_THRESHOLD = 0.005
SELL_THRESHOLD = 0.005
