"""Configuration for Homework 3 Two-Factor Model experiment."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "homework3"
OUTPUT_DIR = ROOT / "outputs" / "homework3"

# --- Time periods ---
TRAIN_START = "2018-01-01"
TRAIN_END = "2022-12-31"
BACKTEST_START = "2023-01-01"
BACKTEST_END = "2025-12-31"

# --- Risk-free rate ---
RISK_FREE_ANNUAL = 0.015  # 一年期国债收益率 1.5%
TRADING_DAYS = 250
RISK_FREE_DAILY = (1 + RISK_FREE_ANNUAL) ** (1 / TRADING_DAYS) - 1

# --- Backtest parameters ---
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0003  # 手续费 0.03%/笔
SLIPPAGE_RATE = 0.001     # 滑点 0.1%/笔

# --- PEG strategy thresholds ---
PEG_BUY_THRESHOLD = 0.8
PEG_SELL_THRESHOLD = 1.5

# --- Market index: 上证指数 ---
MARKET_NAME = "上证指数"
MARKET_CODE = "sh.000001"

# --- Stock pool ---
STOCKS = {
    "贵州茅台": "sh.600519",
    "中国石油": "sh.601857",
    "五粮液": "sz.000858",
    "泸州老窖": "sz.000568",
    "招商银行": "sh.600036",
    "美的集团": "sz.000333",
}

# --- Industry mapping ---
INDUSTRY = {
    "贵州茅台": "白酒",
    "五粮液": "白酒",
    "泸州老窖": "白酒",
    "中国石油": "能源",
    "招商银行": "金融",
    "美的集团": "家电",
}
