"""Configuration for Homework 2 CAPM experiment."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "homework2"
OUTPUT_DIR = ROOT / "outputs" / "homework2"

TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
BACKTEST_START = "2023-01-01"
BACKTEST_END = "2024-12-31"
EXTENDED_BACKTEST_END = "2026-05-19"
EXTENDED_BACKTEST_PERIODS = [
    ("原回测期：2023-2024", "2024-12-31"),
    ("延长至2025年底", "2025-12-31"),
    ("延长至最新可得数据", None),
]

INITIAL_CAPITAL = 1_000_000.0
RISK_FREE_ANNUAL = 0.03
TRADING_DAYS = 250
RISK_FREE_DAILY = RISK_FREE_ANNUAL / TRADING_DAYS
ALPHA_SIGNIFICANCE_LEVEL = 0.10

MARKET_NAME = "沪深300"
MARKET_CODE = "sh.000300"

STOCKS = {
    "贵州茅台": "sh.600519",
    "中国石油": "sh.601857",
    "五粮液": "sz.000858",
    "泸州老窖": "sz.000568",
    "招商银行": "sh.600036",
    "美的集团": "sz.000333",
}
