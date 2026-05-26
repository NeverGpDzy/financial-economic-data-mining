"""Configuration for Homework 4: Multi-Factor Quantitative Stock Selection."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "homework4"
OUTPUT_DIR = ROOT / "outputs" / "homework4"

# --- Time periods ---
TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

# --- Market index ---
MARKET_CODE = "sh.000001"
MARKET_NAME = "上证指数"

# --- Risk-free rate (monthly) ---
RF_MONTHLY = 0.0015  # 0.15%/month

# --- Backtest parameters ---
INIT_CAPITAL = 1_000_000.0
FEE = 0.003  # one-way transaction cost
TOP_N = 3
MAX_SINGLE_WEIGHT = 0.40

# --- Factor list ---
FACTORS = ["SMB", "PE_inv", "Quality"]
FACTOR_NAMES = {"SMB": "规模因子", "PE_inv": "价值因子(PE倒数)", "Quality": "质量因子(npMargin×CFOToNP×YOYNI)"}

# --- IC / IR thresholds ---
IC_MEAN_THRESHOLD = 0.02  # IC > 0.02: predictive
IC_EXCELLENT_THRESHOLD = 0.05  # IC > 0.05: excellent
IR_THRESHOLD = 0.1  # IR > 0.1

# --- Significance level ---
P_VALUE_THRESHOLD = 0.05
