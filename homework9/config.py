"""Configuration for Homework 9 model maintenance and risk control."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "homework9"
ORIGINAL_DIR = DATA_DIR / "original"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = ROOT / "outputs" / "homework9"
REPORT_DIR = ROOT / "report"

STUDENT_NAME = "丁致宇"
STUDENT_ID = "202331060205"

DATA_START = "2015-01-01"
INITIAL_TRAIN_END = "2018-01-01"
BACKTEST_END = "2024-12-31"

ASSETS = {
    "贵州茅台": "sh.600519",
    "泸州老窖": "sz.000568",
}
MAOTAI = "贵州茅台"
LAOJIAO = "泸州老窖"

ADF_P_THRESHOLD = 0.05
THRESHOLD_MULTIPLIER = 1.5
ROLLING_LOOKBACK_YEARS = 3
ROLLING_RESET_MONTHS = 6
PAIR_LEG_WEIGHT = 0.5
