"""Shared paths and experiment parameters."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "experiment1"
DATA_DIR = ROOT / "data" / "experiment1"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = ROOT / "outputs" / "experiment1"
REPORT_DIR = ROOT / "report"

NEWS_FILE = RAW_DIR / "新闻数据.xls"
HS300_FILE = RAW_DIR / "沪深300日价格指数.xls"
DB_FILE = OUTPUT_DIR / "experiment1.db"

START_DATE = "2014-10-01"
END_DATE = "2015-10-31"
TRADING_DAYS_PER_WEEK = 5
ROLLING_BASELINE_WEEKS = 4
MAX_GRANGER_LAG = 5

STUDENT_ID = "202331060205"
STUDENT_NAME = "丁致宇"

