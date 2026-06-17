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

# BERT model stored in parent directory to keep the repo lean
MODEL_DIR = ROOT.parent / "models" / "yiyanghkust-finbert-tone-chinese"

START_DATE = "2014-10-01"
END_DATE = "2015-10-31"
TRADING_DAYS_PER_WEEK = 5
ROLLING_BASELINE_WEEKS = 4
BERT_BATCH_SIZE = 256

# Experiment 3: LightGBM parameters
MAX_LAG = 5
TRAIN_RATIO = 0.8
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "n_estimators": 100,
    "verbose": -1,
    "n_jobs": -1,
}

STUDENT_ID = "202331060205"
STUDENT_NAME = "丁致宇"
