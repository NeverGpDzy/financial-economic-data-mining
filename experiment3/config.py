"""Shared paths and parameters for Experiment 3."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "experiment3"
DATA_DIR = ROOT / "data" / "experiment3"
OUTPUT_DIR = ROOT / "outputs" / "experiment3"
REPORT_DIR = ROOT / "report"

INPUT_HERD_INDEX = ROOT / "outputs" / "experiment2" / "weekly_herd_index.csv"
INPUT_HS300_DAILY = ROOT / "data" / "experiment2" / "raw" / "沪深300日价格指数.xls"
DB_FILE = OUTPUT_DIR / "experiment3.db"

STUDENT_ID = "202331060205"
STUDENT_NAME = "丁致宇"

NATURAL_WEEK_FREQ = "W-SUN"
MAX_LAG = 5
ROLLING_WINDOWS = (3, 5)
TRAIN_RATIO = 0.8
INTERNAL_VALID_RATIO = 0.2
RANDOM_STATE = 42

BASE_LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "boosting_type": "gbdt",
    "n_estimators": 160,
    "learning_rate": 0.05,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 1,
    "min_data_in_bin": 1,
    "reg_alpha": 0.05,
    "reg_lambda": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

PARAM_GRID = [
    {"n_estimators": 80, "learning_rate": 0.05, "num_leaves": 5, "max_depth": 2, "reg_lambda": 0.1},
    {"n_estimators": 120, "learning_rate": 0.05, "num_leaves": 7, "max_depth": 3, "reg_lambda": 0.1},
    {"n_estimators": 160, "learning_rate": 0.03, "num_leaves": 7, "max_depth": 3, "reg_lambda": 0.3},
    {"n_estimators": 120, "learning_rate": 0.08, "num_leaves": 9, "max_depth": 3, "reg_lambda": 0.2},
]

