"""Data alignment, feature engineering, LightGBM, and SHAP analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config


REQUIRED_HERD_COLUMNS = {
    "trade_date",
    "P_t",
    "H1t",
    "H2t",
    "H3t",
}


@dataclass(frozen=True)
class FeatureSpec:
    target: str
    feature_cols: list[str]
    h3_cols: list[str]
    sentiment_cols: list[str]
    time_cols: list[str]


def load_herd_index(path: Path = config.INPUT_HERD_INDEX) -> pd.DataFrame:
    """Load Experiment 2 herd-effect indicators."""
    if not path.exists():
        raise FileNotFoundError(f"缺少实验二羊群效应指标：{path}。请先运行 `python -m experiment2.main`。")
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = REQUIRED_HERD_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"羊群效应指标表缺少字段：{sorted(missing)}")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    for col in ["P_t", "H1t", "H2t", "H3t"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return (
        df.dropna(subset=["trade_date", "P_t", "H1t", "H2t", "H3t"])
        .sort_values("trade_date")
        .reset_index(drop=True)
    )


def load_hs300_daily(path: Path = config.INPUT_HS300_DAILY) -> pd.DataFrame:
    """Load the teacher-provided HS300 daily price file.

    The file extension is .xls, but the actual content is GBK tab-separated
    text. The first six useful columns are date, open, high, low, close, volume.
    """
    if not path.exists():
        raise FileNotFoundError(f"缺少沪深300日度价格数据：{path}")
    raw = pd.read_csv(path, sep="\t", encoding="gbk", skiprows=2)
    raw.columns = [str(c).strip() for c in raw.columns]
    keep = raw.iloc[:, :6].copy()
    keep.columns = ["date", "open", "high", "low", "close", "volume"]
    keep["date"] = pd.to_datetime(keep["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        keep[col] = pd.to_numeric(keep[col], errors="coerce")
    keep = keep.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return keep


def _natural_week_parts(date_series: pd.Series) -> pd.DataFrame:
    period = pd.to_datetime(date_series).dt.to_period(config.NATURAL_WEEK_FREQ)
    return pd.DataFrame(
        {
            "natural_week": period.astype(str),
            "natural_week_start": period.dt.start_time.dt.normalize(),
            "natural_week_end": period.dt.end_time.dt.normalize(),
        },
        index=date_series.index,
    )


def build_natural_week_herd(herd: pd.DataFrame) -> pd.DataFrame:
    """Convert Experiment 2's weekly indicator dates into natural-week labels."""
    df = herd.copy()
    week_parts = _natural_week_parts(df["trade_date"])
    df = pd.concat([df, week_parts], axis=1)
    agg = (
        df.groupby(["natural_week", "natural_week_start", "natural_week_end"], as_index=False)
        .agg(
            herd_trade_date=("trade_date", "max"),
            P_t=("P_t", "last"),
            H1t=("H1t", "last"),
            H2t=("H2t", "last"),
            H3t=("H3t", "last"),
            H3t_mean_in_week=("H3t", "mean"),
            herd_rows_in_week=("H3t", "size"),
        )
        .sort_values("natural_week_end")
        .reset_index(drop=True)
    )
    return agg


def build_hs300_weekly_return(daily: pd.DataFrame) -> pd.DataFrame:
    """Build natural-week HS300 returns from the last trading close of each week."""
    df = daily.copy()
    week_parts = _natural_week_parts(df["date"])
    df = pd.concat([df, week_parts], axis=1)
    weekly = (
        df.groupby(["natural_week", "natural_week_start", "natural_week_end"], as_index=False)
        .agg(
            price_week_close_date=("date", "max"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            trading_days=("date", "size"),
        )
        .sort_values("natural_week_end")
        .reset_index(drop=True)
    )
    weekly["Ret_t"] = weekly["close"].pct_change()
    return weekly


def align_weekly_data(herd_weekly: pd.DataFrame, hs300_weekly: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Inner-join natural-week herd indicators and market returns."""
    aligned = herd_weekly.merge(
        hs300_weekly,
        on=["natural_week", "natural_week_start", "natural_week_end"],
        how="inner",
        validate="one_to_one",
    )
    aligned = (
        aligned.dropna(subset=["P_t", "H3t", "Ret_t"])
        .sort_values("natural_week_end")
        .reset_index(drop=True)
    )
    audit = {
        "herd_rows": int(len(herd_weekly)),
        "hs300_weekly_rows": int(len(hs300_weekly)),
        "aligned_rows": int(len(aligned)),
        "aligned_start": str(aligned["natural_week_start"].min().date()) if not aligned.empty else "",
        "aligned_end": str(aligned["natural_week_end"].max().date()) if not aligned.empty else "",
        "missing_cells_after_align": int(aligned[["P_t", "H3t", "Ret_t"]].isna().sum().sum()),
        "max_herd_rows_per_natural_week": int(aligned["herd_rows_in_week"].max()) if not aligned.empty else 0,
    }
    return aligned, audit


def build_features(aligned: pd.DataFrame) -> tuple[pd.DataFrame, FeatureSpec]:
    """Build leak-free lag, rolling, and calendar dummy features."""
    df = aligned.copy().sort_values("natural_week_end").reset_index(drop=True)
    h3_cols: list[str] = []
    sentiment_cols: list[str] = []
    for lag in range(1, config.MAX_LAG + 1):
        col = f"H3_lag{lag}"
        df[col] = df["H3t"].shift(lag)
        h3_cols.append(col)
    for lag in range(1, 4):
        col = f"P_t_lag{lag}"
        df[col] = df["P_t"].shift(lag)
        sentiment_cols.append(col)

    h3_past = df["H3t"].shift(1)
    p_past = df["P_t"].shift(1)
    for window in config.ROLLING_WINDOWS:
        mean_col = f"H3_roll_mean_{window}"
        std_col = f"H3_roll_std_{window}"
        df[mean_col] = h3_past.rolling(window, min_periods=window).mean()
        df[std_col] = h3_past.rolling(window, min_periods=window).std(ddof=0)
        h3_cols.extend([mean_col, std_col])

    df["P_t_roll_mean_3"] = p_past.rolling(3, min_periods=3).mean()
    sentiment_cols.append("P_t_roll_mean_3")

    date = pd.to_datetime(df["price_week_close_date"])
    month_dummies = pd.get_dummies(date.dt.month, prefix="month", dtype=int)
    quarter_dummies = pd.get_dummies(date.dt.quarter, prefix="quarter", dtype=int)
    df = pd.concat([df, month_dummies, quarter_dummies], axis=1)
    time_cols = list(month_dummies.columns) + list(quarter_dummies.columns)

    feature_cols = h3_cols + sentiment_cols + time_cols
    spec = FeatureSpec(target="Ret_t", feature_cols=feature_cols, h3_cols=h3_cols, sentiment_cols=sentiment_cols, time_cols=time_cols)
    return df, spec


def build_backward_features(aligned: pd.DataFrame) -> tuple[pd.DataFrame, FeatureSpec]:
    """Build reverse-direction features: lagged HS300 returns predict H3."""
    df = aligned.copy().sort_values("natural_week_end").reset_index(drop=True)
    ret_cols: list[str] = []
    for lag in range(1, config.MAX_LAG + 1):
        col = f"Ret_lag{lag}"
        df[col] = df["Ret_t"].shift(lag)
        ret_cols.append(col)
    ret_past = df["Ret_t"].shift(1)
    for window in config.ROLLING_WINDOWS:
        mean_col = f"Ret_roll_mean_{window}"
        std_col = f"Ret_roll_std_{window}"
        df[mean_col] = ret_past.rolling(window, min_periods=window).mean()
        df[std_col] = ret_past.rolling(window, min_periods=window).std(ddof=0)
        ret_cols.extend([mean_col, std_col])

    date = pd.to_datetime(df["price_week_close_date"])
    month_dummies = pd.get_dummies(date.dt.month, prefix="month", dtype=int)
    quarter_dummies = pd.get_dummies(date.dt.quarter, prefix="quarter", dtype=int)
    df = pd.concat([df, month_dummies, quarter_dummies], axis=1)
    time_cols = list(month_dummies.columns) + list(quarter_dummies.columns)

    feature_cols = ret_cols + time_cols
    spec = FeatureSpec(target="H3t", feature_cols=feature_cols, h3_cols=[], sentiment_cols=[], time_cols=time_cols)
    return df, spec


def valid_modeling_rows(featured: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    return featured.dropna(subset=spec.feature_cols + [spec.target]).reset_index(drop=True)


def temporal_train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n < 12:
        raise ValueError(f"可建模样本过少：{n} 行。")
    split = int(math.floor(n * config.TRAIN_RATIO))
    split = min(max(split, 6), n - 3)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def _params_from_grid(overrides: dict) -> dict:
    params = config.BASE_LGBM_PARAMS.copy()
    params.update(overrides)
    return params


def tune_lgbm_params(train_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> tuple[dict, pd.DataFrame]:
    """Tune only inside the training window, keeping the final test set untouched."""
    n_train = len(train_df)
    valid_size = max(3, int(math.ceil(n_train * config.INTERNAL_VALID_RATIO)))
    if n_train - valid_size < 6:
        params = config.BASE_LGBM_PARAMS.copy()
        result = pd.DataFrame([{**params, "validation_mae": np.nan, "selected": True}])
        return params, result

    inner_train = train_df.iloc[: n_train - valid_size]
    valid = train_df.iloc[n_train - valid_size :]
    rows = []
    best_params = config.BASE_LGBM_PARAMS.copy()
    best_mae = float("inf")

    for overrides in config.PARAM_GRID:
        params = _params_from_grid(overrides)
        model = lgb.LGBMRegressor(**params)
        model.fit(inner_train[feature_cols], inner_train[target_col])
        pred = model.predict(valid[feature_cols])
        mae = mean_absolute_error(valid[target_col], pred)
        rows.append({**overrides, "validation_mae": float(mae)})
        if mae < best_mae:
            best_mae = mae
            best_params = params

    tuning = pd.DataFrame(rows)
    tuning["selected"] = tuning["validation_mae"].eq(tuning["validation_mae"].min())
    return best_params, tuning


def evaluate_predictions(y_true, y_pred, direction_baseline=None) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ic, _ = spearmanr(y_true, y_pred)
    if direction_baseline is None:
        direction_acc = np.mean(np.sign(y_true) == np.sign(y_pred))
    else:
        baseline = np.asarray(direction_baseline, dtype=float)
        direction_acc = np.mean(np.sign(y_true - baseline) == np.sign(y_pred - baseline))
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "ic": float(ic) if not np.isnan(ic) else 0.0,
        "direction_acc": float(direction_acc),
        "n_test": int(len(y_true)),
    }


def train_lgbm_model(train_df: pd.DataFrame, test_df: pd.DataFrame, spec: FeatureSpec, direction_baseline=None) -> tuple[lgb.LGBMRegressor, np.ndarray, dict, pd.DataFrame]:
    params, tuning = tune_lgbm_params(train_df, spec.feature_cols, spec.target)
    model = lgb.LGBMRegressor(**params)
    model.fit(train_df[spec.feature_cols], train_df[spec.target])
    pred = model.predict(test_df[spec.feature_cols])
    metrics = evaluate_predictions(test_df[spec.target], pred, direction_baseline=direction_baseline)
    metrics["n_train"] = int(len(train_df))
    metrics["best_params"] = {k: params[k] for k in ["n_estimators", "learning_rate", "num_leaves", "max_depth", "reg_lambda"] if k in params}
    return model, pred, metrics, tuning


def feature_importance_gain(model: lgb.LGBMRegressor, feature_cols: list[str]) -> pd.DataFrame:
    gain = model.booster_.feature_importance(importance_type="gain")
    split = model.booster_.feature_importance(importance_type="split")
    df = pd.DataFrame({"feature": feature_cols, "gain": gain, "split": split})
    total = df["gain"].sum()
    df["gain_share"] = np.where(total > 0, df["gain"] / total, 0.0)
    return df.sort_values(["gain", "split"], ascending=False).reset_index(drop=True)


def compute_shap_values(model: lgb.LGBMRegressor, X_test: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_values = np.asarray(shap_values)
    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    total = importance["mean_abs_shap"].sum()
    importance["shap_share"] = np.where(total > 0, importance["mean_abs_shap"] / total, 0.0)
    return shap_values, importance


def best_lag_from_importance(importance: pd.DataFrame, prefix: str) -> str:
    lagged = importance[importance["feature"].str.startswith(prefix)].copy()
    if lagged.empty:
        return ""
    value_col = "gain" if "gain" in lagged.columns else "mean_abs_shap"
    lagged = lagged.sort_values(value_col, ascending=False)
    return str(lagged.iloc[0]["feature"])


def build_quality_checks(aligned: pd.DataFrame, featured_valid: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    checks = [
        {
            "check": "自然周对齐后无关键缺失",
            "result": bool(not aligned[["natural_week_end", "P_t", "H3t", "Ret_t"]].isna().any().any()),
            "detail": "H3、P_t、Ret_t 均完整。",
        },
        {
            "check": "自然周日期单调递增",
            "result": bool(pd.to_datetime(aligned["natural_week_end"]).is_monotonic_increasing),
            "detail": "所有样本按自然周结束日升序排列。",
        },
        {
            "check": "特征工程后无缺失",
            "result": bool(not featured_valid[spec.feature_cols + [spec.target]].isna().any().any()),
            "detail": "滞后与滚动特征首部缺失已剔除。",
        },
        {
            "check": "未使用同期H3作为正向特征",
            "result": bool("H3t" not in spec.feature_cols and all(c.startswith("H3_lag") or c.startswith("H3_roll") or not c.startswith("H3") for c in spec.feature_cols)),
            "detail": "正向预测只使用 H3 的历史滞后和历史滚动统计。",
        },
        {
            "check": "样本外测试集存在",
            "result": bool(len(featured_valid) >= 12),
            "detail": f"可建模样本 {len(featured_valid)} 行，按时间顺序切分训练和测试。",
        },
    ]
    return pd.DataFrame(checks)


def residual_diagnostics(result_df: pd.DataFrame) -> pd.DataFrame:
    resid = result_df["residual"].astype(float)
    return pd.DataFrame(
        [
            {"metric": "residual_mean", "value": float(resid.mean())},
            {"metric": "residual_std", "value": float(resid.std(ddof=0))},
            {"metric": "residual_min", "value": float(resid.min())},
            {"metric": "residual_max", "value": float(resid.max())},
            {"metric": "residual_abs_mean", "value": float(resid.abs().mean())},
        ]
    )


def contribution_summary(importance: pd.DataFrame, spec: FeatureSpec) -> dict:
    total_gain = float(importance["gain"].sum())
    h3_gain = float(importance.loc[importance["feature"].isin(spec.h3_cols), "gain"].sum())
    sentiment_gain = float(importance.loc[importance["feature"].isin(spec.sentiment_cols), "gain"].sum())
    time_gain = float(importance.loc[importance["feature"].isin(spec.time_cols), "gain"].sum())
    return {
        "total_gain": total_gain,
        "h3_gain": h3_gain,
        "h3_gain_share": h3_gain / total_gain if total_gain > 0 else 0.0,
        "sentiment_gain": sentiment_gain,
        "sentiment_gain_share": sentiment_gain / total_gain if total_gain > 0 else 0.0,
        "time_gain": time_gain,
        "time_gain_share": time_gain / total_gain if total_gain > 0 else 0.0,
    }

