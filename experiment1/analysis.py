"""Herd-index construction and LightGBM market-prediction analysis."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

from . import config


# ---------------------------------------------------------------------------
# Experiment 2: Herd index
# ---------------------------------------------------------------------------

def _minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def winsorize_series(series: pd.Series, sigma: float = 3.0) -> pd.Series:
    mean, std = series.mean(), series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return series
    return series.clip(mean - sigma * std, mean + sigma * std)


def build_herd_index(weekly_sentiment: pd.DataFrame) -> pd.DataFrame:
    df = weekly_sentiment.copy().sort_values("week").reset_index(drop=True)
    df["E_P_t"] = df["P_t"].shift(1).rolling(config.ROLLING_BASELINE_WEEKS, min_periods=1).mean()
    df["E_P_t"] = df["E_P_t"].fillna(df["P_t"])
    df["H1t"] = df["P_t"] - df["E_P_t"]
    denom = df["WeekPositive"] + df["WeekNegative"]
    df["H2t"] = (1 - (df["WeekPositive"] - df["WeekNegative"]).abs() / denom).where(denom.ne(0), 1.0)
    df["H1t_winsor"] = winsorize_series(df["H1t"])
    df["H2t_winsor"] = winsorize_series(df["H2t"])
    df["H1t_norm"] = _minmax(df["H1t_winsor"].abs())
    df["H2t_norm"] = _minmax(df["H2t_winsor"])
    df["H3t"] = df["H1t_norm"] * (1 - df["H2t_norm"])
    columns = [
        "week", "week_start", "week_end", "week_id", "P_t", "E_P_t", "H1t", "H2t",
        "H1t_norm", "H2t_norm", "H3t",
        "WeekPositive", "WeekNeutral", "WeekNegative", "NewsCount",
    ]
    return df[columns]


def build_modeling_dataset(weekly_herd: pd.DataFrame, hs300_weekly: pd.DataFrame) -> pd.DataFrame:
    df = weekly_herd.merge(
        hs300_weekly[["week_id", "week", "close", "return"]],
        on=["week_id", "week"], how="inner",
    )
    return df.dropna(subset=["H3t", "return"]).sort_values("week").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Experiment 3: Feature engineering + LightGBM
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct lag features, rolling statistics, and time features."""
    out = df.copy()
    # Lag features for H3t
    for lag in range(1, config.MAX_LAG + 1):
        out[f"H3_lag{lag}"] = out["H3t"].shift(lag)
    # Rolling statistics
    for w in [3, 5]:
        out[f"H3_roll_mean_{w}"] = out["H3t"].rolling(w, min_periods=1).mean()
        out[f"H3_roll_std_{w}"] = out["H3t"].rolling(w, min_periods=1).std().fillna(0)
    # Time features (use week_end as datetime)
    week_dt = pd.to_datetime(out["week"])
    out["month"] = week_dt.dt.month
    out["quarter"] = week_dt.dt.quarter
    return out


FEATURE_COLS = (
    [f"H3_lag{i}" for i in range(1, config.MAX_LAG + 1)]
    + [f"H3_roll_mean_{w}" for w in [3, 5]]
    + [f"H3_roll_std_{w}" for w in [3, 5]]
    + ["month", "quarter"]
)


def temporal_train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict temporal split: first 80% train, last 20% test."""
    n = len(df)
    split = int(n * config.TRAIN_RATIO)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def train_lgbm(X_train, y_train, X_test, y_test) -> tuple[lgb.LGBMRegressor, dict, np.ndarray]:
    """Train LightGBM, return model, metrics dict, test predictions."""
    model = lgb.LGBMRegressor(**config.LGBM_PARAMS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    return model, metrics, y_pred


def evaluate(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n_test": int(len(y_true)),
    }


def compute_shap(model, X_test) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values and feature importance."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance": np.abs(shap_values).mean(axis=0),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return shap_values, importance


def bidirectional_modeling(df: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    """Forward: H3→return. Backward: return→H3. Return metrics and comparison."""
    df = build_features(df)
    feat_cols_h3 = [c for c in FEATURE_COLS if c in df.columns]
    feat_cols_ret = []
    for lag in range(1, config.MAX_LAG + 1):
        col = f"ret_lag{lag}"
        df[col] = df["return"].shift(lag)
        feat_cols_ret.append(col)
    for w in [3, 5]:
        df[f"ret_roll_mean_{w}"] = df["return"].rolling(w, min_periods=1).mean()
        df[f"ret_roll_std_{w}"] = df["return"].rolling(w, min_periods=1).std().fillna(0)
        feat_cols_ret.extend([f"ret_roll_mean_{w}", f"ret_roll_std_{w}"])
    feat_cols_ret += ["month", "quarter"]

    valid = df.dropna(subset=feat_cols_h3 + ["return"] + feat_cols_ret)
    if len(valid) < 10:
        return {}, {}, valid

    # Forward: predict return from H3 features
    Xf = valid[feat_cols_h3].values
    yf = valid["return"].values
    split_f = int(len(valid) * config.TRAIN_RATIO)
    model_f, metrics_f, pred_f = train_lgbm(
        pd.DataFrame(Xf[:split_f], columns=feat_cols_h3),
        yf[:split_f],
        pd.DataFrame(Xf[split_f:], columns=feat_cols_h3),
        yf[split_f:],
    )

    # Backward: predict H3 from return features
    Xb = valid[feat_cols_ret].values
    yb = valid["H3t"].values
    split_b = int(len(valid) * config.TRAIN_RATIO)
    model_b, metrics_b, pred_b = train_lgbm(
        pd.DataFrame(Xb[:split_b], columns=feat_cols_ret),
        yb[:split_b],
        pd.DataFrame(Xb[split_b:], columns=feat_cols_ret),
        yb[split_b:],
    )

    # Feature importance for both directions
    fi_forward = pd.DataFrame({"feature": feat_cols_h3, "importance": model_f.feature_importances_})
    fi_forward = fi_forward.sort_values("importance", ascending=False).reset_index(drop=True)
    fi_backward = pd.DataFrame({"feature": feat_cols_ret, "importance": model_b.feature_importances_})
    fi_backward = fi_backward.sort_values("importance", ascending=False).reset_index(drop=True)

    comparison = pd.DataFrame([
        {"direction": "H3→return (forward)", **metrics_f, "best_lag": _best_lag_from_importance(fi_forward, "H3_lag")},
        {"direction": "return→H3 (backward)", **metrics_b, "best_lag": _best_lag_from_importance(fi_backward, "ret_lag")},
    ])
    return metrics_f, metrics_b, comparison


def _best_lag_from_importance(fi: pd.DataFrame, prefix: str) -> int:
    """Extract best lag number from feature importance DataFrame."""
    lag_feats = fi[fi["feature"].str.startswith(prefix)]
    if lag_feats.empty:
        return 0
    best = lag_feats.iloc[0]["feature"]
    return int(best.replace(prefix, "").replace("_lag", "").replace("lag", ""))


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_analysis_outputs(
    feature_data: pd.DataFrame,
    forward_pred: np.ndarray,
    backward_comparison: pd.DataFrame,
    shap_values: np.ndarray,
    shap_importance: pd.DataFrame,
    forward_metrics: dict,
    backward_metrics: dict,
    test_data: pd.DataFrame,
    modeling: pd.DataFrame,
) -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Feature engineering dataset
    feature_data.to_csv(config.OUTPUT_DIR / "feature_engineering.csv", index=False, encoding="utf-8-sig")
    # Forward prediction results
    pred_df = test_data[["week", "H3t", "return"]].copy()
    pred_df["predicted_return"] = forward_pred
    pred_df["residual"] = pred_df["return"] - pred_df["predicted_return"]
    pred_df.to_csv(config.OUTPUT_DIR / "lgbm_forward_results.csv", index=False, encoding="utf-8-sig")
    # Backward comparison
    backward_comparison.to_csv(config.OUTPUT_DIR / "bidirectional_comparison.csv", index=False, encoding="utf-8-sig")
    # SHAP importance
    shap_importance.to_csv(config.OUTPUT_DIR / "shap_importance.csv", index=False, encoding="utf-8-sig")
    # Summary
    summary = {"forward": forward_metrics, "backward": backward_metrics}
    pd.DataFrame(summary).to_csv(config.OUTPUT_DIR / "lgbm_metrics.csv", encoding="utf-8-sig")
