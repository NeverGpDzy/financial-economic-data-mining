"""LightGBM modeling for Homework 6."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config


def prepare_training_data(panel: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    train = panel[
        (panel["year"] >= config.TRAIN_START_YEAR)
        & (panel["year"] <= config.MODEL_TRAIN_END_YEAR)
        & panel["target_fcff_growth_1y"].notna()
    ].copy()
    model_df = train[feature_cols + ["target_fcff_growth_1y", "year"]].dropna()
    if model_df.empty:
        raise ValueError("LGBM训练数据为空")
    x = model_df[feature_cols]
    y = model_df["target_fcff_growth_1y"]
    years = model_df["year"]
    return x, y, years


def time_series_cv(x: pd.DataFrame, y: pd.Series, years: pd.Series) -> pd.DataFrame:
    unique_years = sorted(years.unique())
    rows = []
    if len(unique_years) < 2:
        return pd.DataFrame(columns=["fold", "train_years", "val_year", "train_mse", "val_mse", "best_iteration"])

    for fold, val_year in enumerate(unique_years[1:], start=1):
        train_years = [year for year in unique_years if year < val_year]
        train_mask = years.isin(train_years)
        val_mask = years.eq(val_year)
        if train_mask.sum() < 20 or val_mask.sum() < 8:
            continue

        dtrain = lgb.Dataset(x.loc[train_mask], label=y.loc[train_mask])
        dval = lgb.Dataset(x.loc[val_mask], label=y.loc[val_mask], reference=dtrain)
        model = lgb.train(
            config.LGBM_PARAMS,
            dtrain,
            num_boost_round=config.LGBM_NUM_BOOST_ROUND,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(config.LGBM_EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        train_pred = model.predict(x.loc[train_mask], num_iteration=model.best_iteration)
        val_pred = model.predict(x.loc[val_mask], num_iteration=model.best_iteration)
        rows.append(
            {
                "fold": fold,
                "train_years": ",".join(map(str, train_years)),
                "val_year": int(val_year),
                "train_mse": float(np.mean((y.loc[train_mask] - train_pred) ** 2)),
                "val_mse": float(np.mean((y.loc[val_mask] - val_pred) ** 2)),
                "best_iteration": int(model.best_iteration or config.LGBM_NUM_BOOST_ROUND),
            }
        )
    return pd.DataFrame(rows)


def train_final_model(x: pd.DataFrame, y: pd.Series) -> lgb.Booster:
    dtrain = lgb.Dataset(x, label=y)
    model = lgb.train(
        config.LGBM_PARAMS,
        dtrain,
        num_boost_round=config.LGBM_NUM_BOOST_ROUND,
        callbacks=[lgb.log_evaluation(0)],
    )
    return model


def feature_importance(model: lgb.Booster, feature_cols: list[str]) -> pd.DataFrame:
    gain = model.feature_importance(importance_type="gain")
    split = model.feature_importance(importance_type="split")
    df = pd.DataFrame(
        {
            "feature": feature_cols,
            "factor": [col.removesuffix("_z") for col in feature_cols],
            "gain_importance": gain,
            "split_importance": split,
        }
    )
    total_gain = df["gain_importance"].sum()
    df["importance_share"] = df["gain_importance"] / (total_gain + 1e-10)
    df["因子"] = df["factor"].map(lambda f: config.FACTOR_META[f]["label"])
    return df.sort_values("gain_importance", ascending=False).reset_index(drop=True)


def score_panel(panel: pd.DataFrame, model: lgb.Booster, feature_cols: list[str]) -> pd.DataFrame:
    scored = panel.copy()
    valid = scored[feature_cols].notna().all(axis=1)
    scored["ml_score"] = np.nan
    scored.loc[valid, "ml_score"] = model.predict(scored.loc[valid, feature_cols])
    return scored


def run_modeling(panel: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, lgb.Booster, pd.DataFrame, pd.DataFrame]:
    x, y, years = prepare_training_data(panel, feature_cols)
    cv_results = time_series_cv(x, y, years)
    model = train_final_model(x, y)
    importance = feature_importance(model, feature_cols)
    scored = score_panel(panel, model, feature_cols)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.OUTPUT_DIR / "lgbm_fcff_model.pkl", "wb") as f:
        pickle.dump(model, f)
    cv_results.to_csv(config.OUTPUT_DIR / "lgbm_cv_results.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(config.OUTPUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    return scored, model, cv_results, importance
