"""模型训练与评估。"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


def get_models() -> dict:
    """返回三个模型实例。"""
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1),
    }


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算 RMSE 和 MAE。"""
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, dict]:
    """训练所有模型并返回评估结果。

    Returns:
        字典，key 为模型名，value 包含 model, metrics, y_pred
    """
    models = get_models()
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test.values, y_pred)
        results[name] = {
            "model": model,
            "metrics": metrics,
            "y_pred": y_pred,
        }
        print(f"  {name}: RMSE={metrics['RMSE']:.6f}, MAE={metrics['MAE']:.6f}")

    return results
