"""
作业4B 多因子机器学习赋权模块（LGBM非线性）
功能：LGBM模型训练、时间序列交叉验证、特征重要性分析
"""
import os
import warnings
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple

import lightgbm as lgb

from . import config

warnings.filterwarnings('ignore')


def prepare_model_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    准备建模数据

    特征：所有因子标准化值
    标签：个股次日超额收益

    Returns:
        X: 特征DataFrame
        y: 标签Series
    """
    feature_cols = [f'{c}_std' for c in config.FACTOR_NAMES]
    label_col = 'next_excess_ret'

    # 去除NaN
    model_df = df[feature_cols + [label_col, 'trade_date']].dropna()

    X = model_df[feature_cols]
    y = model_df[label_col]

    print(f"[建模数据] 特征: {feature_cols}")
    print(f"[建模数据] 样本数: {len(X)}")
    return X, y


def train_lgbm_with_cv(X: pd.DataFrame, y: pd.Series,
                        dates: pd.Series) -> Tuple[lgb.Booster, Dict, list]:
    """
    使用时间序列交叉验证训练LightGBM模型

    Args:
        X: 特征数据
        y: 标签数据
        dates: 日期序列（用于时间序列分割）

    Returns:
        best_model: 最佳LGBM模型
        cv_results: 交叉验证结果
        importance_list: 特征重要性列表
    """
    print("\n[LGBM训练] 开始时间序列交叉验证...")

    # 按时间排序
    sorted_idx = dates.argsort()
    X = X.iloc[sorted_idx].reset_index(drop=True)
    y = y.iloc[sorted_idx].reset_index(drop=True)
    dates = dates.iloc[sorted_idx].reset_index(drop=True)

    # 时间序列K折分割
    unique_dates = sorted(dates.unique())
    n_dates = len(unique_dates)
    fold_size = n_dates // config.LGBM_K_FOLD

    cv_results = {
        'train_mse': [],
        'val_mse': [],
        'fold': []
    }
    importance_list = []
    models = []

    for fold in range(config.LGBM_K_FOLD):
        # 时间序列分割：前N折为训练，后1折为验证
        val_start_idx = fold * fold_size
        val_end_idx = min((fold + 1) * fold_size, n_dates)

        val_dates = set(unique_dates[val_start_idx:val_end_idx])
        train_dates = set(unique_dates[:val_start_idx])

        if len(train_dates) == 0:
            continue

        train_mask = dates.isin(train_dates)
        val_mask = dates.isin(val_dates)

        X_train, X_val = X[train_mask], X[val_mask]
        y_train, y_val = y[train_mask], y[val_mask]

        if len(X_train) < 100 or len(X_val) < 100:
            continue

        # 创建LGB数据集
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # 训练模型
        model = lgb.train(
            config.LGBM_PARAMS,
            dtrain,
            num_boost_round=config.LGBM_NUM_BOOST_ROUND,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(config.LGBM_EARLY_STOPPING),
                lgb.log_evaluation(0)
            ]
        )

        # 计算MSE
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        train_mse = np.mean((y_train - train_pred) ** 2)
        val_mse = np.mean((y_val - val_pred) ** 2)

        cv_results['train_mse'].append(train_mse)
        cv_results['val_mse'].append(val_mse)
        cv_results['fold'].append(fold + 1)

        # 特征重要性
        importance = model.feature_importance(importance_type='gain')
        importance_list.append(importance)
        models.append(model)

        print(f"  Fold {fold + 1}: 训练MSE={train_mse:.8f}, 验证MSE={val_mse:.8f}, "
              f"迭代次数={model.best_iteration}")

    # 选择验证MSE最低的模型
    best_fold = np.argmin(cv_results['val_mse'])
    best_model = models[best_fold]
    best_fold_num = cv_results['fold'][best_fold]
    print(f"\n[LGBM训练] 最佳模型: Fold {best_fold_num}, "
          f"验证MSE={cv_results['val_mse'][best_fold]:.8f}")

    return best_model, cv_results, importance_list


def train_final_model(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.Booster:
    """
    使用全部训练集数据训练最终模型

    Returns:
        model: 训练好的LGBM模型
    """
    print("\n[最终模型] 使用全部训练集训练LGBM模型...")

    dtrain = lgb.Dataset(X_train, label=y_train)

    model = lgb.train(
        config.LGBM_PARAMS,
        dtrain,
        num_boost_round=config.LGBM_NUM_BOOST_ROUND,
        callbacks=[lgb.log_evaluation(50)]
    )

    print(f"[最终模型] 训练完成，迭代次数={model.best_iteration}")
    return model


def get_feature_importance(model: lgb.Booster) -> pd.DataFrame:
    """
    获取模型特征重要性

    Returns:
        importance_df: 特征重要性DataFrame
    """
    feature_names = config.FACTOR_NAMES
    importance = model.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        '因子': feature_names,
        '重要性': importance,
        '重要性占比': importance / (importance.sum() + 1e-10)
    }).sort_values('重要性', ascending=False).reset_index(drop=True)

    print("[特征重要性]")
    for _, row in importance_df.iterrows():
        print(f"  {row['因子']}: {row['重要性']:.2f} ({row['重要性占比']:.2%})")

    return importance_df


def save_model(model: lgb.Booster) -> str:
    """保存模型到文件"""
    model_path = os.path.join(config.OUTPUT_DIR, 'lgbm_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[模型保存] 模型已保存至 {model_path}")
    return model_path


def run_model_training(train_df: pd.DataFrame) -> Tuple[lgb.Booster, pd.DataFrame]:
    """
    完整模型训练流水线

    Returns:
        model: 训练好的LGBM模型
        importance_df: 特征重要性
    """
    print("\n" + "=" * 60)
    print("模块4：多因子机器学习赋权（LGBM非线性）")
    print("=" * 60)

    # 1. 准备数据
    X, y = prepare_model_data(train_df)
    dates = train_df.loc[X.index, 'trade_date']

    # 2. 交叉验证训练
    best_model, cv_results, importance_list = train_lgbm_with_cv(X, y, dates)

    # 3. 使用全部训练集训练最终模型
    final_model = train_final_model(X, y)

    # 4. 特征重要性
    importance_df = get_feature_importance(final_model)

    # 5. 保存模型和结果
    save_model(final_model)
    importance_df.to_csv(os.path.join(config.OUTPUT_DIR, 'feature_importance.csv'), index=False)

    # 保存交叉验证结果
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(config.OUTPUT_DIR, 'cv_results.csv'), index=False)

    return final_model, importance_df
