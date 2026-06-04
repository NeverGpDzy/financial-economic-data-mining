"""
作业4B 主程序入口
中频短线量化全流程：因子挖掘→IC/共线质检→机器学习赋权→1~2日持仓选股回测

使用方法：
    cd Code
    python -m homework4b.main
"""
import sys
import os
import time

# 确保项目根目录在Python路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from homework4b import config
from homework4b.data import load_and_preprocess
from homework4b.factors import run_factor_quality_check
from homework4b.models import run_model_training
from homework4b.backtest import run_backtest_pipeline
from homework4b.plots import generate_all_plots


def main():
    """主程序：执行完整的中频短线量化流程"""
    print("=" * 70)
    print("作业4B：中频短线量化全流程（AI辅助版）")
    print("因子挖掘→IC/共线质检→机器学习赋权→1~2日持仓选股回测")
    print("=" * 70)

    start_time = time.time()

    # ==================== 模块1：数据加载与预处理 ====================
    print("\n" + "=" * 60)
    print("模块1：本地数据加载与预处理（35分）")
    print("=" * 60)

    train_df, test_df, full_df = load_and_preprocess()

    # ==================== 模块2+3：因子质检 ====================
    full_df, ic_dict, ir_df, vif_df, ols_df = run_factor_quality_check(full_df)

    # 更新训练集和测试集（添加标准化因子列）
    std_cols = [f'{c}_std' for c in config.FACTOR_NAMES]
    train_df = full_df[
        (full_df['trade_date'] >= config.TRAIN_START) &
        (full_df['trade_date'] <= config.TRAIN_END)
    ].copy().reset_index(drop=True)

    test_df = full_df[
        (full_df['trade_date'] >= config.TEST_START) &
        (full_df['trade_date'] <= config.TEST_END)
    ].copy().reset_index(drop=True)

    # ==================== 模块4：LGBM模型训练 ====================
    model, importance_df = run_model_training(train_df)

    # ==================== 模块5：回测 ====================
    # 重新加载指数数据用于回测
    import pandas as pd
    index_path = os.path.join(config.PARQUET_ROOT, 'index_daily.parquet')
    index_df = pd.read_parquet(index_path)
    index_df['trade_date'] = pd.to_datetime(index_df['trade_date'], format='%Y%m%d')
    index_df = index_df.sort_values('trade_date').reset_index(drop=True)

    backtest_df, metrics = run_backtest_pipeline(model, test_df, index_df)

    # ==================== 绘图 ====================
    generate_all_plots(backtest_df, ic_dict, ir_df, importance_df, metrics)

    # ==================== 完成 ====================
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"作业4B全流程完成！总耗时: {elapsed:.1f}秒")
    print(f"输出目录: {config.OUTPUT_DIR}")
    print("=" * 70)

    # 打印输出文件清单
    print("\n输出文件清单:")
    for f in sorted(os.listdir(config.OUTPUT_DIR)):
        fpath = os.path.join(config.OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        print(f"  {f:40s} {size:>10,} bytes")


if __name__ == '__main__':
    main()
