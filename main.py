import pandas as pd
from feature_engineering import DataProcessor, FeatureEngineer
from backtest_framework import LightGBMEnsembleModel, BacktestEngine

if __name__ == '__main__':
    # 设置回测时间范围
    START_DATE = '2022-01-01'
    END_DATE = '2024-12-31'

    # (1) 数据处理
    data_processor = DataProcessor(start_date=START_DATE, end_date=END_DATE)
    # 使用模拟的股票列表
    csi300_stocks = [f'{i:06d}.SH' for i in range(1, 11)]

    # 获取并处理数据
    raw_data = data_processor.process_data_in_chunks(csi300_stocks)

    if not raw_data.empty:
        # (2) 特征工程
        feature_engineer = FeatureEngineer(raw_data)
        final_data = feature_engineer.get_feature_ready_data()

        print("\n--- Final Data Sample ---")
        print(final_data.head())

        print("\n--- Data Info ---")
        final_data.info()

        # 定义特征和目标
        TARGET_COL = 'target_return_20d' # 预测未来20天收益
        # 排除非特征列
        EXCLUDE_COLS = ['ts_code', 'trade_date', 'return', 'open', 'high', 'low', 'close', 'pre_close',
                        'vol', 'amount', 'report_available_date', 'ann_date', 'end_date', 'target_return_20d',
                        'return_neutralized']
        FEATURE_COLS = [col for col in final_data.columns if col not in EXCLUDE_COLS and 'target' not in col]

        # 确保所有特征列都是数值或 'category'
        for col in final_data[FEATURE_COLS].select_dtypes(include=['object']).columns:
             final_data[col] = pd.to_numeric(final_data[col], errors='coerce')
        final_data = final_data.dropna(subset=FEATURE_COLS + [TARGET_COL])

        print("\n--- Starting Backtest ---")

        # (3) 初始化模型
        lgbm_ensemble = LightGBMEnsembleModel(n_estimators=5)

        # (4) 初始化并运行回测引擎
        # 回测从2023年开始，以确保有足够的历史数据进行训练 (36个月)
        backtest_engine = BacktestEngine(
            model=lgbm_ensemble,
            data=final_data,
            target_col=TARGET_COL,
            feature_cols=FEATURE_COLS,
            start_date='2023-01-01',
            end_date='2024-12-31',
            rebalance_freq='ME' # 每月调仓
        )
        backtest_engine.run_backtest()

        # (5) 分析回测结果
        backtest_engine.analyze_performance()

        # (6) 演示模型解释性
        # 在整个数据集上训练一个最终模型以进行解释
        print("\n--- Generating SHAP analysis on the full dataset model ---")
        final_model = LightGBMEnsembleModel(n_estimators=5)
        X_full, y_full = final_data[FEATURE_COLS], final_data[TARGET_COL]
        final_model.train(X_full, y_full)

        # 取一个样本进行解释（加快速度）
        X_sample = X_full.sample(min(1000, len(X_full)), random_state=42)
        final_model.interpret_model(X_sample)