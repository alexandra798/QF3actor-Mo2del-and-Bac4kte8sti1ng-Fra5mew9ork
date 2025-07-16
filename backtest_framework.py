"""
CSI 300 Factor Model: Part2 - LightGBM Modeling, SHAP Interpretation, and Backtesting
"""

import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
import time
import warnings
import lightgbm as lgb
import shap
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('seaborn-v0_8-darkgrid')

# --- 4. LightGBM Integration with Factor Research ---

# 封装了LightGBM集成模型的训练、预测和解释功能
class LightGBMEnsembleModel:

    # 初始化集成模型
    # n_estimators: 集成中基础模型的数量
    # lgb_params: LightGBM模型的参数
    def __init__(self, n_estimators=5, lgb_params=None):
        """

        """
        if lgb_params is None:
            self.lgb_params = {
                "objective": "regression_l1", # L1 loss (MAE) 对异常值更鲁棒
                "metric": "l2", # L2 metric (MSE)
                "boosting_type": "gbdt",
                "num_leaves": 80,
                "max_depth": 7,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
                "bagging_freq": 5,
                "min_child_samples": 20,
                "reg_alpha": 0.1,  # L1 regularization
                "reg_lambda": 0.1, # L2 regularization
                "random_state": 42,
                "n_jobs": -1,
                "force_col_wise": True,
                "verbose": -1
            }
        else:
            self.lgb_params = lgb_params

        self.n_estimators = n_estimators
        self.models = []


    # 训练集成模型，每个模型使用不同的随机种子以增加多样性
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.models = []
        # 将行业特征转换为LightGBM可识别的 'category' 类型
        categorical_features = ['industry'] if 'industry' in X_train.columns else 'auto'

        for i in range(self.n_estimators):
            params = self.lgb_params.copy()
            params['random_state'] = self.lgb_params.get('random_state', 42) + i # 关键：为每个模型设置不同种子

            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train,
                      categorical_feature=categorical_features)
            self.models.append(model)
        print(f"Ensemble model with {self.n_estimators} estimators trained.")

    # 对新数据进行预测，返回集成模型的平均预测值
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        predictions = np.zeros(len(X_test))
        for model in self.models:
            predictions += model.predict(X_test)

        return predictions / len(self.models)

    # 使用SHAP来解释模型;分析第一个基模型作为整个集成的代表
    def interpret_model(self, X_sample: pd.DataFrame):

        if not self.models:
            raise RuntimeError("Model has not been trained yet.")

        explainer = shap.TreeExplainer(self.models[0])
        shap_values = explainer.shap_values(X_sample)

        print("\n--- Model Interpretation with SHAP ---")
        print("SHAP values provide model interpretability, showing the impact of each feature on the prediction.")
        print("Permutation importance offers a robust ranking, while Partial Dependence Plots (PDP) can show specific factor relationships.")

        # 绘制SHAP摘要图
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance')
        plt.show()

        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title('SHAP Value Distribution per Feature')
        plt.show()

# --- 5. Backtesting Framework ---
# 实现一个基于滚动窗口的回测引擎

class BacktestEngine:
    def __init__(self, model, data: pd.DataFrame, target_col: str, feature_cols: list,
                 start_date: str, end_date: str, rebalance_freq='ME',
                 train_period_months=36, transaction_cost=0.001):
        self.model = model
        self.data = data.copy()
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.rebalance_freq = rebalance_freq
        self.train_period = pd.DateOffset(months=train_period_months)
        self.transaction_cost = transaction_cost
        self.portfolio_returns = []

    # 执行回测循环
    def run_backtest(self):
        rebalance_dates = pd.date_range(self.start_date, self.end_date, freq=self.rebalance_freq)

        for i, current_date in enumerate(rebalance_dates):
            # 1. 定义训练和测试时间窗口 (Walk-forward)
            train_start_date = current_date - self.train_period
            train_end_date = current_date - pd.DateOffset(days=1)

            test_start_date = current_date
            # 确定测试结束日期
            if i + 1 < len(rebalance_dates):
                test_end_date = rebalance_dates[i+1] - pd.DateOffset(days=1)
            else:
                test_end_date = self.end_date

            # 2. 划分数据
            train_df = self.data[(self.data['trade_date'] >= train_start_date) & (self.data['trade_date'] <= train_end_date)]
            test_df = self.data[(self.data['trade_date'] >= test_start_date) & (self.data['trade_date'] <= test_end_date)].copy()

            if train_df.empty or test_df.empty:
                print(f"Skipping period starting {current_date.date()}: Not enough data.")
                continue

            X_train, y_train = train_df[self.feature_cols], train_df[self.target_col]
            X_test = test_df[self.feature_cols]

            # 3. 训练模型
            print(f"Rebalancing on {current_date.date()}: Training model from {train_start_date.date()} to {train_end_date.date()}...")
            self.model.train(X_train, y_train)

            # 4. 获取预测
            predictions = self.model.predict(X_test)
            test_df['prediction'] = predictions

            # 5. 投资组合构建 (多空策略)
            # 按天对预测值进行分组，并构建投资组合
            daily_returns = []
            for date, daily_group in test_df.groupby('trade_date'):
                # 根据预测值将股票分为5个分位数
                daily_group['quintile'] = pd.qcut(daily_group['prediction'], 5, labels=False, duplicates='drop')

                # 做多预测收益率最高的分位数，做空最低的
                long_portfolio = daily_group[daily_group['quintile'] == 4]
                short_portfolio = daily_group[daily_group['quintile'] == 0]

                if long_portfolio.empty or short_portfolio.empty:
                    daily_return = 0
                else:
                    # 计算当天多空组合的收益率
                    long_return = long_portfolio['return'].mean()
                    short_return = short_portfolio['return'].mean()
                    daily_return = 0.5 * long_return - 0.5 * short_return

                daily_returns.append(daily_return)

            # 计算本期收益，并扣除交易成本 (在期初调仓时发生)
            period_returns = pd.Series(daily_returns).fillna(0)
            if not period_returns.empty:
                 period_returns.iloc[0] -= self.transaction_cost # 在第一个交易日扣除双边成本

            self.portfolio_returns.extend(period_returns.tolist())
            print(f"Period from {test_start_date.date()} to {test_end_date.date()} complete. Average daily return: {period_returns.mean():.4f}")

    # 计算并展示回测的绩效指标
    def analyze_performance(self):

        if not self.portfolio_returns:
            print("No backtest results to analyze.")
            return

        returns_series = pd.Series(self.portfolio_returns).fillna(0)

        # 1. 累计收益曲线 (Equity Curve)
        equity_curve = (1 + returns_series).cumprod()

        # 2. 计算绩效指标
        total_return = equity_curve.iloc[-1] - 1
        annualized_return = (1 + returns_series.mean())**252 - 1
        annualized_volatility = returns_series.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0

        # 计算最大回撤
        cumulative = equity_curve
        high_watermark = cumulative.cummax()
        drawdown = (cumulative - high_watermark) / high_watermark
        max_drawdown = drawdown.min()

        # 3. 打印报告
        print("\n--- Backtest Performance Analysis ---")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Annualized Volatility: {annualized_volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")

        # 4. 绘制权益曲线
        plt.figure(figsize=(12, 7))
        equity_curve.plot()
        plt.title('Portfolio Equity Curve (Long-Short Strategy)')
        plt.xlabel('Trading Days')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.show()

