# -*- coding: utf-8 -*-
"""
CSI 300 Factor Model: Part1 - Data Processing and Feature Engineering
"""

import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
import time
import warnings

warnings.filterwarnings('ignore')

# --- 1. TuShare API Initialization ---

# ts.set_token('your token here')
# pro = ts.pro_api()

# 为了方便初步调试，此处将使用模拟数据。实际使用中请取消注释上面的代码并注释下面一直到第一部分结束的代码
# Mock Tushare Pro API for demonstration purposes
class MockTushareAPI:
    def pro_bar(self, ts_code, start_date, end_date, asset='E', adj='qfq', freq='D'):
        dates = pd.date_range(start_date, end_date, freq='D')
        data = pd.DataFrame({
            'ts_code': ts_code,
            'trade_date': dates.strftime('%Y%m%d'),
            'open': np.random.uniform(9.5, 10.5, size=len(dates)),
            'high': np.random.uniform(10.5, 11, size=len(dates)),
            'low': np.random.uniform(9, 9.5, size=len(dates)),
            'close': np.random.uniform(10, 10.5, size=len(dates)),
            'vol': np.random.uniform(1e7, 5e7, size=len(dates)),
            'amount': np.random.uniform(1e8, 5e8, size=len(dates)),
        })
        data['pre_close'] = data['close'].shift(1).fillna(10)
        return data

    def index_weight(self, index_code='000300.SH', trade_date=None):
        # 模拟返回10个成分股
        return pd.DataFrame({
            'con_code': [f'{i:06d}.SH' for i in range(1, 11)],
            'trade_date': trade_date,
            'weight': np.random.uniform(0.5, 1.5, size=10)
        })

    def income(self, ts_code, start_date, end_date, fields=''):
        return pd.DataFrame({
            'ts_code': ts_code,
            'ann_date': [f'{int(start_date[:4])-1}0415', f'{int(start_date[:4])-1}1025'],
            'end_date': [f'{int(start_date[:4])-1}1231', f'{int(start_date[:4])-1}0930'],
            'ebit': np.random.uniform(1e9, 2e9, 2),
            'n_income_attr_p': np.random.uniform(5e8, 1e9, 2) # 归母净利润
        })

    def balancesheet(self, ts_code, start_date, end_date, fields=''):
         return pd.DataFrame({
            'ts_code': ts_code,
            'ann_date': [f'{int(start_date[:4])-1}0415', f'{int(start_date[:4])-1}1025'],
            'end_date': [f'{int(start_date[:4])-1}1231', f'{int(start_date[:4])-1}0930'],
            'total_hldr_eqy_exc_min_int': np.random.uniform(5e9, 1e10, 2), # 归母所有者权益
            'total_assets': np.random.uniform(1e10, 2e10, 2),
            'total_liab': np.random.uniform(5e9, 1e10, 2)
        })

    def daily_basic(self, ts_code, start_date, end_date, fields=''):
        # 模拟市值和市净率数据
        dates = pd.date_range(start_date, end_date, freq='D')
        return pd.DataFrame({
            'ts_code': ts_code,
            'trade_date': dates.strftime('%Y%m%d'),
            'total_mv': np.random.uniform(1e10, 5e10, len(dates)),
            'pb': np.random.uniform(1.5, 4.0, len(dates)),
            'pe_ttm': np.random.uniform(10, 30, len(dates)),
        })

    def stock_industry(self, ts_code, src='citic'):
        # 模拟中信行业分类
        industries = ['银行', '非银金融', '房地产', '建筑', '建材', '钢铁', '有色金属', '基础化工', '电力及公用事业', '煤炭']
        return pd.DataFrame({'ts_code': [ts_code], 'industry': [np.random.choice(industries)]})

pro = MockTushareAPI()

# --- 2. Data Processing and Feature Engineering Pipeline ---

# 从TuShare中提取数据、处理数据并进行质量验证
class DataProcessor:

    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date
        self.pro = pro

    # 获取指定日期最新的CSI 300成分股列表
    def get_csi300_constituents(self) -> list:

        latest_date = self.end_date.replace('-', '')
        try:
            df = self.pro.index_weight(index_code='000300.SH', trade_date=latest_date)
            print(f"Successfully fetched CSI 300 constituents for {latest_date}.")
            return df['con_code'].tolist()
        except Exception as e:
            print(f"Error fetching CSI 300 constituents: {e}. Returning empty list.")
            return []

    # 通过转换数据类型优化DataFrame内存使用
    def _optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:

        for col in df.columns:
            if 'float' in str(df[col].dtype):
                df[col] = df[col].astype(np.float32)
            elif 'int' in str(df[col].dtype):
                df[col] = df[col].astype(np.int32)
        return df



    # 获取单只股票的日度行情和市值数据
    def get_daily_data_for_stock(self, ts_code: str) -> pd.DataFrame:

        try:
            # 获取后复权日线行情
            df_daily = self.pro.pro_bar(ts_code=ts_code, start_date=self.start_date, end_date=self.end_date, adj='qfq')
            # 获取日度基本指标（市值、PE、PB等）
            df_basic = self.pro.daily_basic(ts_code=ts_code, start_date=self.start_date, end_date=self.end_date)

            # 合并数据
            df = pd.merge(df_daily, df_basic, on=['ts_code', 'trade_date'], how='left')
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').reset_index(drop=True)
            return df
        except Exception as e:
            print(f"Could not fetch daily data for {ts_code}: {e}")
            return pd.DataFrame()


    # 获取成分股的财务报表数据，并进行point-in-time处理
    # 财报发布有延迟，年报通常在次年4月30日前发布；采用4个月的延迟来确保数据可用性
    def get_financial_data(self, stock_list: list) -> pd.DataFrame:
        """

        """
        all_financials = []
        for stock in stock_list:
            time.sleep(0.2) # 遵守TuShare接口频率限制
            try:
                # 获取利润表和资产负债表
                df_income = self.pro.income(ts_code=stock, start_date=self.start_date, end_date=self.end_date)
                df_balance = self.pro.balancesheet(ts_code=stock, start_date=self.start_date, end_date=self.end_date)

                # 合并财报
                df_fin = pd.merge(df_income, df_balance, on=['ts_code', 'ann_date', 'end_date'], how='inner')

                # Point-in-time adjustment: 财报数据在公告日之后4个月才可被市场完全认知
                df_fin['ann_date'] = pd.to_datetime(df_fin['ann_date'])
                df_fin['report_available_date'] = df_fin['ann_date'] + pd.DateOffset(months=4)

                all_financials.append(df_fin)
            except Exception as e:
                print(f"Could not fetch financial data for {stock}: {e}")

        if not all_financials:
            return pd.DataFrame()

        final_df = pd.concat(all_financials).reset_index(drop=True)
        return final_df

    # 使用并行处理和分块加载来高效处理大规模数据
    def process_data_in_chunks(self, stock_list: list) -> pd.DataFrame:

        all_data = []
        # 使用多进程并行获取数据
        with Pool(processes=min(8, cpu_count())) as pool:
            results = pool.map(self.get_daily_data_for_stock, stock_list)

        # 合并所有股票的日度数据
        all_data = pd.concat([res for res in results if not res.empty]).reset_index(drop=True)

        # 获取并合并财务数据
        financial_data = self.get_financial_data(stock_list)
        if not financial_data.empty:
            # 使用merge_asof实现point-in-time财务数据合并
            all_data = all_data.sort_values('trade_date')
            financial_data = financial_data.sort_values('report_available_date')
            all_data = pd.merge_asof(
                all_data,
                financial_data,
                left_on='trade_date',
                right_on='report_available_date',
                by='ts_code',
                direction='backward' # 使用最近的可用财报
            )

        # 内存优化
        all_data = self._optimize_memory(all_data)

        # 处理缺失值
        all_data = all_data.sort_values(by=['ts_code', 'trade_date']).fillna(method='ffill')

        print(f"Data processing complete. Shape: {all_data.shape}")
        return all_data


class FeatureEngineer:

    def __init__(self, data: pd.DataFrame):
        # 使用copy以避免对原始DataFrame进行修改
        self.data = data.copy()
        self.pro = pro

    def _calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line

    # 计算技术指标
    def add_technical_indicators(self) -> 'FeatureEngineer':
        df = self.data

        # 收益率
        df['return'] = df.groupby('ts_code')['close'].pct_change()

        # 移动平均线 (MA)
        for window in [5, 10, 20, 50, 200]:
            df[f'ma_{window}'] = df.groupby('ts_code')['close'].rolling(window).mean().reset_index(0, drop=True)

        # 布林带 (Bollinger Bands)
        df['bb_mid'] = df.groupby('ts_code')['close'].rolling(20).mean().reset_index(0, drop=True)
        df['bb_std'] = df.groupby('ts_code')['close'].rolling(20).std().reset_index(0, drop=True)
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

        # 相对强弱指数 (RSI)
        df['rsi_14'] = df.groupby('ts_code')['close'].transform(lambda x: self._calculate_rsi(x, 14))

        # MACD
        df['macd'] = df.groupby('ts_code')['close'].transform(lambda x: self._calculate_macd(x))

        # 量价加权平均价 (VWAP)
        df['vwap_5'] = (df['amount'] * 1000 / (df['vol'] * 100 + 1e-6)).rolling(5).mean()

        self.data = df
        print("Technical indicators added.")
        return self

    # 计算基本面因子
    def add_fundamental_factors(self) -> 'FeatureEngineer':
        df = self.data

        # 确保财务数据列存在
        if 'pe_ttm' in df.columns:
            df['EP_ratio'] = 1 / df['pe_ttm']
        if 'pb' in df.columns:
            df['BP_ratio'] = 1 / df['pb']
        if 'total_hldr_eqy_exc_min_int' in df.columns and 'n_income_attr_p' in df.columns:
            df['ROE'] = df['n_income_attr_p'] / df['total_hldr_eqy_exc_min_int']
        if 'total_assets' in df.columns and 'n_income_attr_p' in df.columns:
            df['ROA'] = df['n_income_attr_p'] / df['total_assets']
        if 'total_liab' in df.columns and 'total_hldr_eqy_exc_min_int' in df.columns:
            df['debt_to_equity'] = df['total_liab'] / df['total_hldr_eqy_exc_min_int']

        self.data = df
        print("Fundamental factors added.")
        return self

    # 计算市场微观结构因子
    def add_market_microstructure_factors(self) -> 'FeatureEngineer':

        df = self.data

        # 振幅 (近似买卖价差)
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']

        # 换手率 (近似订单流不平衡)
        df['turnover_rate'] = df['vol'] * 100 / (df['total_mv'] / df['close'] + 1e-6)

        self.data = df
        print("Market microstructure factors added.")
        return self

    # 处理行业分类特征：获取行业分类；对收益率进行行业去均值（中性化）
    def handle_categorical_features(self) -> 'FeatureEngineer':

        df = self.data
        stock_list = df['ts_code'].unique()

        # 1. 获取行业分类
        industry_map = {}
        for stock in stock_list:
            try:
                # 实际使用中，批量获取或从本地数据库读取
                time.sleep(0.1)
                industry_df = self.pro.stock_industry(ts_code=stock, src='citic')
                if not industry_df.empty:
                    industry_map[stock] = industry_df.iloc[0]['industry']
            except Exception as e:
                print(f"Could not get industry for {stock}: {e}")

        df['industry'] = df['ts_code'].map(industry_map)
        df['industry'] = df['industry'].fillna('未知')

        # LightGBM原生支持分类特征，只需转换为 'category' 类型
        df['industry'] = df['industry'].astype('category')

        # 2. 行业收益去均值 (Sector Demeaning)
        # 计算每个行业在每一天的平均收益率
        industry_mean_return = df.groupby(['trade_date', 'industry'])['return'].transform('mean')
        df['return_neutralized'] = df['return'] - industry_mean_return

        self.data = df
        print("Categorical features handled and returns neutralized.")
        return self

    # 创建未来收益率作为预测目标 y
    def create_target_variable(self, period: int = 20) -> 'FeatureEngineer':

        # 目标是未来`period`个交易日的收益率
        self.data[f'target_return_{period}d'] = self.data.groupby('ts_code')['return'].shift(-period)
        return self

    # 运行所有步骤并返回最终处理好的数据
    def get_feature_ready_data(self) -> pd.DataFrame:

        self.add_technical_indicators()
        self.add_fundamental_factors()
        self.add_market_microstructure_factors()
        self.handle_categorical_features()
        self.create_target_variable()

        # 清理数据：删除包含NaN的行，并替换无穷大值
        final_df = self.data.dropna().copy()
        final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        final_df.dropna(inplace=True)

        print(f"Feature engineering complete. Final data shape: {final_df.shape}")
        return final_df

 




        


