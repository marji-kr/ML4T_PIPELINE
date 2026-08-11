import os
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

class DataPipeline:
    def __init__(self, start_date="2000-01-01", end_date="2026-06-30", data_dir="data"):
        self.start = start_date
        self.end = end_date
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def fetch_and_build_dataset(self):
        local_matrix_path = os.path.join(self.data_dir, "sp500_absolute_all_universe.csv")
        
        if os.path.exists(local_matrix_path):
            print("-> 로컬에서 S&P 500 전체 유니버스 매트릭스 데이터 로드 중...")
            price_df = pd.read_csv(local_matrix_path, index_col=0, parse_dates=True)
        else:
            print("-> 🌐 위키피디아에서 실시간 S&P 500 전체 주식 리스트 크롤링 중...")
            try:
                url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
                tables = pd.read_html(url)
                sp500_table = tables[0]
                raw_tickers = sp500_table['Symbol'].tolist()
                tickers = [t.replace('.', '-') for t in raw_tickers]
            except Exception as e:
                tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "JNJ", "JPM", "V"]
            
            data_list = []
            chunk_size = 50
            for i in range(0, len(tickers), chunk_size):
                chunk = tickers[i:i+chunk_size]
                try:
                    spy_stocks = yf.download(chunk, start=self.start, end=self.end, progress=False)
                    close_matrix = spy_stocks['Adj Close'] if 'Adj Close' in spy_stocks.columns.get_level_values(0) else spy_stocks['Close']
                    data_list.append(close_matrix)
                except Exception:
                    continue
                    
            price_df = pd.concat(data_list, axis=1)
            price_df = price_df.dropna(thresh=int(len(price_df) * 0.5), axis=1).ffill().bfill()
            price_df.to_csv(local_matrix_path)
            
        returns_dict = {}
        target_dict = {}
        
        horizons = [1, 5, 21, 63]
        
        for t in price_df.columns:
            price_series = pd.to_numeric(price_df[t].astype(str).str.replace(',', ''), errors='coerce').ffill().bfill()
            returns_dict[f"{t}_Return"] = price_series.pct_change(1, fill_method=None)
            
            for h in horizons:
                target_dict[f"{t}_Target_{h}D"] = price_series.pct_change(h, fill_method=None).shift(-h)
            
        returns_df = pd.DataFrame(returns_dict, index=price_df.index)
        target_df = pd.DataFrame(target_dict, index=price_df.index)
        
        local_ff_path = os.path.join(self.data_dir, "fama_french_5_factors_raw.csv")
        if os.path.exists(local_ff_path):
            ff_df = pd.read_csv(local_ff_path, index_col=0, parse_dates=True)
        else:
            ff_factors = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench', start=self.start, end=self.end)
            ff_df = ff_factors[0] / 100.0 
            ff_df.to_csv(local_ff_path)
            
        mkt_ret = ff_df['Mkt-RF']
        derived_factors = pd.DataFrame(index=ff_df.index)
        derived_factors['Mom_21D'] = mkt_ret.rolling(21).sum()
        derived_factors['Mom_63D'] = mkt_ret.rolling(63).sum()
        derived_factors['Mom_252D'] = mkt_ret.rolling(252).sum()
        derived_factors['Vol_21D'] = mkt_ret.rolling(21).std() * np.sqrt(252)
        derived_factors['Vol_63D'] = mkt_ret.rolling(63).std() * np.sqrt(252)
        
        derived_factors['SMA_Ratio'] = mkt_ret.rolling(21).mean() / (mkt_ret.rolling(63).mean() + 1e-8)
        
        delta = mkt_ret.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        derived_factors['RSI_14D'] = 100 - (100 / (1 + rs))

        master_df = price_df.join(returns_df, how='inner')\
                             .join(target_df, how='inner')\
                             .join(ff_df, how='inner')\
                             .join(derived_factors, how='inner')\
                             .dropna()
                             
        return master_df, list(price_df.columns)