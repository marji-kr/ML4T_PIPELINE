import os
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')
    table = soup.find('table', {'id': 'constituents'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        cols = row.findAll('td')
        if len(cols) > 0:
            ticker = cols[0].text.strip().replace('.', '-')
            tickers.append(ticker)
    return tickers

def fetch_raw_data(tickers_ignored, start_date, end_date):
    save_path = 'data/raw/raw_data.pkl'
    os.makedirs('data/raw', exist_ok=True)
    
    # 이미 데이터 파일이 로컬에 존재한다면 중복 다운로드 방지
    if os.path.exists(save_path):
        print(">> [1단계] 로컬에 기존 로우 데이터 파일이 존재하므로 로드합니다.")
        return pd.read_pickle(save_path)

    tickers = get_sp500_tickers() # 전체 종목 대상
    download_targets = tickers + ['^GSPC'] # 벤치마크용 지수 포함
    
    print(f">> [1단계] S&P 500 전체 종목({len(tickers)}개) 및 지수 데이터 다운로드 시작 (최초 1회 소요)...")
    df = yf.download(download_targets, start=start_date, end=end_date, progress=True)
    
    df.to_pickle(save_path)
    print(f">> [1단계] 원본 데이터가 '{save_path}'에 저장되었습니다.")
    return df

def engineer_features(raw_df, rebalance_period=21):
    print(f">> [1단계] {rebalance_period}일 예측 주기에 맞춤형 5팩터 피처 및 타깃 생성 시작...")
    
    if isinstance(raw_df.columns, pd.MultiIndex):
        if 'Price' in raw_df.columns.levels[0]:
            adj_close = raw_df.xs('Adj Close', axis=1, level=1 if 'Adj Close' in raw_df.columns.levels[1] else 0)
            volume = raw_df.xs('Volume', axis=1, level=1 if 'Volume' in raw_df.columns.levels[1] else 0)
        else:
            adj_close = raw_df['Adj Close'] if 'Adj Close' in raw_df.columns.levels[0] else raw_df['Close']
            volume = raw_df['Volume']
    else:
        adj_close = raw_df['Adj Close'] if 'Adj Close' in raw_df.columns else raw_df['Close']
        volume = raw_df['Volume']
        
    adj_close = adj_close.ffill().fillna(1.0)
    volume = volume.ffill().fillna(1.0)
    
    daily_returns = adj_close.pct_change()
    
    mom_20 = adj_close.pct_change(20)
    turnover = adj_close * volume
    size_factor = np.log(turnover.rolling(20).mean() + 1)
    value_factor = adj_close / adj_close.shift(252)
    profit_factor = (daily_returns > 0).rolling(20).sum() / 20.0
    vol_factor = daily_returns.rolling(20).std()
    
    future_n_days_returns = adj_close.pct_change(rebalance_period).shift(-rebalance_period)
    
    X_list = []
    y_list = []
    
    # 실제 개별 주식 종목들만 루프 수행 (지수 제외)
    tickers = [col for col in adj_close.columns if col != '^GSPC']
    
    for ticker in tickers:
        ticker_df = pd.DataFrame(index=adj_close.index)
        ticker_df['f_momentum'] = mom_20[ticker]
        ticker_df['f_size'] = size_factor[ticker]
        ticker_df['f_value'] = value_factor[ticker]
        ticker_df['f_profit'] = profit_factor[ticker]
        ticker_df['f_volatility'] = vol_factor[ticker]
        ticker_df['ticker'] = ticker
        
        ticker_df = ticker_df.shift(1)
        ticker_df['target'] = future_n_days_returns[ticker]
        ticker_df = ticker_df.dropna()
        
        X_list.append(ticker_df.drop(columns=['target']))
        y_list.append(ticker_df[['target', 'ticker']])
        
    X_final = pd.concat(X_list).sort_index()
    y_final = pd.concat(y_list).sort_index()
    
    X_final = X_final.replace([np.inf, -np.inf], 0).fillna(0)
    y_final = y_final.replace([np.inf, -np.inf], 0).fillna(0)
    
    os.makedirs('data/processed', exist_ok=True)
    X_final.to_pickle('data/processed/features_X.pkl')
    y_final.to_pickle('data/processed/target_y.pkl')
    print(f">> [1단계] S&P 500 전체 대상 맞춤형 타깃 동기화 완료! (총 {len(tickers)}개 종목)")
    return X_final, y_final
