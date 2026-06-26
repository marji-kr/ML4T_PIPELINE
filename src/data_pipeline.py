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
    tickers = get_sp500_tickers()[:50] # 대용량 연산 속도를 위해 상위 50개 종목 샘플링
    print(f">> [1단계] {len(tickers)}개 종목 주가 및 거래량 다운로드 시작...")
    
    df = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    os.makedirs('data/raw', exist_ok=True)
    df.to_pickle('data/raw/raw_data.pkl')
    print(">> [1단계] 원본 데이터가 'data/raw/raw_data.pkl'에 저장되었습니다.")
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
    
    # 5팩터 피처 계산
    mom_20 = adj_close.pct_change(20)
    turnover = adj_close * volume
    size_factor = np.log(turnover.rolling(20).mean() + 1)
    value_factor = adj_close / adj_close.shift(252)
    profit_factor = (daily_returns > 0).rolling(20).sum() / 20.0
    vol_factor = daily_returns.rolling(20).std()
    
    # [동기화 핵심] 리밸런싱 주기(N일) 동안의 누적 미래 수익률을 구하고 시프트합니다.
    # 예: 오늘 종가로 사서 N일 뒤 종가에 팔 때의 수익률을 오늘 행에 매칭
    future_n_days_returns = adj_close.pct_change(rebalance_period).shift(-rebalance_period)
    
    X_list = []
    y_list = []
    
    tickers = adj_close.columns
    for ticker in tickers:
        ticker_df = pd.DataFrame(index=adj_close.index)
        ticker_df['f_momentum'] = mom_20[ticker]
        ticker_df['f_size'] = size_factor[ticker]
        ticker_df['f_value'] = value_factor[ticker]
        ticker_df['f_profit'] = profit_factor[ticker]
        ticker_df['f_volatility'] = vol_factor[ticker]
        ticker_df['ticker'] = ticker
        
        # 어제까지 확정된 5팩터 데이터를 보고 미래를 예측하도록 설정
        ticker_df = ticker_df.shift(1)
        
        # 동기화된 N일 누적 미래 수익률을 정답으로 대입
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
    print(">> [1단계] 맞춤형 타깃 동기화 완료!")
    return X_final, y_final