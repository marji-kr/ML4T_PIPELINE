import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.utils import calculate_performance_metrics

def run_backtest(y_df, weights_history, initial_capital=10000, rebalance_period=21):
    print(f">> [4단계] 머신러닝 {rebalance_period}일 주기 리스크 관리 모델 학습 및 평가지표 연산 시작...")
    
    # 1. 원본 저장 데이터에서 S&P 500 지수 분리하기
    raw_prices_backup = pd.read_pickle('data/raw/raw_data.pkl')
    
    if isinstance(raw_prices_backup.columns, pd.MultiIndex):
        if 'Price' in raw_prices_backup.columns.levels[0]:
            cl = raw_prices_backup.xs('Adj Close', axis=1, level=1 if 'Adj Close' in raw_prices_backup.columns.levels[1] else 0)
        else:
            cl = raw_prices_backup['Adj Close'] if 'Adj Close' in raw_prices_backup.columns.levels[0] else raw_prices_backup['Close']
    else:
        cl = raw_prices_backup['Adj Close'] if 'Adj Close' in raw_prices_backup.columns else raw_prices_backup['Close']
    
    # S&P 500 실제 지수(^GSPC) 분리 추출
    if '^GSPC' in cl.columns:
        spy_prices = cl['^GSPC']
        pure_stock_prices = cl.drop(columns=['^GSPC'])
    else:
        spy_prices = cl.mean(axis=1) # 데이터 부재 시 백업 안전장치
        pure_stock_prices = cl
        
    # 2. 날짜 정렬 동기화
    common_dates = weights_history.index.intersection(pure_stock_prices.index)
    weights_history = weights_history.loc[common_dates]
    
    real_daily_returns = pure_stock_prices.pct_change().loc[common_dates].fillna(0)
    spy_real_daily = spy_prices.pct_change().loc[common_dates].fillna(0) # 실제 지수 일별 수익률
    
    # 3. 마켓 크래시 리스크 분류기 빌드
    market_vol = spy_real_daily.rolling(rebalance_period).std().fillna(0)
    market_mom = spy_real_daily.rolling(rebalance_period).mean().fillna(0)
    X_market = pd.DataFrame({'vol': market_vol, 'mom': market_mom}).shift(1).fillna(0)
    
    market_future_n_days_return = spy_real_daily.rolling(rebalance_period).sum().shift(-rebalance_period).fillna(0)
    crash_threshold = market_future_n_days_return.quantile(0.15)
    y_market = np.where(market_future_n_days_return < crash_threshold, 1, 0)
    
    risk_classifier = LogisticRegression()
    risk_classifier.fit(X_market, y_market)
    
    crash_prob_series = pd.Series(risk_classifier.predict_proba(X_market)[:, 1], index=common_dates)
    
    # 4. 포트폴리오 리스크 방어 가중치 조절
    adjusted_weights = weights_history.copy()
    current_risk_multiplier = 1.0
    risk_active_days = 0
    
    for idx, (date, row) in enumerate(weights_history.iterrows()):
        if idx % rebalance_period == 0:
            if crash_prob_series.loc[date] > 0.55:
                current_risk_multiplier = 0.10
                risk_active_days += 1
            else:
                current_risk_multiplier = 1.0
        adjusted_weights.loc[date] = row * current_risk_multiplier
        
    # 5. 최종 수익률 및 순자산 가치 연산
    portfolio_returns = (real_daily_returns * adjusted_weights).sum(axis=1)
    
    portfolio_equity = initial_capital * (1 + portfolio_returns).cumprod()
    spy_equity = initial_capital * (1 + spy_real_daily).cumprod() # 완벽한 Buy & Hold 구현
    
    final_balance = portfolio_equity.iloc[-1]
    spy_balance = spy_equity.iloc[-1]
    
    # 6. 최종 14가지 평가지표 매핑 및 출력
    my_perf = calculate_performance_metrics(portfolio_returns, benchmark_returns=spy_real_daily)
    spy_perf = calculate_performance_metrics(spy_real_daily, benchmark_returns=spy_real_daily)
    
    my_perf['Start Balance'] = f"${initial_capital:,.0f}"
    spy_perf['Start Balance'] = f"${initial_capital:,.0f}"
    my_perf['End Balance'] = f"${final_balance:,.0f}"
    spy_perf['End Balance'] = f"${spy_balance:,.0f}"
    
    summary_table = pd.DataFrame({
        '나의 ML 포트폴리오': my_perf,
        'S&P 500 (벤치마크)': spy_perf
    })
    
    print(f"   [리스크 관리 시스템] 총 {len(common_dates)}영업일 중 머신러닝이 {risk_active_days}번 위험을 감지하여 포트폴리오를 방어했습니다.")
    return summary_table, final_balance, spy_balance
