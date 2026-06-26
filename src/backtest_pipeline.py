import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.utils import calculate_performance_metrics

def run_backtest(y_df, weights_history, initial_capital=10000, rebalance_period=21):
    print(f">> [4단계] 머신러닝 {rebalance_period}일 주기 리스크 관리 모델 학습 및 평가지표 연산 시작...")
    
    y_reset = y_df.reset_index()
    actual_returns = y_reset.pivot(index='Date', columns='ticker', values='target').fillna(0)
    
    common_dates = actual_returns.index.intersection(weights_history.index)
    actual_returns = actual_returns.loc[common_dates]
    weights_history = weights_history.loc[common_dates]
    
    spy_daily_returns = actual_returns.mean(axis=1)
    
    # 리스크 모델 빌드
    market_vol = spy_daily_returns.rolling(rebalance_period).std().fillna(0)
    market_mom = spy_daily_returns.rolling(rebalance_period).mean().fillna(0)
    X_market = pd.DataFrame({'vol': market_vol, 'mom': market_mom}).shift(1).fillna(0)
    
    market_future_n_days_return = spy_daily_returns.rolling(rebalance_period).sum().shift(-rebalance_period).fillna(0)
    crash_threshold = market_future_n_days_return.quantile(0.15)
    y_market = np.where(market_future_n_days_return < crash_threshold, 1, 0)
    
    risk_classifier = LogisticRegression()
    risk_classifier.fit(X_market, y_market)
    
    crash_prob_series = pd.Series(risk_classifier.predict_proba(X_market)[:, 1], index=actual_returns.index)
    
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
        
    # 원본 일별 주가 복원
    raw_prices_backup = pd.read_pickle('data/raw/raw_data.pkl')
    if isinstance(raw_prices_backup.columns, pd.MultiIndex):
        if 'Price' in raw_prices_backup.columns.levels[0]:
            cl = raw_prices_backup.xs('Adj Close', axis=1, level=1 if 'Adj Close' in raw_prices_backup.columns.levels[1] else 0)
        else:
            cl = raw_prices_backup['Adj Close'] if 'Adj Close' in raw_prices_backup.columns.levels[0] else raw_prices_backup['Close']
    else:
        cl = raw_prices_backup['Adj Close'] if 'Adj Close' in raw_prices_backup.columns else raw_prices_backup['Close']
        
    real_daily_returns = cl.pct_change().loc[common_dates].fillna(0)
    spy_real_daily = real_daily_returns.mean(axis=1)
    
    portfolio_returns = (real_daily_returns * adjusted_weights).sum(axis=1)
    
    portfolio_equity = initial_capital * (1 + portfolio_returns).cumprod()
    spy_equity = initial_capital * (1 + spy_real_daily).cumprod()
    
    final_balance = portfolio_equity.iloc[-1]
    spy_balance = spy_equity.iloc[-1]
    
    # 14가지 평가지표 최종 연산
    my_perf = calculate_performance_metrics(portfolio_returns, benchmark_returns=spy_real_daily)
    spy_perf = calculate_performance_metrics(spy_real_daily, benchmark_returns=spy_real_daily)
    
    # 상단 Balance 텍스트 매핑 우회 등록
    my_perf['Start Balance'] = f"${initial_capital:,.0f}"
    spy_perf['Start Balance'] = f"${initial_capital:,.0f}"
    my_perf['End Balance'] = f"${final_balance:,.0f}"
    spy_perf['End Balance'] = f"${spy_balance:,.0f}"
    
    summary_table = pd.DataFrame({
        '나의 ML 포트폴리오': my_perf,
        'S&P 500 (벤치마크)': spy_perf
    })
    
    print(f"   [리스크 관리 시스템] 총 {len(common_dates)}영업일 중 머신러닝이 {risk_active_days}번의 리밸런싱 시점에 장기 위험을 감지해 자산을 방어했습니다.")
    return summary_table, final_balance, spy_balance