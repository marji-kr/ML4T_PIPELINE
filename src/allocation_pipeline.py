import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def allocate_assets(predictions_df, X_df, y_df, rebalance_period=21):
    print(f">> [3단계] 머신러닝 자산 배분 모델 구동... (전체 종목 리밸런싱 주기: {rebalance_period}일)")
    
    feature_cols = ['f_momentum', 'f_size', 'f_value', 'f_profit', 'f_volatility']
    
    combined_X = X_df.reset_index()
    combined_y = y_df.reset_index()
    combined_pred = predictions_df.reset_index()
    
    combined = combined_X[feature_cols].copy()
    combined['Date'] = combined_X['Date']
    combined['ticker'] = combined_X['ticker']
    combined['pred_return'] = combined_pred['pred_return']
    combined['actual_target'] = combined_y['target']
    
    # S&P 500 전체 종목 중 당일 상위 20% 우수 종목 라벨링
    combined['label'] = combined.groupby('Date')['actual_target'].transform(
        lambda x: (x >= x.quantile(0.80)).astype(int) if len(x) > 0 else 0
    )
    
    ml_features = feature_cols + ['pred_return']
    # 대용량 병렬 처리를 위해 n_jobs=-1 추가 및 트리 수 최적화
    rf_model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1)
    rf_model.fit(combined[ml_features], combined['label'])
    
    # 모델이 학습한 패턴대로 '대박 종목에 속할 확률' 산출
    combined['prob_top'] = rf_model.predict_proba(combined[ml_features])[:, 1]
    
    pivot_prob = combined.pivot(index='Date', columns='ticker', values='prob_top').fillna(0)
    # 머신러닝 학습 규칙(확률)에 비례하여 전체 자산 비중 정규화 분산
    raw_weights = pivot_prob.div(pivot_prob.sum(axis=1), axis=0).fillna(0)
    
    rebalanced_weights = raw_weights.copy()
    last_weights = raw_weights.iloc[0]
    
    for idx, (date, row) in enumerate(raw_weights.iterrows()):
        if idx % rebalance_period == 0:
            last_weights = row
        else:
            rebalanced_weights.loc[date] = last_weights
            
    return rebalanced_weights
