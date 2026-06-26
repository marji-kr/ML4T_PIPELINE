import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def allocate_assets(predictions_df, X_df, y_df, rebalance_period=21):
    print(f">> [3단계] 머신러닝 자산 배분 모델 구동... (리밸런싱 주기: {rebalance_period}일)")
    
    feature_cols = ['f_momentum', 'f_size', 'f_value', 'f_profit', 'f_volatility']
    
    # 인덱스가 꼬여서 KeyError가 나는 것을 방지하기 위해 완전히 초기화 후 병합합니다.
    combined_X = X_df.reset_index()
    combined_y = y_df.reset_index()
    combined_pred = predictions_df.reset_index()
    
    # 날짜와 종목 순서가 완벽히 일치하므로 안전하게 컬럼을 합칩니다.
    combined = combined_X[feature_cols].copy()
    combined['Date'] = combined_X['Date']
    combined['ticker'] = combined_X['ticker']
    combined['pred_return'] = combined_pred['pred_return']
    combined['actual_target'] = combined_y['target']
    
    # 상위 20% 대박 종목 라벨링
    combined['label'] = combined.groupby('Date')['actual_target'].transform(
        lambda x: (x >= x.quantile(0.80)).astype(int) if len(x) > 0 else 0
    )
    
    ml_features = feature_cols + ['pred_return']
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf_model.fit(combined[ml_features], combined['label'])
    
    # 대박 확률 생성
    combined['prob_top'] = rf_model.predict_proba(combined[ml_features])[:, 1]
    
    # 피벗 연산 수행 (KeyError 방지 완료)
    pivot_prob = combined.pivot(index='Date', columns='ticker', values='prob_top').fillna(0)
    raw_weights = pivot_prob.div(pivot_prob.sum(axis=1), axis=0).fillna(0)
    
    # 리밸런싱 주기 반영 루프
    rebalanced_weights = raw_weights.copy()
    last_weights = raw_weights.iloc[0]
    
    for idx, (date, row) in enumerate(raw_weights.iterrows()):
        if idx % rebalance_period == 0:
            last_weights = row
        else:
            rebalanced_weights.loc[date] = last_weights
            
    return rebalanced_weights