import os
import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

def train_and_update_model(X, y, alpha=1.0):
    print(">> [2단계] 6단원 시계열 검증 및 7단원 리지 모델 학습...")
    
    # 수치형 피처만 선택
    feature_cols = ['f_momentum', 'f_size', 'f_value', 'f_profit', 'f_volatility']
    X_train_data = X[feature_cols]
    y_train_data = y['target']
    
    model = Ridge(alpha=alpha)
    model.fit(X_train_data, y_train_data) # 전체 패턴 학습
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/ridge_model.joblib')
    return model

def predict_returns(model, X):
    print(">> [2단계] 5팩터 기반 종목별 미래 수익률 예측 중...")
    feature_cols = ['f_momentum', 'f_size', 'f_value', 'f_profit', 'f_volatility']
    preds = model.predict(X[feature_cols])
    
    result = X[['ticker']].copy()
    result['pred_return'] = preds
    return result