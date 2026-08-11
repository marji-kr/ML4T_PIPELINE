import numpy as np
import pandas as pd
import statsmodels.api as sm

class MLPredictor:
    def __init__(self, fama_features):
        self.features = fama_features

    def analyze_fama_remedy_p_values(self, X, y):
        X_sub = X[self.features]
        X_with_const = sm.add_constant(X_sub, has_constant='add')
        model = sm.OLS(y, X_with_const)
        res_white = model.fit(cov_type='HC3')
        return res_white

    def predict_universe(self, test_df, trained_models, tickers):
        """ 유니버스 전체 자산의 기대수익률을 모델별 팩터 차원에 정확히 맞춰 연산 """
        pred_matrix = {}
        
        for t in tickers:
            if t in trained_models:
                model_obj = trained_models[t]
                params = model_obj.params
                
                # 모델 파라미터 개수 (상수항 포함)
                n_params = len(params)
                
                # 입력 피처 슬라이싱 (상수항 제외 팩터 개수 = n_params - 1)
                feat_cols = self.features[:n_params - 1]
                X_test_sub = test_df[feat_cols]
                X_test_const = sm.add_constant(X_test_sub, has_constant='add')
                
                # 차원 정렬 후 예측 연산
                pred_matrix[t] = np.dot(X_test_const.values, params)
                
        return pd.DataFrame(pred_matrix, index=test_df.index)