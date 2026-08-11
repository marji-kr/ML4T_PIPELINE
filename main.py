import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
sys.path.append(SRC_DIR)

from data_pipeline import DataPipeline
from ml_predictor import MLPredictor
from backtest_engine import EventDrivenBacktester
from risk_manager import RiskManager

def main():
    print("=====================================================================")
    print("  [통합] Sharpe Ratio 기반 37% 최적 정지 & 다변화 백테스트 엔진")
    print("=====================================================================")
    
    dp = DataPipeline()
    df_full, tickers = dp.fetch_and_build_dataset()
    
    # 1. 학습/검증 (2000~2019) 및 테스트 (2020~2026) 데이터 분리
    train_data = df_full.loc["2000-01-01":"2019-12-31"]
    
    # 2. 팩터 그룹 설정 (5-Factor 기본 + 파생 팩터 결합)
    ff5_set = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    
    strategy_factor_groups = [
        {"group_name": "5팩터 펀더멘털 세트", "factors": ff5_set},
        {"group_name": "5팩터 + 모멘텀", "factors": ff5_set + ['Mom_21D', 'Mom_63D']},
        {"group_name": "5팩터 + 변동성", "factors": ff5_set + ['Vol_21D', 'Vol_63D']},
        {"group_name": "5팩터 + 기술적(RSI/SMA)", "factors": ff5_set + ['SMA_Ratio', 'RSI_14D']},
        {"group_name": "5팩터 + 종합 올인원", "factors": ff5_set + ['Mom_63D', 'Vol_21D', 'SMA_Ratio', 'RSI_14D']}
    ]
    
    # 3. 예측 타겟 기간 (1일, 5일, 21일, 63일 뒤 예측) & 자산배분 방식
    prediction_horizons = [1, 5, 21, 63]
    allocation_methods = ['equal', 'mvp', 'risk_parity', 'mvo']
    
    candidate_strategies = []
    idx = 1
    for horizon in prediction_horizons:
        for group in strategy_factor_groups:
            for alloc in allocation_methods:
                factors_str = ", ".join(group["factors"])
                candidate_strategies.append({
                    "id": idx,
                    "horizon": horizon,
                    "group_name": group["group_name"],
                    "factors": group["factors"],
                    "allocation": alloc,
                    "name": f"전략_{idx} [{horizon}일뒤 예측 | {group['group_name']} | 자산배분: {alloc}]",
                    "description": f"예측주기: {horizon}일 | 사용팩터: [{factors_str}] | 자산배분: {alloc}"
                })
                idx += 1
            
    print(f"-> 총 생성된 다변화 전략 후보군 개수: {len(candidate_strategies)}개")
    
    # 4. Sharpe Ratio 기반 37% 최적 정지 이론 (Optimal Stopping Rule) 적용
    N = len(candidate_strategies)
    k = int(np.ceil(N * 0.37))
    
    backtester = EventDrivenBacktester()
    evaluation_logs = []
    
    def evaluate_strategy(strat_cfg):
        strat_factors = strat_cfg["factors"]
        horizon = strat_cfg["horizon"]
        
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(train_data[strat_factors]), 
            columns=strat_factors, 
            index=train_data.index
        )
        
        predictor = MLPredictor(strat_factors)
        trained_models = {}
        
        for t in tickers[:50]: # 유니버스 샘플 스캔
            target_col = f"{t}_Target_{horizon}D"
            if target_col in train_data.columns:
                fit_res = predictor.analyze_fama_remedy_p_values(X_train_scaled, train_data[target_col])
                trained_models[t] = fit_res
                
        pred_matrix = predictor.predict_universe(X_train_scaled, trained_models, list(trained_models.keys()))
        port_vals = backtester.run_absolute_top10_strategy(
            train_data, pred_matrix, list(trained_models.keys()), 
            rebalance_freq=horizon, alloc_method=strat_cfg["allocation"], slippage=0.0005
        )
        returns = port_vals.pct_change().dropna()
        sharpe_score = RiskManager.calculate_sharpe_ratio(returns)
        return sharpe_score, trained_models, scaler

    print(f"\n[1단계: 37% 표본 관찰 구간] 총 {k}개 전략 백테스트 및 샤프지수 평가...")
    sample_best_sharpe = -999.0
    
    for i in range(k):
        strat = candidate_strategies[i]
        sharpe, models, scaler = evaluate_strategy(strat)
        strat["sharpe_ratio"] = sharpe
        evaluation_logs.append({
            "stage": "Observation (37%)", 
            "strategy": strat["name"], 
            "sharpe_ratio": sharpe, 
            "selected": False
        })
        if sharpe > sample_best_sharpe:
            sample_best_sharpe = sharpe

    print(f"-> 37% 관찰 구간 최고 샤프지수 Threshold: {sample_best_sharpe:.4f}")
    
    print(f"\n[2단계: 선택 구간] Threshold 초과 시 즉시 최종 모델 채택...")
    selected_strategy = None
    selected_models = None
    selected_scaler = None
    
    for i in range(k, N):
        strat = candidate_strategies[i]
        sharpe, models, scaler = evaluate_strategy(strat)
        strat["sharpe_ratio"] = sharpe
        
        if sharpe > sample_best_sharpe and selected_strategy is None:
            selected_strategy = strat
            selected_models = models
            selected_scaler = scaler
            evaluation_logs.append({
                "stage": "Selection", 
                "strategy": strat["name"], 
                "sharpe_ratio": sharpe, 
                "selected": True
            })
            print(f"🎯 최적 모델 선발! [{strat['name']}] (Sharpe: {sharpe:.4f} > Cutoff: {sample_best_sharpe:.4f})")
        else:
            evaluation_logs.append({
                "stage": "Selection", 
                "strategy": strat["name"], 
                "sharpe_ratio": sharpe, 
                "selected": False
            })

    if selected_strategy is None:
        selected_strategy = candidate_strategies[-1]
        _, selected_models, selected_scaler = evaluate_strategy(selected_strategy)

    # 5. 저장용 아티팩트 보관
    os.makedirs("model", exist_ok=True)
    stop_summary = {
        "total_candidates": N,
        "sample_size_37pct": k,
        "threshold_sharpe": sample_best_sharpe,
        "selected_strategy": selected_strategy,
        "logs": evaluation_logs
    }
    
    with open("model/dsr_optimal_stopping.json", "w", encoding="utf-8") as f:
        json.dump(stop_summary, f, ensure_ascii=False, indent=4)
        
    with open("model/universe_models.pkl", "wb") as m_f:
        pickle.dump(selected_models, m_f)
        
    with open("model/scaler.pkl", "wb") as s_f:
        pickle.dump(selected_scaler, s_f)
        
    with open("model/selected_factors.pkl", "wb") as f_f:
        pickle.dump(selected_strategy["factors"], f_f)
        
    print("\n💾 최적 정지 수행 기록 및 모델 저장 완료.")

if __name__ == "__main__":
    main()