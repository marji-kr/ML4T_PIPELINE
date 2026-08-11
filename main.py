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
    print("  [통합] DSR 37% 최적 정지 및 5-Factor 결합 다변화 백테스트 엔진")
    print("=====================================================================")
    
    dp = DataPipeline()
    df_full, tickers = dp.fetch_and_build_dataset()
    
    # 1. 학습/검증 (2000~2019) 및 테스트 (2020~2026) 데이터 분리
    train_data = df_full.loc["2000-01-01":"2019-12-31"]
    
    # 2. 5팩터를 '하나의 세트'로 통합 + 파생 팩터 결합 전략 후보군 구축
    ff5_set = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    
    strategy_factor_groups = [
        {"name": "5팩터 펀더멘털 통합 세트", "factors": ff5_set},
        {"name": "5팩터 + 모멘텀 세트", "factors": ff5_set + ['Mom_21D', 'Mom_63D']},
        {"name": "5팩터 + 변동성 세트", "factors": ff5_set + ['Vol_21D', 'Vol_63D']},
        {"name": "5팩터 + 기술적/RSI 세트", "factors": ff5_set + ['SMA_Ratio', 'RSI_14D']},
        {"name": "5팩터 + 올인원 종합 세트", "factors": ff5_set + ['Mom_63D', 'Vol_21D', 'SMA_Ratio', 'RSI_14D']}
    ]
    
    allocation_methods = ['equal', 'mvp', 'risk_parity', 'mvo']
    
    candidate_strategies = []
    idx = 1
    for group in strategy_factor_groups:
        for alloc in allocation_methods:
            candidate_strategies.append({
                "id": idx,
                "factors": group["factors"],
                "allocation": alloc,
                "name": f"전략_{idx} ({group['name']}, Alloc:{alloc})"
            })
            idx += 1
            
    print(f"-> 총 생성된 전략 후보군 개수: {len(candidate_strategies)}개")
    
    # 3. DSR 37% 최적 정지 이론 (Optimal Stopping Rule) 적용
    N = len(candidate_strategies)
    k = int(np.ceil(N * 0.37))
    
    backtester = EventDrivenBacktester()
    evaluation_logs = []
    
    def evaluate_strategy(strat_cfg):
        strat_factors = strat_cfg["factors"]
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(train_data[strat_factors]), 
            columns=strat_factors, 
            index=train_data.index
        )
        
        predictor = MLPredictor(strat_factors)
        trained_models = {}
        
        for t in tickers[:50]: # 유니버스 스캔 (속도 최적화를 위해 상위 50개)
            target_col = f"{t}_Target"
            if target_col in train_data.columns:
                fit_res = predictor.analyze_fama_remedy_p_values(X_train_scaled, train_data[target_col])
                trained_models[t] = fit_res
                
        pred_matrix = predictor.predict_universe(X_train_scaled, trained_models, list(trained_models.keys()))
        port_vals = backtester.run_absolute_top10_strategy(
            train_data, pred_matrix, list(trained_models.keys()), 
            rebalance_freq=21, alloc_method=strat_cfg["allocation"], slippage=0.0005
        )
        returns = port_vals.pct_change().dropna()
        dsr_score = RiskManager.calculate_dsr(returns, num_trials=N)
        return dsr_score, trained_models, scaler

    print(f"\n[1단계: 37% 표본 관찰 구간] 총 {k}개 전략 백테스트 및 DSR 산출...")
    sample_best_dsr = -1.0
    
    for i in range(k):
        strat = candidate_strategies[i]
        dsr, models, scaler = evaluate_strategy(strat)
        strat["dsr"] = dsr
        evaluation_logs.append({"stage": "Observation (37%)", "strategy": strat["name"], "dsr": dsr, "selected": False})
        if dsr > sample_best_dsr:
            sample_best_dsr = dsr

    print(f"-> 37% 관찰 구간 최고 DSR Cutoff: {sample_best_dsr:.4f}")
    
    print(f"\n[2단계: 선택 구간] Threshold 초과 시 즉시 최종 모델 채택...")
    selected_strategy = None
    selected_models = None
    selected_scaler = None
    
    for i in range(k, N):
        strat = candidate_strategies[i]
        dsr, models, scaler = evaluate_strategy(strat)
        strat["dsr"] = dsr
        
        if dsr > sample_best_dsr and selected_strategy is None:
            selected_strategy = strat
            selected_models = models
            selected_scaler = scaler
            evaluation_logs.append({"stage": "Selection", "strategy": strat["name"], "dsr": dsr, "selected": True})
            print(f"🎯 최적 모델 선발! [{strat['name']}] (DSR: {dsr:.4f} > Cutoff: {sample_best_dsr:.4f})")
        else:
            evaluation_logs.append({"stage": "Selection", "strategy": strat["name"], "dsr": dsr, "selected": False})

    if selected_strategy is None:
        selected_strategy = candidate_strategies[-1]
        _, selected_models, selected_scaler = evaluate_strategy(selected_strategy)

    # 4. 모델 직렬화 및 UI 출력용 JSON 파일 저장
    os.makedirs("model", exist_ok=True)
    stop_summary = {
        "total_candidates": N,
        "sample_size_37pct": k,
        "threshold_dsr": sample_best_dsr,
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
        
    print("\n💾 최적 정지 수행 기록(`model/dsr_optimal_stopping.json`) 및 모델 보관 완료.")

if __name__ == "__main__":
    main()