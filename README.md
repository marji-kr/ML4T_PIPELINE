# 📊 S&P 500 다변화 팩터 전략 백테스트 & 최적 정지 탐색 시스템

본 프로젝트는 **S&P 500 유니버스**를 대상으로 Fama-French 5-Factor 및 파생 팩터를 활용한 머신러닝 기반 멀티-호라이즌(1일, 5일, 21일, 63일 뒤) 예측 포트폴리오 백테스트 파이프라인 및 시뮬레이터입니다.

다중 백테스트 시 발생하는 과적합(Data Snooping)을 방지하기 위해 **Sharpe Ratio 기반 37% 최적 정지 이론(Optimal Stopping Rule)** 알고리즘을 적용하여 최적의 예측 기간, 팩터 그룹 및 자산배분 기법을 자동으로 탐색 및 선발합니다.

---

## 📄 목차 (Table of Contents)

1. [전략 명세 및 구조 (Strategy Breakdown)](#1-전략-명세-및-구조-strategy-breakdown)
2. [수수료 및 슬리피지 모델링 (Trading Costs)](#2-수수료-및-슬리피지-모델링-trading-costs)
3. [포트폴리오 평가지표 상세 수식 (Performance Metrics)](#3-포트폴리오-평가지표-상세-수식-performance-metrics)
4. [모듈별 상세 설명 및 연동 파이프라인 (Module Details & Pipeline)](#4-모듈별-상세-설명-및-연동-파이프라인-module-details--pipeline)
5. [설치 및 실행 방법 (Quick Start)](#5-설치-및-실행-방법-quick-start)

---

## 1. 전략 명세 및 구조 (Strategy Breakdown)

본 시스템은 **`[예측 타겟 기간] × [팩터 조합 그룹] × [자산배분 기법]`**의 조합으로 총 **64가지 전략 후보군 ($4 \times 5 \times 4 = 64$)**을 생성하여 탐색합니다.

### ① 예측 타겟 기간 (Prediction Horizons)
각 종목의 미래 수익률 $Y$를 몇 영업일 뒤 기준으로 예측할지 설정합니다. 백테스트 시 해당 horizon 주기에 맞춰 리밸런싱(`rebalance_freq`)이 함께 진행됩니다.
* **`1D` (1일 뒤 예측)**: 일별 단기 기대수익률 예측 및 매일 리밸런싱
* **`5D` (5일 뒤 예측)**: 주간 기대수익률 예측 및 1주일 주기(5영업일) 리밸런싱
* **`21D` (21일 뒤 예측)**: 월간 기대수익률 예측 및 1달 주기(21영업일) 리밸런싱
* **`63D` (63일 뒤 예측)**: 분기 기대수익률 예측 및 1분기 주기(63영업일) 리밸런싱

### ② 팩터 요소 그룹 (Factor Sets)
* **Fama-French 5-Factor 기본 세트**:
  - `Mkt-RF`: 시장 위험 프리미엄 (Market Excess Return)
  - `SMB` (Small Minus Big): 규모 팩터 (소형주 - 대형주)
  - `HML` (High Minus Low): 가치 팩터 (고장부가치주 - 저장부가치주)
  - `RMW` (Robust Minus Weak): 수익성 팩터 (고수익성 - 저수익성)
  - `CMA` (Conservative Minus Aggressive): 투자 팩터 (보수적 투자 - 공격적 투자)
* **5팩터 펀더멘털 세트**: `['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']`
* **5팩터 + 모멘텀 세트**: 5팩터 + `['Mom_21D', 'Mom_63D']` (21일/63일 누적 수익률)
* **5팩터 + 변동성 세트**: 5팩터 + `['Vol_21D', 'Vol_63D']` (21일/63일 연율화 표준편차)
* **5팩터 + 기술적(RSI/SMA) 세트**: 5팩터 + `['SMA_Ratio', 'RSI_14D']` (이동평균 비율 및 14일 RSI)
* **5팩터 + 종합 올인원 세트**: 5팩터 + `['Mom_63D', 'Vol_21D', 'SMA_Ratio', 'RSI_14D']`

### ③ 자산배분 기법 (Allocation Methods)
예측된 절대 수익률 상위 10개 종목(Top 10 Absolute)을 대상으로 비중을 산출합니다.
* **`equal` (동일 비중)**: 상위 10개 종목을 동일하게 10%씩 배분 ($1/N$).
* **`mvp` (Minimum Variance Portfolio)**: 공분산 행렬 $\Sigma$에 기반하여 포트폴리오 전체 변동성($w^T \Sigma w$)을 최소화.
* **`risk_parity` (위험 기여도 평등화)**: 모든 구성 종목의 위험 기여도(Risk Contribution)가 균등해지도록 최적화.
* **`mvo` (Mean-Variance Optimization)**: 위험회피계수 $\lambda=3.0$ 적용 하에 기대수익률 대비 위험을 고려한 마코위츠 최적화.

---

## 2. 수수료 및 슬리피지 모델링 (Trading Costs)

이벤트 기반 백테스터(`backtest_engine.py`)에서는 현실적인 매매 마찰 비용을 반영합니다.

$$\text{Total Cost Rate} = \text{Commission Rate} + \text{Slippage Rate}$$

* **거래 수수료 (`commission`)**: 기본값 **`0.001` (0.10% 또는 10 bps)**
* **슬리피지 계수 (`slippage`)**: 기본값 **`0.0005` (0.05% 또는 5 bps)**
* **총 실행 비용률 (`total_execution_cost_rate`)**: **`0.0015` (0.15% 또는 15 bps)**

### 실행 반영 로직
단순히 전체 포트폴리오 자산에서 일괄 차감하는 방식이 아니라, **리밸런싱 시 발생하는 실제 거래 대금 규모(`trade_value`)의 절대값**에 비례하여 차감됩니다.

$$
\text{TradeValue}_i = (\text{TargetShares}_i - \text{CurrentShares}_i) \times P_{i, \text{open}}
$$
$$
\text{TradeCost}_i = |\text{TradeValue}_i| \times (\text{Commission} + \text{Slippage})
$$
$$
\text{Cash}_{\text{new}} = \text{Cash}_{\text{old}} - \sum (\text{TradeValue}_i + \text{TradeCost}_i)
$$

* 체결가는 예측 당일 종가가 아닌 **익일 시가(`tomorrow_open_prices`)**를 사용하여 편향을 제거합니다.

---

## 3. 포트폴리오 평가지표 상세 수식 (Performance Metrics)

`web_app.py`에서 계산하는 18가지 평가 지표들의 정의 및 수학적 수식입니다. (일별 수익률 $R_t$, 벤치마크 수익률 $R_{b,t}$, 초기 자산 $V_0$, 최종 자산 $V_T$, 연간 거래일 $N=252$)

### 1) CAGR (Compound Annual Growth Rate, 연평균 복리 수익률)
$$\text{CAGR} = \left(\frac{V_T}{V_0}\right)^{\frac{1}{\text{Years}}} - 1, \quad \text{Years} = \frac{\text{Total Days}}{252}$$

### 2) Standard Deviation (연율화 표준편차 / 변동성)
$$\sigma_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252} = \sqrt{\frac{1}{T-1} \sum_{t=1}^{T} (R_t - \bar{R})^2} \times \sqrt{252}$$

### 3) Sharpe Ratio (샤프 지수)
$$\text{Sharpe Ratio} = \frac{\bar{R}_{\text{daily}}}{\sigma_{\text{daily}}} \times \sqrt{252}$$

### 4) Sortino Ratio (소티노 지수)
하방 리스크(음의 수익률)만을 위험으로 정의한 지수입니다.
$$\text{Sortino Ratio} = \frac{\text{CAGR}}{\sigma_{\text{downside}}}, \quad \sigma_{\text{downside}} = \sqrt{\frac{1}{T_d} \sum_{R_t < 0} R_t^2} \times \sqrt{252}$$

### 5) Maximum Drawdown (MDD, 최고점 대비 최대 낙폭)
$$\text{Drawdown}_t = \frac{V_t - \max_{\tau \le t}(V_\tau)}{\max_{\tau \le t}(V_\tau)}, \quad \text{MDD} = \min_{t} (\text{Drawdown}_t)$$

### 6) Information Ratio (IR, 정보 비율)
벤치마크 대비 초과수익률의 안정성을 측정합니다.
$$\text{IR} = \frac{\overline{(R_t - R_{b,t})} \times 252}{\text{Tracking Error}}, \quad \text{Tracking Error} = \sigma_{(R_t - R_{b,t})} \times \sqrt{252}$$

### 7) Alpha ($\alpha$, 젠센의 알파) & Beta ($\beta$, 베타)
OLS 회귀분석 모델 $R_t = \alpha_{\text{daily}} + \beta R_{b,t} + \epsilon_t$ 을 통해 추정합니다.
$$\beta = \frac{\text{Cov}(R_t, R_{b,t})}{\text{Var}(R_{b,t})}, \quad \text{Alpha (연간화)} = \alpha_{\text{daily}} \times 252$$

### 8) Benchmark Correlation ($r$) & $R^2$ (결정계수)
$$r = \frac{\text{Cov}(R_t, R_{b,t})}{\sigma_R \cdot \sigma_{R_b}}, \quad R^2 = r^2$$

### 9) Value-at-Risk (VaR 5%, 손실 위험 값)
* **Historical VaR (5%)**: 수익률 분포의 하위 5% 분위수 ($Q_{0.05}(R_t)$)
* **Analytical VaR (5%)**: 정규분포 가정 하의 손실 한계값 ($\bar{R} + \sigma_R \cdot Z_{0.05}$)
* **Conditional VaR (5% / CVaR, Expected Shortfall)**: VaR을 초과하는 극단적 손실 구간의 평균값 ($E[R_t \mid R_t \le \text{VaR}_{0.05}]$)

### 10) Upside / Downside Capture Ratio (상승/하강 포착 비율)
* **Upside Capture**: 벤치마크 상승일($R_{b,t} > 0$) 동안의 전략 평균 수익률 비율
* **Downside Capture**: 벤치마크 하락일($R_{b,t} < 0$) 동안의 전략 평균 수익률 비율

---

## 4. 모듈별 상세 설명 및 연동 파이프라인 (Module Details & Pipeline)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         1. DataPipeline                                  │
│   • S&P 500 주가 수집 및 1D, 5D, 21D, 63D 타겟 변수 (Target_hD) 생성     │
│   • Fama-French 5-Factor & 기술적 파생 팩터 결합                         │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         2. MLPredictor                                   │
│   • OLS + HC3 (White) Robust Covariance 회귀모델 피팅                    │
│   • 각 호라이즌(Horizon)에 맞춘 유니버스 기대수익률 예측 행렬 산출       │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         3. RiskManager                                   │
│   • 자산배분 비중 산출 (Equal, MVP, Risk Parity, MVO)                    │
│   • 연율화 Sharpe Ratio 산출                                             │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      4. EventDrivenBacktester                            │
│   • Absolute Top 10 종목 선정, 익일 시가 체결                             │
│   • 수수료 0.1% + 슬리피지 0.05% 매매대금 비례 차감                      │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   5. main.py & web_app.py                                │
│   • 37% Optimal Stopping Rule 적용 최적 모델 선택 및 저장                │
│   • Streamlit 대시보드를 통해 성과 비교 및 시각화                        │
└──────────────────────────────────────────────────────────────────────────┘
```

### 모듈 역할 요약
1. **`data_pipeline.py`**: 주가 데이터 수집 및 1일/5일/21일/63일 타겟 변수(`Target_1D`, `Target_5D`, `Target_21D`, `Target_63D`) 생성.
2. **`ml_predictor.py`**: HC3 Robust OLS 기반 미래 기대수익률 행렬(`pred_matrix`) 계산.
3. **`risk_manager.py`**: 4가지 최적화 자산배분 및 연율화 샤프지수 산출.
4. **`backtest_engine.py`**: 리밸런싱 주기(`horizon`) 및 슬리피지/수수료를 적용한 포트폴리오 가치 추적.
5. **`cv_splitter.py`**: Purged & Embargo 시계열 교차검증 지원.
6. **`main.py`**: 총 64개 전략 후보군에 대한 **37% Rule 탐색**을 진행하여 샤프지수 Cutoff 초과 전략을 최종 모델로 자동 저장.
7. **`web_app.py`**: 최적 모델 분석 결과 시각화 및 백테스트 실행 패널.

---

## 5. 설치 및 실행 방법 (Quick Start)

### 1) 환경 구축
```bash
pip install numpy pandas yfinance pandas-datareader statsmodels scipy scikit-learn streamlit plotly
```

### 2) 파이프라인 수행 및 최적 모델 채택
```bash
python main.py
```

### 3) 대시보드 시뮬레이터 실행
```bash
streamlit run web_app.py
