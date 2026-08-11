import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
import scipy.stats as stats
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
import yfinance as yf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(CURRENT_DIR)

st.set_page_config(page_title="포트폴리오 백테스트 시뮬레이터", layout="wide")

@st.cache_resource
def load_artifacts():
    m_path = os.path.join(PROJECT_ROOT, "model", "universe_models.pkl")
    s_path = os.path.join(PROJECT_ROOT, "model", "scaler.pkl")
    f_path = os.path.join(PROJECT_ROOT, "model", "selected_factors.pkl")
    
    if not os.path.exists(m_path) or not os.path.exists(s_path):
        return None, None, None
        
    with open(m_path, 'rb') as m_f, open(s_path, 'rb') as s_f:
        models = pickle.load(m_f)
        scaler = pickle.load(s_f)
        
    selected_factors = None
    if os.path.exists(f_path):
        with open(f_path, 'rb') as f_f:
            selected_factors = pickle.load(f_f)
            
    return models, scaler, selected_factors

models, scaler, selected_factors = load_artifacts()

def calculate_asset_metrics(returns_series, bench_returns_series, start_cash, final_val):
    """ 18가지 상세 포트폴리오 평가지표 전체 계산 """
    metrics = {}
    metrics['Start Balance (초기 자산)'] = start_cash
    metrics['End Balance (최종 자산)'] = final_val
    
    strat_cum = (1 + returns_series).cumprod()
    years = len(returns_series) / 252.0 if len(returns_series) > 0 else 1.0
    cagr = (final_val / start_cash) ** (1 / years) - 1 if years > 0 else 0.0
    
    metrics['Annualized Return (CAGR)'] = cagr
    vol = returns_series.std() * np.sqrt(252)
    metrics['Standard Deviation (표준편차)'] = vol
    
    # 1. Information Ratio (IR)
    diff_returns = returns_series - bench_returns_series
    tracking_error = diff_returns.std() * np.sqrt(252)
    metrics['IR (Information Ratio)'] = (diff_returns.mean() * 252) / tracking_error if tracking_error != 0 else 0.0
    
    # 2. Best Year & Worst Year
    yearly_returns = returns_series.groupby(returns_series.index.year).apply(lambda x: (1 + x).prod() - 1)
    metrics['Best Year (최고의 해)'] = yearly_returns.max() if len(yearly_returns) > 0 else 0.0
    metrics['Worst Year (최악의 해)'] = yearly_returns.min() if len(yearly_returns) > 0 else 0.0
    
    # 3. Maximum Drawdown (MDD)
    running_max = strat_cum.cummax()
    drawdown = (strat_cum - running_max) / running_max
    mdd = drawdown.min()
    metrics['Maximum Drawdown (MDD)'] = mdd
    
    # 4. Sharpe & Sortino Ratio
    metrics['Sharpe Ratio (샤프 지수)'] = (returns_series.mean() / returns_series.std()) * np.sqrt(252) if returns_series.std() != 0 else 0.0
    downside_std = returns_series[returns_series < 0].std() * np.sqrt(252)
    metrics['Sortino Ratio (소티노 지수)'] = cagr / downside_std if downside_std != 0 else 0.0
    
    # 5. Benchmark Correlation, Beta, Alpha, R^2
    corr = returns_series.corr(bench_returns_series)
    metrics['Benchmark Correlation (상관관계)'] = corr if not np.isnan(corr) else 0.0
    
    X_reg = sm.add_constant(bench_returns_series)
    reg = sm.OLS(returns_series, X_reg).fit()
    metrics['Beta (베타)'] = reg.params.iloc[1] if len(reg.params) > 1 else 0.0
    metrics['Alpha (알파 - 연간화)'] = reg.params.iloc[0] * 252 if len(reg.params) > 0 else 0.0
    metrics['R2 (결정계수)'] = reg.rsquared
    
    # 6. Historical, Analytical, Conditional VaR (5%)
    hist_var_5 = returns_series.quantile(0.05)
    metrics['Historical Value-at-Risk (5%)'] = hist_var_5
    
    mean_ret = returns_series.mean()
    std_ret = returns_series.std()
    analytical_var_5 = mean_ret + std_ret * stats.norm.ppf(0.05)
    metrics['Analytical Value-at-Risk (5%)'] = analytical_var_5
    
    cvar_5 = returns_series[returns_series <= hist_var_5].mean()
    metrics['Conditional Value-at-Risk (5%)'] = cvar_5 if not np.isnan(cvar_5) else hist_var_5
    
    # 7. Upside & Downside Capture Ratio (%)
    up_periods = bench_returns_series > 0
    down_periods = bench_returns_series < 0
    
    up_capture = (returns_series[up_periods].mean() / bench_returns_series[up_periods].mean()) * 100 if bench_returns_series[up_periods].mean() != 0 else 100.0
    down_capture = (returns_series[down_periods].mean() / bench_returns_series[down_periods].mean()) * 100 if bench_returns_series[down_periods].mean() != 0 else 100.0
    
    metrics['Upside Capture Ratio (%)'] = up_capture / 100.0
    metrics['Downside Capture Ratio (%)'] = down_capture / 100.0
    
    return pd.Series(metrics)

# -----------------------------------------------------------------------------
# 🎯 Sharpe Ratio 기반 최적 정지 이론 (37% Rule) 대시보드
# -----------------------------------------------------------------------------
st.title("📊 S&P 500 다변화 전략 백테스트 시뮬레이터")

dsr_json_path = os.path.join(PROJECT_ROOT, "model", "dsr_optimal_stopping.json")
sel_strat = {}

if os.path.exists(dsr_json_path):
    with open(dsr_json_path, 'r', encoding='utf-8') as f:
        dsr_info = json.load(f)
        
    with st.expander("🔍 샤프지수 기반 37% 최적 정지 이론 모델 선택 결과 및 상세 정보", expanded=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("총 후보 전략 수", f"{dsr_info.get('total_candidates', 0)}개")
        m2.metric("37% 관찰 표본 수", f"{dsr_info.get('sample_size_37pct', 0)}개")
        m3.metric("표본 Cutoff (Sharpe)", f"{dsr_info.get('threshold_sharpe', 0.0):.4f}")
        
        sel_strat = dsr_info.get("selected_strategy", {})
        m4.metric("선발 전략 Sharpe Ratio", f"{sel_strat.get('sharpe_ratio', 0.0):.4f}")
        
        # 선발 전략에 대한 세부 명세 표시
        st.markdown("---")
        st.subheader("📌 최종 선발 전략 상세 명세")
        c1, c2, c3 = st.columns(3)
        c1.info(f"**⏱️ 예측 타겟 기간**: `{sel_strat.get('horizon', 'N/A')}일 뒤` 예측")
        c2.info(f"**⚖️ 자산배분 기법**: `{sel_strat.get('allocation', 'N/A')}`")
        factors_used = ", ".join(sel_strat.get('factors', []))
        c3.info(f"**🧬 사용 팩터 세트**: `{factors_used}`")
        
        logs = dsr_info.get("logs", [])
        if logs:
            logs_df = pd.DataFrame(logs)
            fig_dsr = px.bar(
                logs_df, x="strategy", y="sharpe_ratio", color="selected",
                color_discrete_map={True: "#2ca02c", False: "#1f77b4"},
                labels={"strategy": "전략 후보군", "sharpe_ratio": "Sharpe Ratio (샤프지수)", "selected": "최종 선발 여부"},
                title="전략 후보군별 샤프지수 및 37% 최적 정지 탐색 기록"
            )
            fig_dsr.add_hline(
                y=dsr_info.get('threshold_sharpe', 0.0), line_dash="dash", line_color="red",
                annotation_text=f"37% Threshold ({dsr_info.get('threshold_sharpe', 0.0):.4f})"
            )
            fig_dsr.update_layout(xaxis_tickangle=-45, height=360, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_dsr, width='stretch')
else:
    st.warning("⚠️ `model/dsr_optimal_stopping.json` 파일이 없습니다. `python main.py`를 먼저 실행해 주세요.")

st.markdown("---")

# -----------------------------------------------------------------------------
# 메인 백테스트 실행 및 결과 시각화
# -----------------------------------------------------------------------------
if models is not None and scaler is not None:
    from data_pipeline import DataPipeline
    from backtest_engine import EventDrivenBacktester
    from ml_predictor import MLPredictor
    
    st.sidebar.header("⚙️ 백테스트 조건 설정")
    start_year = st.sidebar.slider("시작 년도", min_value=2000, max_value=2026, value=2020)
    end_year = st.sidebar.slider("종료 년도", min_value=start_year, max_value=2026, value=2026)
    
    start_cash = st.sidebar.number_input("초기 자본 ($)", min_value=1000, max_value=10000000, value=10000, step=1000)
    commission = st.sidebar.slider("수수료율 (Commission)", 0.0, 0.01, 0.001, step=0.0005, format="%.4f")
    impact = st.sidebar.slider("슬리피지 계수 (Slippage)", 0.0, 0.5, 0.1, step=0.05)

    @st.cache_data
    def get_backtest_base_data(start, end):
        dp = DataPipeline(data_dir=os.path.join(PROJECT_ROOT, "data"))
        df_combined, tickers = dp.fetch_and_build_dataset()
        qqq = yf.download("QQQ", start=f"{start}-01-01", end=f"{end}-06-30", progress=False)
        
        if isinstance(qqq.columns, pd.MultiIndex):
            qqq_close = qqq['Adj Close'] if 'Adj Close' in qqq.columns.get_level_values(0) else qqq['Close']
            qqq_close = qqq_close.iloc[:, 0]
        else:
            qqq_close = qqq['Adj Close'] if 'Adj Close' in qqq.columns else qqq['Close']
            
        qqq_returns = qqq_close.pct_change(1)
        qqq_returns.name = "QQQ_Return"
        return df_combined.join(qqq_returns, how='inner'), tickers

    df_combined, tickers = get_backtest_base_data(start_year, end_year)
    test_data = df_combined.loc[f"{start_year}-01-01":f"{end_year}-06-30"]
    
    if selected_factors is not None:
        fama_features = selected_factors
    elif hasattr(scaler, 'feature_names_in_'):
        fama_features = list(scaler.feature_names_in_)
    else:
        fama_features = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    
    fama_features = [f for f in fama_features if f in test_data.columns]
    
    active_tickers = list(models.keys())
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(test_data[fama_features]), 
        columns=fama_features, 
        index=test_data.index
    )

    predictor = MLPredictor(fama_features)
    test_pred_matrix = predictor.predict_universe(X_test_scaled, models, active_tickers)
    
    selected_alloc = sel_strat.get('allocation', 'equal')
    selected_horizon = sel_strat.get('horizon', 21)
    
    backtester = EventDrivenBacktester(start_cash=start_cash, commission=commission)
    portfolio_values = backtester.run_absolute_top10_strategy(
        test_data, test_pred_matrix, active_tickers, 
        rebalance_freq=selected_horizon,
        alloc_method=selected_alloc,
        slippage=impact
    )
    
    spy_returns = test_data['Mkt-RF']
    spy_values = start_cash * (1 + spy_returns).cumprod()
    qqq_values = start_cash * (1 + test_data['QQQ_Return']).cumprod()
    
    strat_returns = portfolio_values.pct_change().dropna()
    spy_returns_m = spy_returns.loc[strat_returns.index]
    qqq_returns_m = test_data['QQQ_Return'].loc[strat_returns.index]

    col1, col2 = st.columns([1.7, 1.3])
    
    # 📈 1. 누적 자산 성장 비교 그래프
    with col1:
        st.write("### 📈 누적 자산 성장 추이 (Asset Growth)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=portfolio_values.index, y=portfolio_values, name="ML4T 다변화 전략", line=dict(color='#1f77b4', width=3)))
        fig.add_trace(go.Scatter(x=spy_values.index, y=spy_values, name="S&P 500 (SPY)", line=dict(color='#7f7f7f', width=1.5, dash='dash')))
        fig.add_trace(go.Scatter(x=qqq_values.index, y=qqq_values, name="NASDAQ 100 (QQQ)", line=dict(color='#2ca02c', width=1.5, dash='dot')))
        
        fig.update_layout(
            xaxis_title="연도 (Date)", yaxis_title="자산 가치 ($)", yaxis_type="log",
            yaxis=dict(tickformat="$~s"), hovermode="x unified", margin=dict(l=10, r=10, t=15, b=10), height=600,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, width='stretch')
        
    # 📋 2. 18가지 포트폴리오 평가지표 비교 표
    with col2:
        st.write("### 📋 18가지 상세 포트폴리오 평가 비교 표")
        strat_metrics = calculate_asset_metrics(strat_returns, spy_returns_m, start_cash, portfolio_values.iloc[-1])
        spy_metrics = calculate_asset_metrics(spy_returns_m, spy_returns_m, start_cash, spy_values.iloc[-1])
        qqq_metrics = calculate_asset_metrics(qqq_returns_m, spy_returns_m, start_cash, qqq_values.iloc[-1])
        
        comparison_table = pd.DataFrame({"ML4T 전략": strat_metrics, "S&P 500": spy_metrics, "NASDAQ 100": qqq_metrics})
        formatted_table = pd.DataFrame(index=comparison_table.index, columns=comparison_table.columns)
        
        for col in comparison_table.columns:
            for idx in comparison_table.index:
                val = comparison_table.loc[idx, col]
                if "Balance" in idx or "자산" in idx:
                    formatted_table.loc[idx, col] = f"${val:,.2f}"
                elif any(k in idx for k in ["Ratio", "Beta", "Correlation", "R2", "IR", "지수", "비율", "베타", "결정계수", "상관관계"]):
                    formatted_table.loc[idx, col] = f"{val:.4f}"
                else:
                    formatted_table.loc[idx, col] = f"{val * 100:.2f}%"
                    
        st.dataframe(formatted_table, height=600, width='stretch')