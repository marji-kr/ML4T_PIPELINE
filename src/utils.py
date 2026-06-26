import numpy as np
import pandas as pd

def calculate_performance_metrics(returns, benchmark_returns=None):
    """일별 수익률 시계열을 받아 이미지에 등장하는 14가지 평가지표를 연산합니다."""
    n_days = len(returns)
    years = n_days / 252
    
    # 1. 누적 수익률 & 최종 자산 기반 연평균 복리 수익률 (CAGR)
    cum_return = (1 + returns).cumprod().iloc[-1] - 1
    ann_return = (1 + cum_return) ** (1 / years) - 1
    
    # 2. 연율화 변동성 (Standard Deviation)
    ann_vol = returns.std() * np.sqrt(252)
    
    # 3. 최대 낙폭 (Maximum Drawdown)
    cum_wealth = (1 + returns).cumprod()
    running_max = cum_wealth.cummax()
    drawdown = (cum_wealth - running_max) / running_max
    mdd = drawdown.min()
    
    # 4. 최고의 해 / 최악의 해 계산 (252영업일 기준 rolling 복리 수익률)
    # 1년 단위 윈도우를 밀어가며 도출되는 연간 수익률 중 최댓값과 최솟값을 찾습니다.
    if n_days >= 252:
        annual_rolling = (1 + returns).rolling(252).apply(lambda x: np.prod(x)) - 1
        best_year = annual_rolling.max()
        worst_year = annual_rolling.min()
    else:
        best_year = cum_return
        worst_year = cum_return
    
    # 5. 위험 대비 효율성 지표 (Sharpe & Sortino)
    sharpe_ratio = ann_return / ann_vol if ann_vol != 0 else 0
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = ann_return / downside_vol if downside_vol != 0 else 0
    
    # 6. 월간 단위 리스크 관리 지표 환산 (Value at Risk - 5% 신뢰수준 월간 기준)
    # 일별 데이터를 월간(21영업일) 기준으로 샘플링하여 5% 하위 극단값을 추정합니다.
    monthly_returns = returns.rolling(21).apply(lambda x: np.prod(1 + x) - 1).dropna()
    if len(monthly_returns) > 0:
        # 역사적 VaR (Historical VaR)
        hist_var_5 = np.percentile(monthly_returns, 5)
        # 조건부 VaR (Conditional VaR / CVaR)
        cvar_5 = monthly_returns[monthly_returns <= hist_var_5].mean()
        # 분석적 VaR (Analytical VaR: 정규분포 가정공식 = 평균 - 1.645 * 표준편차)
        anal_var_5 = monthly_returns.mean() - 1.645 * monthly_returns.std()
    else:
        hist_var_5, cvar_5, anal_var_5 = 0.0, 0.0, 0.0

    # 7. 벤치마크 상대 지표 연산
    if benchmark_returns is not None:
        common_idx = returns.index.intersection(benchmark_returns.index)
        ret_c = returns.loc[common_idx]
        bench_c = benchmark_returns.loc[common_idx]
        
        # 상관관계 (Benchmark Correlation)
        corr_matrix = np.corrcoef(ret_c, bench_c)
        correlation = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
        
        # 베타 (Beta) & 결정계수 (R2)
        covariance = np.cov(ret_c, bench_c)[0][1]
        market_variance = np.var(bench_c)
        beta = covariance / market_variance if market_variance != 0 else 1.0
        r2 = correlation ** 2
        
        # 연율화 알파 (Alpha annualized)
        alpha = ann_return - (beta * ((1 + bench_c.mean())**252 - 1))
        
        # 상승/하방 캡처 비율 (Upside / Downside Capture Ratio)
        up_idx = bench_c > 0
        down_idx = bench_c < 0
        upside_capture = (ret_c[up_idx].mean() / bench_c[up_idx].mean()) if bench_c[up_idx].mean() != 0 else 1.0
        downside_capture = (ret_c[down_idx].mean() / bench_c[down_idx].mean()) if bench_c[down_idx].mean() != 0 else 1.0
    else:
        correlation = 1.0
        beta = 1.0
        alpha = 0.0
        r2 = 1.0
        upside_capture = 1.0
        downside_capture = 1.0

    return {
        'Start Balance': "", # main.py에서 동적으로 채움
        'End Balance': "",   # main.py에서 동적으로 채움
        'Annualized Return (CAGR)': f"{ann_return:.2%}",
        'Standard Deviation': f"{ann_vol:.2%}",
        'Best Year': f"{best_year:.2%}",
        'Worst Year': f"{worst_year:.2%}",
        'Maximum Drawdown': f"{mdd:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Sortino Ratio': f"{sortino_ratio:.2f}",
        'Benchmark Correlation': f"{correlation:.2f}",
        'Beta': f"{beta:.2f}",
        'Alpha (annualized)': f"{alpha:.2%}",
        'R2': f"{r2:.2%}",
        'Historical Value-at-Risk (5%)': f"{hist_var_5:.2%}",
        'Analytical Value-at-Risk (5%)': f"{anal_var_5:.2%}",
        'Conditional Value-at-Risk (5%)': f"{cvar_5:.2%}",
        'Upside Capture Ratio (%)': f"{upside_capture:.2%}",
        'Downside Capture Ratio (%)': f"{downside_capture:.2%}"
    }