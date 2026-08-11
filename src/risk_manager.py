import numpy as np
import pandas as pd
from scipy.optimize import minimize

class RiskManager:
    @staticmethod
    def get_portfolio_weights(returns_df, method="equal"):
        n = returns_df.shape[1]
        if n == 0: 
            return np.array([])
        
        cov = returns_df.cov().values
        mu = returns_df.mean().values
        
        if method == "equal":
            return np.ones(n) / n
            
        elif method == "mvp":
            def mvp_obj(w): 
                return w.T @ cov @ w
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
            bounds = [(0, 1) for _ in range(n)]
            res = minimize(mvp_obj, np.ones(n)/n, bounds=bounds, constraints=cons)
            return res.x if res.success else np.ones(n)/n
            
        elif method == "risk_parity":
            def rp_obj(w):
                port_vol = np.sqrt(w.T @ cov @ w)
                marginal_contrib = cov @ w
                risk_contrib = w * marginal_contrib / (port_vol + 1e-8)
                target_risk = port_vol / n
                return np.sum((risk_contrib - target_risk)**2)
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
            bounds = [(0, 1) for _ in range(n)]
            res = minimize(rp_obj, np.ones(n)/n, bounds=bounds, constraints=cons)
            return res.x if res.success else np.ones(n)/n
            
        elif method == "mvo":
            lmbda = 3.0
            def mvo_obj(w): 
                return -(w.T @ mu - (lmbda / 2) * (w.T @ cov @ w))
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
            bounds = [(0, 1) for _ in range(n)]
            res = minimize(mvo_obj, np.ones(n)/n, bounds=bounds, constraints=cons)
            return res.x if res.success else np.ones(n)/n
            
        return np.ones(n) / n

    @staticmethod
    def calculate_sharpe_ratio(returns):
        """
        연율화 샤프지수 (Annualized Sharpe Ratio) 산출
        """
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        sr = (returns.mean() / returns.std()) * np.sqrt(252)
        return float(sr)