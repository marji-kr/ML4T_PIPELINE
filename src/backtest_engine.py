import numpy as np
import pandas as pd
from risk_manager import RiskManager

class EventDrivenBacktester:
    def __init__(self, start_cash=1000000.0, commission=0.001):
        self.start_cash = start_cash
        self.commission = commission

    def run_absolute_top10_strategy(self, test_df, pred_df, tickers, rebalance_freq=21, alloc_method="equal", slippage=0.0005):
        cash = self.start_cash
        current_portfolio = {t: 0.0 for t in tickers}
        portfolio_history = []
        
        for i in range(len(test_df)):
            stock_value = 0.0
            for t in tickers:
                if t in test_df.columns:
                    stock_value += current_portfolio[t] * test_df[t].iloc[i]
            total_portfolio_value = cash + stock_value
            portfolio_history.append(total_portfolio_value)
            
            if i % rebalance_freq == 0 and i < len(test_df) - 1:
                tomorrow_open_prices = {t: test_df[t].iloc[i+1] for t in tickers if t in test_df.columns}
                today_preds = pred_df.iloc[i]
                
                abs_preds = today_preds.abs().sort_values(ascending=False)
                top10_tickers = list(abs_preds.head(10).index)
                
                lookback_window = 60
                start_idx = max(0, i - lookback_window)
                hist_returns = test_df[[f"{t}_Return" for t in top10_tickers if f"{t}_Return" in test_df.columns]].iloc[start_idx:i+1]
                
                target_weights = {t: 0.0 for t in tickers}
                
                if len(hist_returns) > 10 and alloc_method != "equal":
                    raw_weights = RiskManager.get_portfolio_weights(hist_returns, method=alloc_method)
                    for idx, t in enumerate(top10_tickers):
                        pred_sign = np.sign(today_preds[t])
                        target_weights[t] = raw_weights[idx] * (1.0 if pred_sign >= 0 else -1.0)
                else:
                    for t in top10_tickers:
                        pred_sign = today_preds[t]
                        target_weights[t] = 0.10 if pred_sign > 0 else -0.10
                
                total_execution_cost_rate = self.commission + slippage
                
                for t in tickers:
                    if t not in tomorrow_open_prices or tomorrow_open_prices[t] <= 0:
                        continue
                    target_shares = (total_portfolio_value * target_weights[t]) / tomorrow_open_prices[t]
                    shares_to_trade = target_shares - current_portfolio[t]
                    
                    if shares_to_trade != 0:
                        trade_value = shares_to_trade * tomorrow_open_prices[t]
                        trade_cost = abs(trade_value) * total_execution_cost_rate
                        
                        cash -= (trade_value + trade_cost)
                        current_portfolio[t] = target_shares
                        
        return pd.Series(portfolio_history, index=test_df.index)