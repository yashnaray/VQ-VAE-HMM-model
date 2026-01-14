import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BacktestResult:
    returns: np.ndarray
    positions: np.ndarray
    trades: np.ndarray
    metrics: Dict[str, float]
    equity_curve: np.ndarray
    drawdowns: np.ndarray


class Backtester:
    def __init__(self, initial_capital: float = 100000, tx_cost: float = 0.001,
                 slippage: float = 0.0005, max_leverage: float = 1.0):
        self.initial_capital = initial_capital
        self.tx_cost = tx_cost
        self.slippage = slippage
        self.max_leverage = max_leverage
        
    def run(self, model, vae_hmm, data: torch.Tensor, prices: np.ndarray,
            returns: np.ndarray, rebalance_freq: int = 1) -> BacktestResult:
        
        n_periods = len(prices)
        n_assets = prices.shape[1]
        
        positions = np.zeros((n_periods, n_assets))
        trades = np.zeros((n_periods, n_assets))
        portfolio_values = np.zeros(n_periods)
        portfolio_values[0] = self.initial_capital
        
        model.eval()
        vae_hmm.eval()
        
        with torch.no_grad():
            for t in range(1, n_periods):
                if t % rebalance_freq == 0 and t > 20:
                    x_window = data[:, :, max(0, t-20):t]
                    regime_probs = torch.softmax(vae_hmm.encode(x_window), dim=1)
                    
                    target_weights = model(regime_probs).cpu().numpy().flatten()
                    
                    if target_weights.sum() > self.max_leverage: target_weights = target_weights / target_weights.sum() * self.max_leverage
                    
                    target_positions = target_weights * portfolio_values[t-1] / prices[t]
                    
                    trades[t] = target_positions - positions[t-1]
                    positions[t] = target_positions
                else:
                    positions[t] = positions[t-1]
                
                trade_value = np.abs(trades[t] * prices[t]).sum()
                costs = trade_value * (self.tx_cost + self.slippage)
                
                position_value = (positions[t] * prices[t]).sum()
                cash = portfolio_values[t-1] - (positions[t-1] * prices[t-1]).sum()
                portfolio_values[t] = position_value + cash - costs
        
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        metrics = self._calculate_metrics(portfolio_returns, portfolio_values)
        
        drawdowns = self._calculate_drawdowns(portfolio_values)
        
        return BacktestResult(
            returns=portfolio_returns,
            positions=positions,
            trades=trades,
            metrics=metrics,
            equity_curve=portfolio_values,
            drawdowns=drawdowns
        )
    
    def _calculate_metrics(self, returns: np.ndarray, equity: np.ndarray) -> Dict[str, float]:
        total_return = (equity[-1] - equity[0]) / equity[0]
        ann_return = (1 + total_return) ** (252 / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        downside = returns[returns < 0]
        downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-8
        sortino = ann_return / downside_std
        
        cummax = np.maximum.accumulate(equity)
        drawdowns = (equity - cummax) / cummax
        max_dd = drawdowns.min()
        
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
        win_rate = (returns > 0).sum() / len(returns)
        
        return {
            'total_return': total_return,
            'annual_return': ann_return,
            'annual_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'final_value': equity[-1]
        }
    
    def _calculate_drawdowns(self, equity: np.ndarray) -> np.ndarray:
        cummax = np.maximum.accumulate(equity)
        return (equity - cummax) / cummax


class WalkForwardBacktest:
    def __init__(self, train_window: int = 252, test_window: int = 21,
                 retrain_freq: int = 21, backtester: Optional[Backtester] = None):
        self.train_window = train_window
        self.test_window = test_window
        self.retrain_freq = retrain_freq
        self.backtester = backtester or Backtester()
        
    def run(self, model, vae_hmm, train_fn, data: torch.Tensor,
            prices: np.ndarray, returns: np.ndarray) -> List[BacktestResult]:
        
        results = []
        n_periods = len(prices)
        
        for start in range(0, n_periods - self.train_window - self.test_window, self.retrain_freq):
            train_end = start + self.train_window
            test_end = min(train_end + self.test_window, n_periods)
            
            train_data = data[:, :, start:train_end]
            train_fn(model, vae_hmm, train_data)
            
            test_data = data[:, :, train_end:test_end]
            test_prices = prices[train_end:test_end]
            test_returns = returns[train_end:test_end]
            
            result = self.backtester.run(model, vae_hmm, test_data, 
                                        test_prices, test_returns)
            results.append(result)
        
        return results


class RegimeBacktest:
    def __init__(self, backtester: Optional[Backtester] = None):
        self.backtester = backtester or Backtester()
        
    def run(self, model, vae_hmm, data: torch.Tensor, prices: np.ndarray,
            returns: np.ndarray, K: int) -> Dict[int, BacktestResult]:
        
        # regimes
        with torch.no_grad():
            regime_probs = torch.softmax(vae_hmm.encode(data), dim=1)
            regimes = regime_probs.argmax(dim=1).squeeze().cpu().numpy()
        
        results = {}
        for k in range(K):
            mask = regimes == k
            if mask.sum() < 20:
                continue
            
            regime_data = data[:, :, mask]
            regime_prices = prices[mask]
            regime_returns = returns[mask]
            
            result = self.backtester.run(model, vae_hmm, regime_data,
                                        regime_prices, regime_returns)
            results[k] = result
        
        return results


def compare_strategies(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    metrics_list = []
    for name, result in results.items():
        metrics = result.metrics.copy()
        metrics['strategy'] = name
        metrics_list.append(metrics)
    
    return pd.DataFrame(metrics_list).set_index('strategy')


def plot_results(result: BacktestResult, title: str = "Backtest Results"):
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        axes[0].plot(result.equity_curve)
        axes[0].set_title(f'{title} - Equity Curve')
        axes[0].set_ylabel('Portfolio Value')
        axes[0].grid(True)
        
        axes[1].fill_between(range(len(result.drawdowns)), result.drawdowns, 0, alpha=0.3)
        axes[1].set_title('Drawdown')
        axes[1].set_ylabel('Drawdown %')
        axes[1].grid(True)
        
        axes[2].hist(result.returns, bins=50, alpha=0.7)
        axes[2].set_title('Returns Distribution')
        axes[2].set_xlabel('Return')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True)
        
        plt.tight_layout()
        return fig
    except ImportError:
        print("matplotlib not available for plotting")
        return None
