"""
Backtesting framework for portfolio strategies.

This module provides tools to evaluate portfolio strategies with realistic
market conditions including transaction costs, slippage, and regime analysis.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BacktestResult:
    """Container for backtest results and performance metrics."""
    
    equity_curve: np.ndarray
    returns: np.ndarray
    positions: np.ndarray
    trades: List[Dict]
    metrics: Dict[str, float]
    regime_history: Optional[np.ndarray] = None
    
    def summary(self) -> pd.DataFrame:
        """Return a formatted summary of key metrics."""
        return pd.DataFrame([self.metrics]).T.rename(columns={0: 'Value'})
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        return pd.DataFrame({
            'equity': self.equity_curve,
            'returns': self.returns,
            'regime': self.regime_history if self.regime_history is not None else np.nan
        })


class Backtester:
    """
    Core backtesting engine with realistic market simulation.
    
    Features:
    - Transaction costs and slippage
    - Position tracking
    - Performance metrics calculation
    - Trade logging
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005,
                 max_leverage: float = 1.0):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting portfolio value
            transaction_cost: Cost per trade as fraction (0.001 = 0.1%)
            slippage: Price slippage as fraction
            max_leverage: Maximum allowed leverage
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_leverage = max_leverage
        
    def run(self, 
            portfolio_model: torch.nn.Module,
            vae_hmm: torch.nn.Module,
            market_data: torch.Tensor,
            prices: np.ndarray,
            returns: np.ndarray,
            dates: Optional[List[datetime]] = None) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            portfolio_model: Trained portfolio optimizer
            vae_hmm: Trained regime detection model
            market_data: Input features for regime detection
            prices: Asset prices (T x n_assets)
            returns: Asset returns (T x n_assets)
            dates: Optional timestamps
            
        Returns:
            BacktestResult with performance metrics
        """
        T, n_assets = prices.shape
        
        # Initialize tracking
        equity = np.zeros(T)
        equity[0] = self.initial_capital
        positions = np.zeros((T, n_assets))
        trades = []
        regime_history = np.zeros(T)
        
        portfolio_model.eval()
        vae_hmm.eval()
        
        with torch.no_grad():
            # Get regime probabilities
            logits = vae_hmm.encode(market_data)
            regime_probs = torch.softmax(logits, dim=1)
            regimes = regime_probs.argmax(dim=1).cpu().numpy()
            
            for t in range(1, T):
                # Get current regime
                current_regime_probs = regime_probs[:, :, t-1:t]
                regime_history[t] = regimes[0, t-1]
                
                # Get target weights
                target_weights = portfolio_model(current_regime_probs).cpu().numpy()[0]
                
                # Calculate target positions in dollars
                target_positions = target_weights * equity[t-1]
                
                # Calculate trades needed
                if t == 1:
                    trades_needed = target_positions
                else:
                    current_value = positions[t-1] * prices[t-1]
                    trades_needed = target_positions - current_value
                
                # Apply transaction costs and slippage
                trade_costs = np.abs(trades_needed) * (self.transaction_cost + self.slippage)
                total_cost = trade_costs.sum()
                
                # Execute trades
                positions[t] = target_positions / prices[t]
                
                # Update equity
                position_value = (positions[t] * prices[t]).sum()
                equity[t] = equity[t-1] + (positions[t-1] * (prices[t] - prices[t-1])).sum() - total_cost
                
                # Log significant trades
                if np.abs(trades_needed).sum() > equity[t-1] * 0.01:  # >1% rebalance
                    trades.append({
                        'timestamp': dates[t] if dates else t,
                        'regime': int(regime_history[t]),
                        'cost': total_cost,
                        'equity': equity[t]
                    })
        
        # Calculate metrics
        portfolio_returns = np.diff(equity) / equity[:-1]
        metrics = self._calculate_metrics(equity, portfolio_returns, trades)
        
        return BacktestResult(
            equity_curve=equity,
            returns=portfolio_returns,
            positions=positions,
            trades=trades,
            metrics=metrics,
            regime_history=regime_history
        )
    
    def _calculate_metrics(self, equity: np.ndarray, returns: np.ndarray, 
                          trades: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        total_return = (equity[-1] - equity[0]) / equity[0]
        
        # Annualized metrics (assuming daily data)
        annual_return = (1 + total_return) ** (252 / len(equity)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annual_return / downside_vol if downside_vol > 0 else 0
        
        # Drawdown analysis
        cumulative = equity / equity[0]
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns)
        
        # Transaction costs
        total_costs = sum(trade['cost'] for trade in trades)
        cost_ratio = total_costs / equity[0]
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'total_costs': total_costs,
            'cost_ratio': cost_ratio,
            'final_equity': equity[-1]
        }


class RegimeBacktest:
    """Analyze strategy performance by regime."""
    
    def __init__(self, backtester: Backtester):
        self.backtester = backtester
        
    def run(self, *args, **kwargs) -> Dict[int, BacktestResult]:
        """Run backtest and split results by regime."""
        result = self.backtester.run(*args, **kwargs)
        
        if result.regime_history is None:
            raise ValueError("Regime history not available")
        
        # Split by regime
        regime_results = {}
        unique_regimes = np.unique(result.regime_history)
        
        for regime in unique_regimes:
            mask = result.regime_history == regime
            regime_returns = result.returns[mask[1:]]  # Align with returns
            
            if len(regime_returns) > 0:
                regime_metrics = {
                    'regime': int(regime),
                    'n_periods': mask.sum(),
                    'mean_return': regime_returns.mean(),
                    'volatility': regime_returns.std(),
                    'sharpe': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                    'win_rate': (regime_returns > 0).sum() / len(regime_returns)
                }
                regime_results[int(regime)] = regime_metrics
        
        return regime_results


class WalkForwardBacktest:
    """
    Walk-forward backtesting with periodic retraining.
    
    This simulates realistic trading where models are retrained
    on rolling windows of historical data.
    """
    
    def __init__(self,
                 train_window: int = 252,
                 test_window: int = 21,
                 retrain_freq: int = 21):
        """
        Initialize walk-forward backtester.
        
        Args:
            train_window: Number of periods for training
            test_window: Number of periods for testing
            retrain_freq: How often to retrain (in periods)
        """
        self.train_window = train_window
        self.test_window = test_window
        self.retrain_freq = retrain_freq
        
    def run(self,
            portfolio_model: torch.nn.Module,
            vae_hmm: torch.nn.Module,
            train_fn: callable,
            data: torch.Tensor,
            prices: np.ndarray,
            returns: np.ndarray) -> List[BacktestResult]:
        """
        Run walk-forward backtest with retraining.
        
        Args:
            portfolio_model: Portfolio optimizer to train
            vae_hmm: Pre-trained regime model
            train_fn: Function to train portfolio model
            data: Full dataset
            prices: Full price history
            returns: Full returns history
            
        Returns:
            List of BacktestResult for each test period
        """
        results = []
        n_periods = (len(data) - self.train_window) // self.retrain_freq
        
        for i in range(n_periods):
            start_idx = i * self.retrain_freq
            train_end = start_idx + self.train_window
            test_end = min(train_end + self.test_window, len(data))
            
            # Training data
            train_data = data[start_idx:train_end]
            train_returns = returns[start_idx:train_end]
            
            # Retrain model
            print(f"Period {i+1}/{n_periods}: Training on {start_idx}:{train_end}")
            train_fn(portfolio_model, vae_hmm, train_data, train_returns)
            
            # Test data
            test_data = data[train_end:test_end]
            test_prices = prices[train_end:test_end]
            test_returns = returns[train_end:test_end]
            
            # Run backtest on test period
            backtester = Backtester()
            result = backtester.run(
                portfolio_model, vae_hmm, test_data, 
                test_prices, test_returns
            )
            results.append(result)
            
            print(f"  Test Sharpe: {result.metrics['sharpe_ratio']:.2f}, "
                  f"Return: {result.metrics['total_return']:.2%}")
        
        return results


def compare_strategies(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    """
    Compare multiple strategies side-by-side.
    
    Args:
        results: Dictionary mapping strategy names to BacktestResult
        
    Returns:
        DataFrame with comparative metrics
    """
    comparison = {}
    
    for name, result in results.items():
        comparison[name] = result.metrics
    
    df = pd.DataFrame(comparison).T
    
    # Sort by Sharpe ratio
    df = df.sort_values('sharpe_ratio', ascending=False)
    
    return df


def plot_results(result: BacktestResult, title: str = "Backtest Results"):
    """
    Plot backtest results.
    
    Args:
        result: BacktestResult to visualize
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Equity curve
        axes[0].plot(result.equity_curve)
        axes[0].set_title(f"{title} - Equity Curve")
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].grid(True, alpha=0.3)
        
        # Returns distribution
        axes[1].hist(result.returns, bins=50, alpha=0.7, edgecolor='black')
        axes[1].set_title("Returns Distribution")
        axes[1].set_xlabel("Return")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(True, alpha=0.3)
        
        # Drawdown
        cumulative = result.equity_curve / result.equity_curve[0]
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        axes[2].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.5, color='red')
        axes[2].set_title("Drawdown")
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Drawdown (%)")
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")
