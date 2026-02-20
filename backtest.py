import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train_simple import SimpleRegimeDetector, ImprovedPortfolioOptimizer, prepare_regime_features
from data_loader import load_portfolio_data
import pickle


def load_models(n_regimes=3, n_assets=10, path='models/'):
    """Load trained models."""
    with open(f'{path}regime_detector.pkl', 'rb') as f:
        regime_detector = pickle.load(f)
    
    portfolio_model = ImprovedPortfolioOptimizer(n_regimes, n_assets, hidden_dim=128)
    portfolio_model.load_state_dict(torch.load(f'{path}portfolio_improved.pt'))
    portfolio_model.eval()
    
    return regime_detector, portfolio_model


def backtest(regime_detector, portfolio_model, data, initial_capital=100000, 
             tx_cost=0.001, rebalance_freq=5):
    """
    Backtest the portfolio strategy.
    
    Args:
        regime_detector: Trained regime detector
        portfolio_model: Trained portfolio optimizer
        data: Market data
        initial_capital: Starting capital
        tx_cost: Transaction cost (0.001 = 0.1%)
        rebalance_freq: Rebalance every N days
    """
    returns = data['returns'].values
    prices = data['prices'].values
    features = prepare_regime_features(data)
    
    # Get regime predictions
    regime_probs = regime_detector.predict_proba(features)
    
    # Initialize
    portfolio_value = [initial_capital]
    weights_history = []
    current_weights = np.zeros(returns.shape[1])
    
    for t in range(len(returns)):
        # Get target weights
        if t % rebalance_freq == 0:
            regime_tensor = torch.FloatTensor(regime_probs[t]).unsqueeze(0)
            with torch.no_grad():
                target_weights = portfolio_model(regime_tensor)[0].tolist()
            target_weights = np.array(target_weights)
            
            # Calculate transaction costs
            weight_change = np.abs(target_weights - current_weights).sum()
            cost = portfolio_value[-1] * weight_change * tx_cost
            
            current_weights = target_weights
            portfolio_value[-1] -= cost
        
        weights_history.append(current_weights.copy())
        
        # Calculate portfolio return
        port_return = (current_weights * returns[t]).sum()
        new_value = portfolio_value[-1] * (1 + port_return)
        portfolio_value.append(new_value)
    
    portfolio_value = portfolio_value[:-1]  # Remove last element
    
    return np.array(portfolio_value), np.array(weights_history)


def calculate_metrics(portfolio_value, returns_data):
    """Calculate performance metrics."""
    returns = np.diff(portfolio_value) / portfolio_value[:-1]
    
    # Annualized metrics (assuming daily data)
    total_return = (portfolio_value[-1] / portfolio_value[0]) - 1
    n_years = len(portfolio_value) / 252
    cagr = (1 + total_return) ** (1 / n_years) - 1
    
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / (annual_vol + 1e-8)
    
    # Drawdown
    cummax = np.maximum.accumulate(portfolio_value)
    drawdown = (portfolio_value - cummax) / cummax
    max_drawdown = drawdown.min()
    
    # Sortino (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-8
    sortino = (returns.mean() * 252) / downside_std
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns)
    
    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate
    }


def plot_results(portfolio_value, benchmark_value, data):
    """Plot backtest results."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Portfolio value
    dates = data['returns'].index[:len(portfolio_value)]
    axes[0].plot(dates, portfolio_value, label='Strategy', linewidth=2)
    axes[0].plot(dates, benchmark_value, label='Buy & Hold (Equal Weight)', linewidth=2, alpha=0.7)
    axes[0].set_title('Portfolio Value Over Time')
    axes[0].set_ylabel('Value ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Drawdown
    cummax = np.maximum.accumulate(portfolio_value)
    drawdown = (portfolio_value - cummax) / cummax
    axes[1].fill_between(dates, drawdown * 100, 0, alpha=0.3, color='red')
    axes[1].set_title('Drawdown')
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].set_xlabel('Date')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150)
    print("\nPlot saved as 'backtest_results.png'")
    plt.close()


def monte_carlo_simulation(regime_detector, portfolio_model, data, n_simulations=1000, 
                           n_days=252, initial_capital=100000, tx_cost=0.001, rebalance_freq=5):
    """
    Monte Carlo simulation for future performance.
    
    Args:
        n_simulations: Number of simulation runs
        n_days: Number of days to simulate (252 = 1 year)
    """
    print(f"\n=== Running {n_simulations} Monte Carlo Simulations ===")
    print(f"Simulating {n_days} days ({n_days/252:.1f} years) forward")
    
    returns = data['returns'].values
    features = prepare_regime_features(data)
    regime_probs = regime_detector.predict_proba(features)
    regimes = regime_detector.predict(features)
    
    # Calculate statistics per regime
    regime_stats = {}
    for r in range(3):
        regime_mask = regimes == r
        regime_returns = returns[regime_mask]
        regime_stats[r] = {
            'mean': regime_returns.mean(axis=0),
            'cov': np.cov(regime_returns.T)
        }
    
    # Run simulations
    final_values = []
    all_paths = []
    
    for sim in range(n_simulations):
        portfolio_value = initial_capital
        path = [portfolio_value]
        current_weights = np.zeros(returns.shape[1])
        
        # Start from random recent regime
        current_regime = np.random.choice(3, p=regime_probs[-1])
        
        for day in range(n_days):
            # Rebalance
            if day % rebalance_freq == 0:
                # Sample regime transition
                regime_probs_current = np.zeros(3)
                regime_probs_current[current_regime] = 1.0
                regime_tensor = torch.FloatTensor(regime_probs_current).unsqueeze(0)
                
                with torch.no_grad():
                    target_weights = portfolio_model(regime_tensor)[0].tolist()
                target_weights = np.array(target_weights)
                
                # Transaction costs
                weight_change = np.abs(target_weights - current_weights).sum()
                cost = portfolio_value * weight_change * tx_cost
                portfolio_value -= cost
                
                current_weights = target_weights
                
                # Regime transition (simple Markov)
                if np.random.rand() < 0.05:  # 5% chance of regime change per rebalance
                    current_regime = np.random.choice(3)
            
            # Sample returns from current regime
            mean_ret = regime_stats[current_regime]['mean']
            cov_ret = regime_stats[current_regime]['cov']
            sampled_returns = np.random.multivariate_normal(mean_ret, cov_ret)
            
            # Update portfolio
            port_return = (current_weights * sampled_returns).sum()
            portfolio_value *= (1 + port_return)
            path.append(portfolio_value)
        
        final_values.append(portfolio_value)
        all_paths.append(path)
        
        if (sim + 1) % 200 == 0:
            print(f"  Completed {sim + 1}/{n_simulations} simulations")
    
    return np.array(final_values), np.array(all_paths)


def analyze_monte_carlo(final_values, all_paths, initial_capital, n_days):
    """Analyze Monte Carlo results."""
    returns = (final_values - initial_capital) / initial_capital
    n_years = n_days / 252
    
    print("\n=== Monte Carlo Results ===")
    print(f"\nFinal Portfolio Value:")
    print(f"  Mean                : ${final_values.mean():,.2f}")
    print(f"  Median              : ${np.median(final_values):,.2f}")
    print(f"  Std Dev             : ${final_values.std():,.2f}")
    print(f"  5th Percentile      : ${np.percentile(final_values, 5):,.2f}")
    print(f"  95th Percentile     : ${np.percentile(final_values, 95):,.2f}")
    
    print(f"\nReturns:")
    print(f"  Mean                : {returns.mean():>8.2%}")
    print(f"  Median              : {np.median(returns):>8.2%}")
    print(f"  Std Dev             : {returns.std():>8.2%}")
    print(f"  5th Percentile      : {np.percentile(returns, 5):>8.2%}")
    print(f"  95th Percentile     : {np.percentile(returns, 95):>8.2%}")
    
    # Probability of profit
    prob_profit = (final_values > initial_capital).sum() / len(final_values)
    print(f"\nProbability of Profit : {prob_profit:>8.2%}")
    
    # Expected Sharpe
    mean_ret = returns.mean() / n_years
    std_ret = returns.std() / np.sqrt(n_years)
    expected_sharpe = mean_ret / (std_ret + 1e-8)
    print(f"Expected Sharpe Ratio : {expected_sharpe:>8.2f}")
    
    return returns


def plot_monte_carlo(all_paths, initial_capital, n_days):
    """Plot Monte Carlo simulation paths."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Sample paths
    n_plot = min(100, len(all_paths))
    sample_indices = np.random.choice(len(all_paths), n_plot, replace=False)
    
    for idx in sample_indices:
        axes[0].plot(all_paths[idx], alpha=0.1, color='blue', linewidth=0.5)
    
    # Mean and percentiles
    mean_path = all_paths.mean(axis=0)
    p5_path = np.percentile(all_paths, 5, axis=0)
    p95_path = np.percentile(all_paths, 95, axis=0)
    
    axes[0].plot(mean_path, color='red', linewidth=2, label='Mean')
    axes[0].plot(p5_path, color='orange', linewidth=2, linestyle='--', label='5th Percentile')
    axes[0].plot(p95_path, color='green', linewidth=2, linestyle='--', label='95th Percentile')
    axes[0].axhline(initial_capital, color='black', linestyle=':', label='Initial Capital')
    axes[0].set_title(f'Monte Carlo Simulation ({len(all_paths)} paths, {n_days} days)')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Distribution of final values
    final_values = all_paths[:, -1]
    axes[1].hist(final_values, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(initial_capital, color='red', linestyle='--', linewidth=2, label='Initial Capital')
    axes[1].axvline(final_values.mean(), color='green', linestyle='--', linewidth=2, label='Mean')
    axes[1].axvline(np.median(final_values), color='blue', linestyle='--', linewidth=2, label='Median')
    axes[1].set_title('Distribution of Final Portfolio Values')
    axes[1].set_xlabel('Portfolio Value ($)')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('monte_carlo_results.png', dpi=150)
    print("\nMonte Carlo plot saved as 'monte_carlo_results.png'")
    plt.close()


def benchmark_equal_weight(returns, initial_capital=100000):
    """Calculate buy-and-hold equal weight benchmark."""
    n_assets = returns.shape[1]
    equal_weights = np.ones(n_assets) / n_assets
    
    portfolio_value = [initial_capital]
    for t in range(len(returns)):
        port_return = (equal_weights * returns[t]).sum()
        portfolio_value.append(portfolio_value[-1] * (1 + port_return))
    
    return np.array(portfolio_value[:-1])
    """Calculate buy-and-hold equal weight benchmark."""
    n_assets = returns.shape[1]
    equal_weights = np.ones(n_assets) / n_assets
    
    portfolio_value = [initial_capital]
    for t in range(len(returns)):
        port_return = (equal_weights * returns[t]).sum()
        portfolio_value.append(portfolio_value[-1] * (1 + port_return))
    
    return np.array(portfolio_value[:-1])


if __name__ == '__main__':
    TICKERS = ['SPY', 'QQQ', 'IWM', 'EFA', 'TLT', 'LQD', 'GLD', 'DBC', 'XLE', 'XLF']
    
    print("Loading data...")
    data = load_portfolio_data(
        tickers=TICKERS,
        start_date='2010-01-01',
        end_date='2024-01-01'
    )
    
    print("Loading models...")
    regime_detector, portfolio_model = load_models(n_regimes=3, n_assets=len(TICKERS))
    
    print("\n=== Running Backtest ===")
    portfolio_value, weights_history = backtest(
        regime_detector, 
        portfolio_model, 
        data,
        initial_capital=100000,
        tx_cost=0.001,  # 0.1% transaction cost
        rebalance_freq=5  # Rebalance every 5 days
    )
    
    # Benchmark
    benchmark_value = benchmark_equal_weight(data['returns'].values, initial_capital=100000)
    
    # Calculate metrics
    print("\n=== Strategy Performance ===")
    strategy_metrics = calculate_metrics(portfolio_value, data['returns'].values)
    for metric, value in strategy_metrics.items():
        if 'Rate' in metric or 'Return' in metric or 'Drawdown' in metric:
            print(f"{metric:20s}: {value:>8.2%}")
        else:
            print(f"{metric:20s}: {value:>8.2f}")
    
    print("\n=== Benchmark Performance (Equal Weight) ===")
    benchmark_metrics = calculate_metrics(benchmark_value, data['returns'].values)
    for metric, value in benchmark_metrics.items():
        if 'Rate' in metric or 'Return' in metric or 'Drawdown' in metric:
            print(f"{metric:20s}: {value:>8.2%}")
        else:
            print(f"{metric:20s}: {value:>8.2f}")
    
    # Comparison
    print("\n=== Strategy vs Benchmark ===")
    print(f"Excess Return       : {(strategy_metrics['Total Return'] - benchmark_metrics['Total Return']):>8.2%}")
    print(f"Sharpe Improvement  : {(strategy_metrics['Sharpe Ratio'] - benchmark_metrics['Sharpe Ratio']):>8.2f}")
    
    # Plot
    plot_results(portfolio_value, benchmark_value, data)
    
    print(f"\nFinal Portfolio Value: ${portfolio_value[-1]:,.2f}")
    print(f"Initial Capital      : ${100000:,.2f}")
    print(f"Profit/Loss          : ${portfolio_value[-1] - 100000:,.2f}")
    
    # Monte Carlo Simulation
    print("\n" + "="*60)
    print("MONTE CARLO SIMULATION FOR FUTURE PERFORMANCE")
    print("="*60)
    
    final_values, all_paths = monte_carlo_simulation(
        regime_detector,
        portfolio_model,
        data,
        n_simulations=1000,
        n_days=252,  # 1 year forward
        initial_capital=100000,
        tx_cost=0.001,
        rebalance_freq=5
    )
    
    analyze_monte_carlo(final_values, all_paths, initial_capital=100000, n_days=252)
    plot_monte_carlo(all_paths, initial_capital=100000, n_days=252)
