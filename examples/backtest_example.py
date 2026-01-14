import torch
import numpy as np
import pandas as pd
from backtesting import Backtester, WalkForwardBacktest, RegimeBacktest, compare_strategies, plot_results
from VQ_VAE_HMM_fixed import VAE_HMM
from portfolio_optimizer import TransformerPortfolioOptimizer
from loss_functions import portfolio_loss


def load_data():
    # load your data here
    df = pd.read_csv('data.csv')
    prices = df[['asset1', 'asset2', 'asset3']].values
    returns = np.diff(prices, axis=0) / prices[:-1]
    
    # prepare features for VAE-HMM
    features = df[['feature1', 'feature2', 'feature3']].values
    x_data = torch.tensor(features.T, dtype=torch.float32).unsqueeze(0)
    
    return x_data, prices, returns


def train_model(model, vae_hmm, data):
    # simple training loop
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    for _ in range(10):
        opt.zero_grad()
        regime_probs = torch.softmax(vae_hmm.encode(data), dim=1)
        weights = model(regime_probs)
        
        # dummy loss for example
        loss = -weights.mean()
        loss.backward()
        opt.step()


def main():
    # initialize models
    vae_hmm = VAE_HMM(input_dim=5, hidden_dim=64, K=3, hidden_dim2=32, u_dim=4)
    portfolio_model = TransformerPortfolioOptimizer(K=3, n_assets=10, hidden_dim=64)
    
    # load pretrained weights
    vae_hmm.load_state_dict(torch.load('checkpoints/vae_hmm.pth'))
    
    # load data
    data, prices, returns = load_data()
    
    # basic backtest
    print("Running basic backtest...")
    backtester = Backtester(initial_capital=100000, tx_cost=0.001)
    result = backtester.run(portfolio_model, vae_hmm, data, prices, returns)
    
    print("\nBacktest Metrics:")
    for metric, value in result.metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # walk-forward backtest
    print("\n\nRunning walk-forward backtest...")
    wf_backtest = WalkForwardBacktest(train_window=252, test_window=21, retrain_freq=21)
    wf_results = wf_backtest.run(portfolio_model, vae_hmm, train_model, data, prices, returns)
    
    print(f"Completed {len(wf_results)} walk-forward periods")
    avg_sharpe = np.mean([r.metrics['sharpe_ratio'] for r in wf_results])
    print(f"Average Sharpe Ratio: {avg_sharpe:.4f}")
    
    # regime-specific backtest
    print("\n\nRunning regime-specific backtest...")
    regime_backtest = RegimeBacktest()
    regime_results = regime_backtest.run(portfolio_model, vae_hmm, data, prices, returns, K=3)
    
    print("\nRegime-specific metrics:")
    for regime, result in regime_results.items():
        print(f"\nRegime {regime}:")
        print(f"  Sharpe: {result.metrics['sharpe_ratio']:.4f}")
        print(f"  Max DD: {result.metrics['max_drawdown']:.4f}")
    
    # compare multiple strategies
    print("\n\nComparing strategies...")
    strategies = {
        'transformer': result,
        'regime_0': regime_results.get(0),
        'regime_1': regime_results.get(1),
    }
    strategies = {k: v for k, v in strategies.items() if v is not None}
    
    comparison = compare_strategies(strategies)
    print("\nStrategy Comparison:")
    print(comparison[['sharpe_ratio', 'max_drawdown', 'win_rate']])
    
    # plot results
    fig = plot_results(result, title="Portfolio Backtest")
    if fig:
        fig.savefig('backtest_results.png')
        print("\nPlot saved to backtest_results.png")


if __name__ == "__main__":
    main()
