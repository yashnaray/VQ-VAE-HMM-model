# VQ-VAE-HMM Portfolio Optimization System

Implementation of regime-switching models with portfolio optimization and delta hedging.

## Core Modules

### 1. VQ_VAE_HMM_fixed.py
Base VQ-VAE-HMM model with proper Gaussian emissions and numerical stability.

**Key Classes:**
- `Encoder`: Conv1D encoder for regime detection
- `Prior`: HMM prior with input-conditioned transitions
- `Decoder`: Gaussian emission decoder with mean and log-variance
- `VAE_HMM`: Main model combining all components
- `RandomChunkDataset`: Dataset for variable-length sequences

### 2. portfolio_optimizer.py
Neural network architectures for portfolio optimization.

**Models:**
- `AttentionPortfolioOptimizer`: Multi-head attention for regime weighting
- `TransformerPortfolioOptimizer`: Full transformer encoder for sequences
- `BayesianPortfolioOptimizer`: Uncertainty quantification with weight distributions
- `EnsemblePortfolioOptimizer`: Multiple models for robust predictions
- `HierarchicalPortfolioOptimizer`: Macro → micro regime hierarchy

### 3. loss_functions.py
Comprehensive loss functions for portfolio optimization.

**Loss Functions:**
- `portfolio_loss`: Multi-objective with transaction costs, position limits, leverage, drawdown, CVaR
- `sortino_loss`: Downside risk optimization
- `calmar_loss`: Return/max drawdown ratio
- `risk_parity_loss`: Equal risk contribution
- `regime_conditional_loss`: Regime-specific covariance optimization
- `adversarial_portfolio_loss`: Robustness to regime misclassification
- `transition_aware_loss`: Accounts for expected regime changes

### 4. training.py
Training strategies and optimization techniques.

**Classes:**
- `MetaPortfolioOptimizer`: MAML-style meta-learning for fast adaptation
- `OnlinePortfolioOptimizer`: Continuous learning with EMA
- `WalkForwardTrainer`: Realistic backtesting with rolling windows

**Functions:**
- `train_portfolio`: Comprehensive training with schedulers, adversarial training, ensembles

### 5. regime_utilities.py
Utilities for regime analysis and portfolio construction.

**Models:**
- `RegimeChangeDetector`: Predict upcoming regime transitions
- `ForwardTransitionPredictor`: Multi-step ahead regime forecasting
- `RegimePersistenceModel`: Estimate expected regime duration
- `TemperatureScaling`: Calibrate regime probabilities
- `RegimeFactorModel`: Factor decomposition per regime

**Functions:**
- `estimate_regime_covariance`: Regime-conditional covariance matrices
- `confidence_based_sizing`: Scale positions by regime certainty
- `optimize_rebalancing_frequency`: Optimal trade-off between alpha and costs
- `optimize_leverage`: Target volatility with leverage constraints

### 6. delta_hedger.py
Regime-aware delta hedging strategies.

**Models:**
- `RegimeDeltaHedger`: Basic regime-conditional delta hedging
- `DynamicDeltaHedger`: Includes gamma hedging
- `LSTMDeltaHedger`: Sequential hedging decisions
- `TransactionCostAwareHedger`: Optimal rehedging thresholds
- `TransitionAwareHedger`: Anticipates regime changes

**Functions:**
- `minimum_variance_hedge_ratio`: Regime-conditional minimum variance hedge
- `delta_hedge_loss`: Variance minimization with transaction costs
- `optimal_hedge_frequency`: Leland (1985) with regime persistence
- `train_delta_hedger`: Training loop for hedging models

### 7. backtesting.py
Backtesting framework for portfolio strategies.

**Classes:**
- `Backtester`: Core backtesting engine with transaction costs and slippage
- `WalkForwardBacktest`: Rolling window backtesting with retraining
- `RegimeBacktest`: Regime-specific performance analysis
- `BacktestResult`: Results container with metrics and equity curve

**Functions:**
- `compare_strategies`: Compare multiple strategies side-by-side
- `plot_results`: Visualize backtest results

## Usage Examples

### Basic VQ-VAE-HMM Training
```python
from VQ_VAE_HMM_fixed import VAE_HMM, train_model, collate_fn, RandomChunkDataset

# Create model
model = VAE_HMM(input_dim=5, hidden_dim=64, K=3, hidden_dim2=32, u_dim=4)

# Create dataset
dataset = RandomChunkDataset(x_sequences, u_sequences, min_len=20, max_len=200)
dataloader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn)

# Train
trained_model = train_model(model, dataloader, num_epochs=150, lr=1e-5)
```

### Portfolio Optimization with Transformer
```python
from portfolio_optimizer import TransformerPortfolioOptimizer
from loss_functions import portfolio_loss
from training import train_portfolio

# Create optimizer
portfolio_model = TransformerPortfolioOptimizer(K=3, n_assets=10, hidden_dim=64)

# Train with features
trained_portfolio = train_portfolio(
    portfolio_model, vae_hmm, dataloader, returns_data,
    num_epochs=100, lr=0.001, use_scheduler=True,
    use_adversarial=True, use_ensemble=False
)
```

### Delta Hedging
```python
from delta_hedger import TransitionAwareHedger, train_delta_hedger

# Create hedger
hedger = TransitionAwareHedger(K=3, n_assets=10, hidden_dim=64, lookahead=5)

# Train
trained_hedger = train_delta_hedger(
    hedger, vae_hmm, spot_data, futures_data,
    num_epochs=50, lr=0.001
)

# Get hedge ratios
with torch.no_grad():
    hedge_ratios = hedger(regime_probs, transition_matrix, spot_prices)
```

### Meta-Learning for Fast Adaptation
```python
from training import MetaPortfolioOptimizer

meta_optimizer = MetaPortfolioOptimizer(
    model, inner_lr=0.01, outer_lr=0.001, n_inner_steps=5
)

# Meta-train on multiple market conditions
for epoch in range(100):
    tasks = sample_tasks(dataloader)  # Different market regimes
    meta_loss = meta_optimizer.meta_update(tasks, loss_fn)
```

### Walk-Forward Validation
```python
from training import WalkForwardTrainer

trainer = WalkForwardTrainer(
    model, loss_fn, train_window=252, test_window=21, retrain_freq=21
)

results = trainer.run(full_data, n_periods=50)
```

### Backtesting
```python
from backtesting import Backtester, WalkForwardBacktest, compare_strategies

# basic backtest
backtester = Backtester(initial_capital=100000, tx_cost=0.001)
result = backtester.run(portfolio_model, vae_hmm, data, prices, returns)

print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")

# walk-forward backtest
wf_backtest = WalkForwardBacktest(train_window=252, test_window=21)
wf_results = wf_backtest.run(portfolio_model, vae_hmm, train_fn, data, prices, returns)

# compare strategies
comparison = compare_strategies({'strategy1': result1, 'strategy2': result2})
print(comparison)
```

## Key Features

### Risk Management
- Transaction cost modeling
- Position and leverage limits
- Maximum drawdown constraints
- CVaR optimization
- Regime-conditional covariance

### Architectures
- Multi-head attention
- Transformer encoders
- Bayesian uncertainty quantification
- Ensemble methods
- Hierarchical regime modeling

### Training Enhancements
- Meta-learning (MAML)
- Online learning with EMA
- Walk-forward validation
- Adversarial training
- Learning rate scheduling
- Gradient clipping

### Regime Analysis
- Transition prediction
- Regime persistence modeling
- Probability calibration
- Factor models per regime
- Confidence-based sizing

### Delta Hedging
- Regime-aware hedging
- Gamma hedging
- Transaction cost optimization
- Optimal rehedging frequency
- Transition-aware hedging

### Backtesting
- Transaction costs and slippage
- Walk-forward validation
- Regime-specific analysis
- Performance metrics (Sharpe, Sortino, Calmar)
- Drawdown analysis

## Model Improvements Summary

1. **Architecture**: Attention, Transformers, Bayesian, Ensemble, Hierarchical
2. **Loss Functions**: Sharpe, Sortino, Calmar, Risk Parity, CVaR, Multi-objective
3. **Training**: Meta-learning, Online learning, Walk-forward, Adversarial
4. **Risk Management**: Transaction costs, Position limits, Drawdown, Leverage
5. **Regime Modeling**: Transitions, Persistence, Calibration, Factors
6. **Hedging**: Delta, Gamma, Transaction-aware, Transition-aware
7. **Uncertainty**: Bayesian weights, Confidence sizing, Ensemble predictions
8. **Backtesting**: Walk-forward, Regime-specific, Transaction costs, Performance metrics

## Dependencies
- PyTorch
- NumPy
- Pandas
- Math

## Notes
- All models support GPU acceleration
- Gradient clipping prevents training instability
- Numerical stability built into all loss functions
- Supports variable-length sequences with masking
