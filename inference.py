import torch
import pandas as pd
import pickle
from train_simple import ImprovedPortfolioOptimizer, prepare_regime_features, SimpleRegimeDetector
from data_loader import load_portfolio_data


def load_improved_models(n_regimes=3, n_assets=10, path='models/'):
    """Load trained models."""
    # Load GMM
    with open(f'{path}regime_detector.pkl', 'rb') as f:
        regime_detector = pickle.load(f)
    
    # Load portfolio model
    portfolio_model = ImprovedPortfolioOptimizer(n_regimes, n_assets, hidden_dim=128)
    portfolio_model.load_state_dict(torch.load(f'{path}portfolio_improved.pt'))
    portfolio_model.eval()
    
    return regime_detector, portfolio_model


if __name__ == '__main__':
    TICKERS = ['SPY', 'QQQ', 'IWM', 'EFA', 'TLT', 'LQD', 'GLD', 'DBC', 'XLE', 'XLF']
    
    print("Loading data...")
    data = load_portfolio_data(tickers=TICKERS, start_date='2010-01-01', end_date='2024-01-01')
    
    print("Loading models...")
    regime_detector, portfolio_model = load_improved_models(n_regimes=3, n_assets=len(TICKERS))
    
    # Prepare features
    features = prepare_regime_features(data)
    
    # Get regime predictions
    regime_probs = regime_detector.predict_proba(features)
    regimes = regime_detector.predict(features)
    
    # Current regime
    print("\n=== Current Market Regime ===")
    current_probs = regime_probs[-1]
    current_regime = regimes[-1]
    confidence = current_probs.max()
    
    print(f"Regime: {current_regime} (Confidence: {confidence:.2%})")
    print(f"Regime probabilities: {[f'{p:.3f}' for p in current_probs]}")
    
    # Get portfolio allocation
    print("\n=== Optimal Portfolio Allocation ===")
    regime_tensor = torch.FloatTensor(current_probs).unsqueeze(0)
    with torch.no_grad():
        weights = portfolio_model(regime_tensor)[0].tolist()
    
    allocation = pd.DataFrame({
        'Ticker': TICKERS,
        'Weight': weights
    }).sort_values('Weight', ascending=False)
    
    print(allocation.to_string(index=False))
    print(f"\nTotal: {sum(weights):.2%}")
    
    # Statistics
    print("\n=== Regime Distribution ===")
    regime_counts = pd.Series(regimes).value_counts().sort_index()
    for regime, count in regime_counts.items():
        pct = count / len(regimes) * 100
        print(f"Regime {regime}: {count} periods ({pct:.1f}%)")
    
    avg_confidence = regime_probs.max(axis=1).mean()
    print(f"\nAverage confidence: {avg_confidence:.2%}")
    
    # Recent allocations
    print("\n=== Recent Allocations (Last 5 Periods) ===")
    for i in range(max(0, len(regimes)-5), len(regimes)):
        regime_tensor = torch.FloatTensor(regime_probs[i]).unsqueeze(0)
        with torch.no_grad():
            weights = portfolio_model(regime_tensor)[0].tolist()
        
        top_3 = sorted(zip(TICKERS, weights), key=lambda x: x[1], reverse=True)[:3]
        print(f"\nPeriod {i-len(regimes)+6}:")
        print(f"  Regime: {regimes[i]} (Confidence: {regime_probs[i].max():.2%})")
        print(f"  Top holdings: {', '.join([f'{t}: {w:.1%}' for t, w in top_3])}")
