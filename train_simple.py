import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
from data_loader import load_portfolio_data
import pickle


class SimpleRegimeDetector:
    """Simpler regime detection using Gaussian Mixture Model."""
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.gmm = GaussianMixture(n_components=n_regimes, covariance_type='full', 
                                    random_state=42, n_init=10)
    
    def fit(self, features):
        """Fit GMM on market features."""
        self.gmm.fit(features)
        return self
    
    def predict_proba(self, features):
        """Get regime probabilities."""
        return self.gmm.predict_proba(features)
    
    def predict(self, features):
        """Get most likely regime."""
        return self.gmm.predict(features)


class ImprovedPortfolioOptimizer(nn.Module):
    """Regime-conditional portfolio optimizer."""
    def __init__(self, n_regimes, n_assets, hidden_dim=128):
        super().__init__()
        self.n_regimes = n_regimes
        self.n_assets = n_assets
        
        # Separate network for each regime
        self.regime_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_regimes, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, n_assets)
            ) for _ in range(n_regimes)
        ])
    
    def forward(self, regime_probs):
        """Get portfolio weights based on regime probabilities."""
        batch_size = regime_probs.shape[0]
        weights = torch.zeros(batch_size, self.n_assets)
        
        for i in range(self.n_regimes):
            regime_weights = F.softmax(self.regime_nets[i](regime_probs), dim=-1)
            weights += regime_probs[:, i:i+1] * regime_weights
        
        return weights


def prepare_regime_features(data):
    """Extract features for regime detection."""
    returns = data['returns'].values
    prices = data['prices'].values
    
    # Basic features
    volatility = np.std(returns, axis=1)
    mean_return = np.mean(returns, axis=1)
    
    # Rolling features (multiple windows)
    features_list = [volatility, mean_return]
    
    for window in [10, 20, 60]:  # Short, medium, long term
        rolling_vol = np.array([np.std(returns[max(0, i-window):i+1]) if i >= window else np.std(returns[:i+1]) for i in range(len(returns))])
        rolling_ret = np.array([np.mean(returns[max(0, i-window):i+1]) if i >= window else np.mean(returns[:i+1]) for i in range(len(returns))])
        features_list.extend([rolling_vol, rolling_ret])
    
    # Momentum indicators
    momentum_20 = np.array([np.mean(returns[max(0, i-20):i+1], axis=0).mean() if i >= 20 else np.mean(returns[:i+1], axis=0).mean() for i in range(len(returns))])
    momentum_60 = np.array([np.mean(returns[max(0, i-60):i+1], axis=0).mean() if i >= 60 else np.mean(returns[:i+1], axis=0).mean() for i in range(len(returns))])
    features_list.extend([momentum_20, momentum_60])
    
    # Correlation (market stress indicator)
    correlation = np.array([np.corrcoef(returns[max(0, i-20):i+1].T).mean() if i >= 20 and len(returns[max(0, i-20):i+1]) > 1 else 0.5 for i in range(len(returns))])
    features_list.append(correlation)
    
    # Skewness (tail risk)
    from scipy.stats import skew
    skewness = np.array([skew(returns[max(0, i-20):i+1].flatten()) if i >= 20 and len(returns[max(0, i-20):i+1]) > 2 else 0.0 for i in range(len(returns))])
    features_list.append(skewness)
    
    # Combine features
    features = np.column_stack(features_list)
    
    # Replace any remaining NaN/inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features


def train_improved_system(data, n_regimes=3, epochs=200, lr=0.001):
    """Train regime detector and portfolio optimizer."""
    print("\n=== Training Improved System ===")
    
    # Step 1: Fit regime detector
    print("\n1. Fitting Regime Detector (GMM)...")
    features = prepare_regime_features(data)
    regime_detector = SimpleRegimeDetector(n_regimes=n_regimes)
    regime_detector.fit(features)
    
    # Get regime predictions
    regime_probs = regime_detector.predict_proba(features)
    regimes = regime_detector.predict(features)
    
    print(f"   Regime distribution: {np.bincount(regimes)}")
    print(f"   Average confidence: {regime_probs.max(axis=1).mean():.2%}")
    
    # Step 2: Train portfolio optimizer
    print("\n2. Training Portfolio Optimizer...")
    n_assets = data['returns'].shape[1]
    portfolio_model = ImprovedPortfolioOptimizer(n_regimes, n_assets, hidden_dim=128)
    optimizer = torch.optim.Adam(portfolio_model.parameters(), lr=lr)
    
    returns_tensor = torch.FloatTensor(data['returns'].values)
    regime_probs_tensor = torch.FloatTensor(regime_probs)
    
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        # Sample batches
        indices = torch.randperm(len(returns_tensor))[:min(256, len(returns_tensor))]
        batch_regime_probs = regime_probs_tensor[indices]
        
        # Forward pass
        optimizer.zero_grad()
        weights = portfolio_model(batch_regime_probs)
        
        # Calculate returns for multiple periods
        port_returns = []
        for i in range(len(indices)):
            idx = indices[i].item()
            future_returns = returns_tensor[idx:min(idx+20, len(returns_tensor))]
            if len(future_returns) > 0:
                port_ret = (weights[i:i+1] * future_returns).sum(dim=-1)
                port_returns.append(port_ret)
        
        if len(port_returns) > 0:
            port_returns = torch.cat(port_returns)
            
            # Sharpe ratio loss
            mean_ret = port_returns.mean()
            std_ret = port_returns.std() + 1e-8
            sharpe = mean_ret / std_ret
            
            # Regularization: encourage diversification
            diversity_penalty = (weights ** 2).sum(dim=1).mean()
            
            loss = -sharpe + 0.1 * diversity_penalty
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(portfolio_model.parameters(), 1.0)
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Sharpe: {sharpe.item():.4f}")
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   Early stopping at epoch {epoch+1}")
                    break
    
    print("\n=== Training Complete ===")
    return regime_detector, portfolio_model


def save_improved_models(regime_detector, portfolio_model, path='models/'):
    """Save trained models."""
    import os
    os.makedirs(path, exist_ok=True)
    
    # Save GMM
    with open(f'{path}regime_detector.pkl', 'wb') as f:
        pickle.dump(regime_detector, f)
    
    # Save portfolio model
    torch.save(portfolio_model.state_dict(), f'{path}portfolio_improved.pt')
    
    print(f"\nModels saved to {path}")


if __name__ == '__main__':
    # Load data
    TICKERS = ['SPY', 'QQQ', 'IWM', 'EFA', 'TLT', 'LQD', 'GLD', 'DBC', 'XLE', 'XLF']
    
    print("Loading data...")
    data = load_portfolio_data(
        tickers=TICKERS,
        start_date='2010-01-01',
        end_date='2024-01-01'
    )
    
    print(f"Data: {data['returns'].shape[0]} days, {len(TICKERS)} assets")
    
    # Train
    regime_detector, portfolio_model = train_improved_system(data, n_regimes=3, epochs=200, lr=0.001)
    
    # Save
    save_improved_models(regime_detector, portfolio_model)
    
    print("\nRun 'python inference_improved.py' to test")
