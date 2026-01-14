import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Regime change detector
class RegimeChangeDetector(nn.Module):
    def __init__(self, K, hidden_dim=64):
        super().__init__()
        self.K = K
        self.lstm = nn.LSTM(K, hidden_dim, 2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, regime_probs_seq):
        lstm_out, _ = self.lstm(regime_probs_seq)
        change_prob = torch.sigmoid(self.fc(lstm_out[:, -1, :]))
        return change_prob

# Forward transition predictor
class ForwardTransitionPredictor(nn.Module):
    def __init__(self, K, n_steps=5, hidden_dim=64):
        super().__init__()
        self.K = K
        self.n_steps = n_steps
        self.lstm = nn.LSTM(K, hidden_dim, 2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, K * n_steps)
        
    def forward(self, regime_probs_seq):
        lstm_out, _ = self.lstm(regime_probs_seq)
        logits = self.fc(lstm_out[:, -1, :]).view(-1, self.n_steps, self.K)
        future_probs = F.softmax(logits, dim=-1)
        return future_probs

# Regime persistence model
class RegimePersistenceModel(nn.Module):
    def __init__(self, K, hidden_dim=32):
        super().__init__()
        self.K = K
        self.fc1 = nn.Linear(K, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, regime_probs, transition_matrix):
        if regime_probs.dim() == 3:
            regime_probs = regime_probs[:, :, -1]
        
        self_transition = torch.diagonal(transition_matrix, dim1=-2, dim2=-1)
        weighted_persistence = (regime_probs * self_transition).sum(dim=-1, keepdim=True)
        
        x = F.relu(self.fc1(regime_probs))
        duration_logits = self.fc2(x)
        expected_duration = F.softplus(duration_logits) + weighted_persistence
        
        return expected_duration

# Calibration for regime probabilities
def calibrate_probabilities(regime_probs, true_regimes, n_bins=10):
    regime_probs_np = regime_probs.detach().cpu().numpy()
    true_regimes_np = true_regimes.detach().cpu().numpy()
    
    max_probs = regime_probs_np.max(axis=-1)
    predicted_regimes = regime_probs_np.argmax(axis=-1)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    calibration_data = []
    
    for i in range(n_bins):
        mask = (max_probs >= bin_edges[i]) & (max_probs < bin_edges[i + 1])
        if mask.sum() > 0:
            accuracy = (predicted_regimes[mask] == true_regimes_np[mask]).mean()
            confidence = max_probs[mask].mean()
            calibration_data.append((confidence, accuracy))
    
    return calibration_data

# Temperature scaling for calibration
class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, logits):
        return logits / self.temperature
    
    def calibrate(self, logits, labels, lr=0.01, max_iter=50):
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        return self.temperature.item()

# Covariance estimation per regime
def estimate_regime_covariance(returns, regime_probs, K):
    B, T, n_assets = returns.shape
    regime_probs_t = regime_probs.permute(0, 2, 1) if regime_probs.shape[1] == K else regime_probs
    
    covariances = []
    for k in range(K):
        regime_weight = regime_probs_t[:, :, k].unsqueeze(-1)
        weighted_returns = returns * regime_weight
        
        mean_returns = weighted_returns.sum(dim=1, keepdim=True) / regime_weight.sum(dim=1, keepdim=True).clamp(min=1e-8)
        centered_returns = weighted_returns - mean_returns
        
        cov = torch.bmm(centered_returns.transpose(1, 2), centered_returns * regime_weight) / regime_weight.sum(dim=1, keepdim=True).clamp(min=1e-8)
        covariances.append(cov)
    
    return torch.stack(covariances, dim=1)

# Factor model per regime
class RegimeFactorModel(nn.Module):
    def __init__(self, K, n_assets, n_factors=5):
        super().__init__()
        self.K = K
        self.n_factors = n_factors
        self.factor_loadings = nn.Parameter(torch.randn(K, n_assets, n_factors))
        self.specific_risk = nn.Parameter(torch.ones(K, n_assets))
        
    def get_covariance(self, regime_probs):
        if regime_probs.dim() == 3:
            regime_probs = regime_probs[:, :, -1]
        
        B = regime_probs.shape[0]
        covariances = []
        
        for b in range(B):
            cov = torch.zeros(self.n_assets, self.n_assets, device=regime_probs.device)
            for k in range(self.K):
                F_k = self.factor_loadings[k]
                D_k = torch.diag(self.specific_risk[k] ** 2)
                cov_k = torch.mm(F_k, F_k.t()) + D_k
                cov += regime_probs[b, k] * cov_k
            covariances.append(cov)
        
        return torch.stack(covariances)

# Confidence-based position sizing
def confidence_based_sizing(weights, regime_probs, min_confidence=0.5, max_scale=1.5):
    if regime_probs.dim() == 3:
        regime_probs = regime_probs[:, :, -1]
    
    confidence = regime_probs.max(dim=-1)[0]
    confidence_normalized = (confidence - min_confidence).clamp(min=0) / (1 - min_confidence)
    scale = 1.0 + (max_scale - 1.0) * confidence_normalized
    
    scaled_weights = weights * scale.unsqueeze(-1)
    scaled_weights = scaled_weights / scaled_weights.sum(dim=-1, keepdim=True)
    
    return scaled_weights

# Rebalancing frequency optimizer
def optimize_rebalancing_frequency(regime_probs, transition_probs, returns, 
                                  transaction_cost=0.001, max_freq=21):
    expected_duration = 1.0 / (1.0 - transition_probs.diagonal(dim1=-2, dim2=-1).mean() + 1e-8)
    
    volatility = returns.std(dim=1).mean()
    
    optimal_freq = torch.sqrt(transaction_cost / (2 * volatility)) * 252
    optimal_freq = optimal_freq.clamp(min=1, max=max_freq)
    
    return optimal_freq.int()

# Leverage constraint optimizer
def optimize_leverage(weights, returns, max_leverage=2.0, target_vol=0.15):
    portfolio_returns = (weights.unsqueeze(1) * returns).sum(dim=-1)
    portfolio_vol = portfolio_returns.std(dim=1)
    
    leverage_multiplier = (target_vol / portfolio_vol).clamp(max=max_leverage)
    
    leveraged_weights = weights * leverage_multiplier.unsqueeze(-1)
    
    return leveraged_weights
