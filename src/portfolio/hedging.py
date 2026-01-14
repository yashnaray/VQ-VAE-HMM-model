import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RegimeDeltaHedger(nn.Module):
    def __init__(self, K: int, n_assets: int, hidden_dim: int = 64):
        super().__init__()
        self.delta_net = nn.Sequential(
            nn.Linear(K + n_assets, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_assets)
        )
        self.uncertainty_net = nn.Sequential(
            nn.Linear(K, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, regime_probs: torch.Tensor, portfolio_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if regime_probs.dim() == 3:
            regime_probs = regime_probs[:, :, -1]
        
        features = torch.cat([regime_probs, portfolio_pos], dim=-1)
        delta = torch.tanh(self.delta_net(features))
        uncertainty = self.uncertainty_net(regime_probs)
        adjusted = delta * uncertainty
        hedge = -adjusted * portfolio_pos
        
        return hedge, delta


class DynamicDeltaHedger(nn.Module):
    def __init__(self, K: int, n_assets: int, hidden_dim: int = 64, use_gamma: bool = True):
        super().__init__()
        self.use_gamma = use_gamma
        input_dim = K + n_assets * 2 + (n_assets if use_gamma else 0)
        
        self.delta_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_assets)
        )
        
        if use_gamma:
            self.gamma_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_assets)
            )
    
    def forward(self, regime_probs: torch.Tensor, spot_prices: torch.Tensor,
                portfolio_pos: torch.Tensor, gamma: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if regime_probs.dim() == 3:
            regime_probs = regime_probs[:, :, -1]
        
        features = [regime_probs, portfolio_pos, spot_prices]
        if self.use_gamma and gamma is not None:
            features.append(gamma)
        
        x = torch.cat(features, dim=-1)
        delta = self.delta_net(x)
        
        if self.use_gamma and gamma is not None:
            gamma_hedge = self.gamma_net(x)
            total = delta + 0.5 * gamma_hedge * spot_prices
        else:
            total = delta
        
        return total, delta
