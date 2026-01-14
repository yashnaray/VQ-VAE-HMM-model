import torch
import torch.nn as nn
import torch.nn.functional as F


class RegimePortfolioOptimizer(nn.Module):
    def __init__(self, K: int, n_assets: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_assets)
        )
        
    def forward(self, regime_probs: torch.Tensor) -> torch.Tensor:
        if regime_probs.dim() == 3:
            regime_probs = regime_probs[:, :, -1]
        return F.softmax(self.net(regime_probs), dim=-1)


class AttentionPortfolioOptimizer(nn.Module):
    def __init__(self, K: int, n_assets: int, hidden_dim: int = 64, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(K, n_heads, batch_first=True)
        self.fc1 = nn.Linear(K, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_assets)
        
    def forward(self, regime_probs: torch.Tensor) -> torch.Tensor:
        if regime_probs.dim() == 3:
            regime_probs = regime_probs.permute(0, 2, 1)
            out, _ = self.attn(regime_probs, regime_probs, regime_probs)
            regime_probs = out[:, -1, :]
        h = F.relu(self.fc1(regime_probs))
        return F.softmax(self.fc2(h), dim=-1)


class TransformerPortfolioOptimizer(nn.Module):
    def __init__(self, K: int, n_assets: int, hidden_dim: int = 64, 
                 n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.K = K
        layer = nn.TransformerEncoderLayer(K, n_heads, hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(K, n_assets)
        
    def forward(self, regime_seq: torch.Tensor) -> torch.Tensor:
        if regime_seq.dim() == 3 and regime_seq.shape[1] == self.K:
            regime_seq = regime_seq.permute(0, 2, 1)
        out = self.transformer(regime_seq)
        return F.softmax(self.head(out[:, -1]), dim=-1)


class BayesianPortfolioOptimizer(nn.Module):
    def __init__(self, K: int, n_assets: int, hidden_dim: int = 64, n_samples: int = 10):
        super().__init__()
        self.n_samples = n_samples
        self.fc1_mu = nn.Linear(K, hidden_dim)
        self.fc1_logvar = nn.Linear(K, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_assets)
        
    def forward(self, regime_probs: torch.Tensor, 
                return_uncertainty: bool = False) -> torch.Tensor:
        if regime_probs.dim() == 3:
            regime_probs = regime_probs[:, :, -1]
        
        mu = F.relu(self.fc1_mu(regime_probs))
        logvar = self.fc1_logvar(regime_probs)
        
        if self.training or return_uncertainty:
            samples = []
            for _ in range(self.n_samples):
                h = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
                w = F.softmax(self.fc2(h), dim=-1)
                samples.append(w)
            
            weights = torch.stack(samples).mean(dim=0)
            if return_uncertainty:
                return weights, torch.stack(samples).std(dim=0)
            return weights
        
        return F.softmax(self.fc2(mu), dim=-1)
