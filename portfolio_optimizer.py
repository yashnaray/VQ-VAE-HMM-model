import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionPortfolioOptimizer(nn.Module):
    def __init__(self, K, n_assets, hidden_dim=64, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(K, n_heads, batch_first=True)
        self.fc1 = nn.Linear(K, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_assets)
        
    def forward(self, regime_probs):
        if regime_probs.dim() == 3:
            regime_probs = regime_probs.permute(0, 2, 1)
            out, _ = self.attn(regime_probs, regime_probs, regime_probs)
            regime_probs = out[:, -1, :]
        h = F.relu(self.fc1(regime_probs))
        return F.softmax(self.fc2(h), dim=-1)


class TransformerPortfolioOptimizer(nn.Module):
    def __init__(self, K, n_assets, hidden_dim=64, n_layers=2, n_heads=1):
        super().__init__()
        self.K = K
        layer = nn.TransformerEncoderLayer(K, n_heads, hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(K, n_assets)
        
    def forward(self, regime_seq):
        if regime_seq.dim() == 3 and regime_seq.shape[1] == self.K:
            regime_seq = regime_seq.permute(0, 2, 1)
        out = self.transformer(regime_seq)
        return F.softmax(self.head(out[:, -1]), dim=-1)


class BayesianPortfolioOptimizer(nn.Module):
    def __init__(self, K, n_assets, hidden_dim=64, n_samples=10):
        super().__init__()
        self.n_samples = n_samples
        self.fc1_mu = nn.Linear(K, hidden_dim)
        self.fc1_logvar = nn.Linear(K, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_assets)
        
    def forward(self, regime_probs, return_uncertainty=False):
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


class EnsemblePortfolioOptimizer(nn.Module):
    def __init__(self, K, n_assets, n_models=5, hidden_dim=64):
        super().__init__()
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(K, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_assets)
            ) for _ in range(n_models)
        ])
        
    def forward(self, regime_probs):
        if regime_probs.dim() == 3:
            regime_probs = regime_probs[:, :, -1]
        weights = [F.softmax(m(regime_probs), dim=-1) for m in self.models]
        return torch.stack(weights).mean(dim=0)


class HierarchicalPortfolioOptimizer(nn.Module):
    def __init__(self, K, n_assets, hidden_dim=64):
        super().__init__()
        self.macro = nn.Sequential(nn.Linear(K, hidden_dim), nn.ReLU())
        self.micro = nn.Sequential(nn.Linear(hidden_dim + K, hidden_dim), nn.ReLU())
        self.head = nn.Linear(hidden_dim, n_assets)
        
    def forward(self, regime_probs):
        if regime_probs.dim() == 3:
            regime_probs = regime_probs[:, :, -1]
        macro_feat = self.macro(regime_probs)
        micro_feat = self.micro(torch.cat([macro_feat, regime_probs], dim=-1))
        return F.softmax(self.head(micro_feat), dim=-1)
