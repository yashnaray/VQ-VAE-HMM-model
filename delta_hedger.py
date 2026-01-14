import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RegimeDeltaHedger(nn.Module):
    def __init__(self, K, n_assets, hidden_dim=64):
        super().__init__()
        self.delta_net = nn.Sequential(
            nn.Linear(K + n_assets, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_assets)
        )
        self.uncertainty_net = nn.Sequential(
            nn.Linear(K, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, regime_probs, spot_prices, portfolio_pos):
        if regime_probs.dim() == 3:
            regime_probs = regime_probs[:, :, -1]
        
        features = torch.cat([regime_probs, portfolio_pos], dim=-1)
        delta = torch.tanh(self.delta_net(features))
        uncertainty = self.uncertainty_net(regime_probs)
        adjusted = delta * uncertainty
        hedge = -adjusted * portfolio_pos
        
        return hedge, delta


class DynamicDeltaHedger(nn.Module):
    def __init__(self, K, n_assets, hidden_dim=64, use_gamma=True):
        super().__init__()
        self.use_gamma = use_gamma
        input_dim = K + n_assets * 2 + (n_assets if use_gamma else 0)
        
        self.delta_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_assets)
        )
        
        if use_gamma:
            self.gamma_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_assets)
            )
    
    def forward(self, regime_probs, spot_prices, portfolio_pos, gamma=None):
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


class LSTMDeltaHedger(nn.Module):
    def __init__(self, K, n_assets, hidden_dim=64, num_layers=2, lookback=10):
        super().__init__()
        self.lookback = lookback
        self.lstm = nn.LSTM(K + n_assets, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, n_assets)
        
    def forward(self, regime_seq, price_seq):
        if regime_seq.dim() == 3 and regime_seq.shape[1] != regime_seq.shape[2]:
            regime_seq = regime_seq.permute(0, 2, 1)
        
        combined = torch.cat([regime_seq, price_seq], dim=-1)
        out, _ = self.lstm(combined)
        return torch.tanh(self.head(out[:, -1]))


def minimum_variance_hedge_ratio(spot_ret, futures_ret, regime_probs=None, K=None):
    if regime_probs is not None and K is not None:
        B, T, n_assets = spot_ret.shape
        regime_probs_t = regime_probs.permute(0, 2, 1) if regime_probs.shape[1] == K else regime_probs
        
        hedge_ratios = []
        for k in range(K):
            weight = regime_probs_t[:, :, k].unsqueeze(-1)
            
            spot_w = spot_ret * weight
            futures_w = futures_ret * weight
            
            cov = (spot_w * futures_w).sum(dim=1) / weight.sum(dim=1).clamp(min=1e-8)
            var = (futures_w ** 2).sum(dim=1) / weight.sum(dim=1).clamp(min=1e-8)
            
            hedge_ratios.append(cov / var.clamp(min=1e-8))
        
        hedge_ratios = torch.stack(hedge_ratios, dim=1)
        return (hedge_ratios * regime_probs_t[:, -1, :].unsqueeze(-1)).sum(dim=1)
    else:
        cov = (spot_ret * futures_ret).mean(dim=1)
        var = (futures_ret ** 2).mean(dim=1).clamp(min=1e-8)
        return cov / var


class TransactionCostAwareHedger(nn.Module):
    def __init__(self, K, n_assets, hidden_dim=64, tx_cost=0.001):
        super().__init__()
        self.tx_cost = tx_cost
        
        self.hedge_net = nn.Sequential(
            nn.Linear(K + n_assets * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_assets)
        )
        self.threshold_net = nn.Sequential(
            nn.Linear(K, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, regime_probs, current_hedge, target_delta, spot_prices):
        if regime_probs.dim() == 3:
            regime_probs = regime_probs[:, :, -1]
        
        features = torch.cat([regime_probs, current_hedge, spot_prices], dim=-1)
        optimal = self.hedge_net(features)
        threshold = self.threshold_net(regime_probs) * 0.1
        
        deviation = (optimal - current_hedge).abs()
        rehedge_mask = (deviation > threshold).float()
        new_hedge = current_hedge + (optimal - current_hedge) * rehedge_mask
        
        trade_size = (new_hedge - current_hedge).abs()
        cost = self.tx_cost * trade_size * spot_prices
        
        return new_hedge, cost.sum(dim=-1)


class TransitionAwareHedger(nn.Module):
    def __init__(self, K, n_assets, hidden_dim=64, lookahead=5):
        super().__init__()
        self.lookahead = lookahead
        self.hedge_net = nn.Sequential(
            nn.Linear(K * (lookahead + 1) + n_assets, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_assets)
        )
        
    def forward(self, regime_probs, trans_matrix, spot_prices):
        if regime_probs.dim() == 3:
            regime_probs = regime_probs[:, :, -1]
        
        # predict future regimes
        future_probs = [regime_probs]
        current = regime_probs.unsqueeze(1)
        
        for _ in range(self.lookahead):
            next_probs = torch.bmm(current, trans_matrix[:, -1, :, :])
            future_probs.append(next_probs.squeeze(1))
            current = next_probs
        
        all_probs = torch.cat(future_probs, dim=-1)
        features = torch.cat([all_probs, spot_prices], dim=-1)
        
        return torch.tanh(self.hedge_net(features))


def delta_hedge_loss(hedge_pos, spot_ret, futures_ret, tx_costs=None, lambda_cost=0.1):
    hedged_ret = spot_ret + hedge_pos.unsqueeze(1) * futures_ret
    hedge_var = hedged_ret.var(dim=1).mean()
    
    cost_penalty = 0.0
    if tx_costs is not None:
        cost_penalty = tx_costs.mean()
    
    return hedge_var + lambda_cost * cost_penalty


def optimal_hedge_frequency(spot_vol, tx_cost, regime_persistence):
    # based on Leland (1985)
    base_freq = torch.sqrt(8 * tx_cost / (np.pi * spot_vol ** 2))
    return base_freq * (1.0 / regime_persistence.clamp(min=1.0))


def train_delta_hedger(hedger, vae_hmm, spot_data, futures_data, num_epochs=50, lr=0.001):
    opt = torch.optim.Adam(hedger.parameters(), lr=lr)
    vae_hmm.eval()
    hedger.train()
    
    for ep in range(num_epochs):
        epoch_loss = 0
        
        for idx, (x, u, lengths) in enumerate(spot_data):
            with torch.no_grad():
                regime_probs = F.softmax(vae_hmm.encode(x), dim=1)
            
            spot_ret = x[:, :, 1:] - x[:, :, :-1]
            futures_ret = futures_data[idx]
            
            opt.zero_grad()
            
            if isinstance(hedger, LSTMDeltaHedger):
                hedge_ratios = hedger(regime_probs, x)
            else:
                portfolio_pos = torch.ones_like(x[:, :, -1])
                hedge_ratios, _ = hedger(regime_probs, x[:, :, -1], portfolio_pos)
            
            loss = delta_hedge_loss(hedge_ratios, spot_ret, futures_ret)
            loss.backward()
            nn.utils.clip_grad_norm_(hedger.parameters(), 1.0)
            opt.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {ep+1}/{num_epochs}, Loss: {epoch_loss/len(spot_data):.6f}")
    
    return hedger
