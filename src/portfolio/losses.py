import torch
import torch.nn.functional as F
from typing import Optional


def sharpe_loss(weights: torch.Tensor, returns: torch.Tensor, rf: float = 0.0) -> torch.Tensor:
    port_ret = (weights.unsqueeze(1) * returns).sum(dim=-1)
    mu = port_ret.mean(dim=1)
    sigma = port_ret.std(dim=1).clamp(min=1e-8)
    return -(mu - rf).div(sigma).mean()


def sortino_loss(weights: torch.Tensor, returns: torch.Tensor,
                rf: float = 0.0, target: float = 0.0) -> torch.Tensor:
    port_ret = (weights.unsqueeze(1) * returns).sum(dim=-1)
    mu = port_ret.mean(dim=1)
    downside = torch.clamp(port_ret - target, max=0.0)
    downside_std = torch.sqrt((downside ** 2).mean(dim=1)).clamp(min=1e-8)
    return -(mu - rf).div(downside_std).mean()


def portfolio_loss(weights: torch.Tensor, returns: torch.Tensor,
                           prev_weights: Optional[torch.Tensor] = None,
                           tx_cost: float = 0.001,
                           max_weight: float = 0.3,
                           lambda_turnover: float = 0.1) -> torch.Tensor:
    port_ret = (weights.unsqueeze(1) * returns).sum(dim=-1)
    mu = port_ret.mean(dim=1)
    sigma = port_ret.std(dim=1).clamp(min=1e-8)
    sharpe = mu / sigma
    
    turnover_loss = 0.0
    if prev_weights is not None:
        turnover = (weights - prev_weights).abs().sum(dim=-1)
        turnover_loss = tx_cost * turnover.mean()
    
    position_penalty = F.relu(weights - max_weight).sum(dim=-1).mean()
    
    return -sharpe.mean() + lambda_turnover * turnover_loss + position_penalty


def delta_hedge_loss(hedge_pos: torch.Tensor, spot_ret: torch.Tensor,
                    futures_ret: torch.Tensor, lambda_cost: float = 0.1) -> torch.Tensor:
    hedged_ret = spot_ret + hedge_pos.unsqueeze(1) * futures_ret
    return hedged_ret.var(dim=1).mean()
