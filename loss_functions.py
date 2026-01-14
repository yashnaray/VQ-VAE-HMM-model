import torch
import torch.nn.functional as F
import math

# Multi-objective loss with transaction costs and constraints
def portfolio_loss(weights, returns, prev_weights=None, regime_probs=None, 
                           covariance=None, risk_free_rate=0.0, transaction_cost=0.001,
                           max_weight=0.3, max_leverage=1.0, lambda_turnover=0.1,
                           lambda_drawdown=0.1, lambda_cvar=0.1):
    B, T, n_assets = returns.shape
    portfolio_returns = (weights.unsqueeze(1) * returns).sum(dim=-1)
    
    # Sharpe ratio
    mean_return = portfolio_returns.mean(dim=1)
    std_return = portfolio_returns.std(dim=1).clamp(min=1e-8)
    sharpe = (mean_return - risk_free_rate) / std_return
    
    # Transaction costs
    turnover_loss = 0.0
    if prev_weights is not None:
        turnover = (weights - prev_weights).abs().sum(dim=-1)
        turnover_loss = transaction_cost * turnover.mean()
    
    # Position limit penalty
    position_penalty = F.relu(weights - max_weight).sum(dim=-1).mean()
    
    # Leverage constraint
    leverage = weights.sum(dim=-1)
    leverage_penalty = F.relu(leverage - max_leverage).mean()
    
    # Maximum drawdown
    cumulative_returns = torch.cumsum(portfolio_returns, dim=1)
    running_max = torch.cummax(cumulative_returns, dim=1)[0]
    drawdown = running_max - cumulative_returns
    max_drawdown = drawdown.max(dim=1)[0].mean()
    
    # CVaR (Conditional Value at Risk) at 5%
    alpha = 0.05
    sorted_returns, _ = torch.sort(portfolio_returns, dim=1)
    var_idx = int(alpha * T)
    cvar = -sorted_returns[:, :var_idx].mean()
    
    loss = -sharpe.mean() + lambda_turnover * turnover_loss + \
           position_penalty + leverage_penalty + \
           lambda_drawdown * max_drawdown + lambda_cvar * cvar
    
    return loss

# Sortino ratio loss (downside risk only)
def sortino_loss(weights, returns, risk_free_rate=0.0, target_return=0.0):
    portfolio_returns = (weights.unsqueeze(1) * returns).sum(dim=-1)
    mean_return = portfolio_returns.mean(dim=1)
    downside_returns = torch.clamp(portfolio_returns - target_return, max=0.0)
    downside_std = torch.sqrt((downside_returns ** 2).mean(dim=1)).clamp(min=1e-8)
    sortino = (mean_return - risk_free_rate) / downside_std
    return -sortino.mean()

# Calmar ratio loss (return / max drawdown)
def calmar_loss(weights, returns):
    portfolio_returns = (weights.unsqueeze(1) * returns).sum(dim=-1)
    mean_return = portfolio_returns.mean(dim=1)
    cumulative_returns = torch.cumsum(portfolio_returns, dim=1)
    running_max = torch.cummax(cumulative_returns, dim=1)[0]
    drawdown = running_max - cumulative_returns
    max_drawdown = drawdown.max(dim=1)[0].clamp(min=1e-8)
    calmar = mean_return / max_drawdown
    return -calmar.mean()

# Risk parity loss
def risk_parity_loss(weights, returns, covariance=None):
    if covariance is None:
        B, T, n_assets = returns.shape
        returns_centered = returns - returns.mean(dim=1, keepdim=True)
        covariance = torch.bmm(returns_centered.transpose(1, 2), returns_centered) / T
    
    portfolio_variance = torch.bmm(weights.unsqueeze(1), 
                                   torch.bmm(covariance, weights.unsqueeze(2))).squeeze()
    portfolio_std = torch.sqrt(portfolio_variance.clamp(min=1e-8))
    
    marginal_risk = torch.bmm(covariance, weights.unsqueeze(2)).squeeze()
    risk_contribution = weights * marginal_risk / portfolio_std.unsqueeze(-1)
    
    target_risk = risk_contribution.mean(dim=-1, keepdim=True)
    risk_parity_penalty = ((risk_contribution - target_risk) ** 2).sum(dim=-1).mean()
    
    return risk_parity_penalty

# Regime-conditional covariance loss
def regime_conditional_loss(weights, returns, regime_probs, K):
    B, T, n_assets = returns.shape
    regime_probs_t = regime_probs.permute(0, 2, 1) if regime_probs.shape[1] == K else regime_probs
    
    total_loss = 0.0
    for k in range(K):
        regime_weight = regime_probs_t[:, :, k].unsqueeze(-1)
        weighted_returns = returns * regime_weight
        
        returns_centered = weighted_returns - weighted_returns.mean(dim=1, keepdim=True)
        cov_k = torch.bmm(returns_centered.transpose(1, 2), returns_centered) / T
        
        portfolio_var = torch.bmm(weights.unsqueeze(1), 
                                 torch.bmm(cov_k, weights.unsqueeze(2))).squeeze()
        portfolio_returns = (weights.unsqueeze(1) * weighted_returns).sum(dim=-1)
        mean_return = portfolio_returns.mean(dim=1)
        
        sharpe_k = mean_return / torch.sqrt(portfolio_var.clamp(min=1e-8))
        total_loss -= sharpe_k.mean() * regime_probs_t[:, -1, k].mean()
    
    return total_loss

# Adversarial robustness loss
def adversarial_portfolio_loss(model, regime_probs, returns, epsilon=0.01):
    regime_probs.requires_grad = True
    weights = model(regime_probs)
    portfolio_returns = (weights.unsqueeze(1) * returns).sum(dim=-1)
    loss = -portfolio_returns.mean()
    
    grad = torch.autograd.grad(loss, regime_probs, create_graph=True)[0]
    perturbed_probs = regime_probs + epsilon * grad.sign()
    perturbed_probs = F.softmax(perturbed_probs, dim=1)
    
    weights_adv = model(perturbed_probs)
    portfolio_returns_adv = (weights_adv.unsqueeze(1) * returns).sum(dim=-1)
    
    return -portfolio_returns_adv.mean()

# Transition-aware rebalancing loss
def transition_aware_loss(weights, returns, regime_probs, transition_probs, 
                         rebalance_cost=0.001, lookahead=5):
    B, K, T = regime_probs.shape
    current_regime = regime_probs[:, :, -1]
    
    future_regime_probs = current_regime.unsqueeze(1)
    for _ in range(lookahead):
        future_regime_probs = torch.bmm(future_regime_probs, 
                                       transition_probs[:, -1, :, :])
    
    regime_change_prob = 1.0 - (current_regime * future_regime_probs.squeeze(1)).sum(dim=-1)
    
    portfolio_returns = (weights.unsqueeze(1) * returns).sum(dim=-1)
    mean_return = portfolio_returns.mean(dim=1)
    std_return = portfolio_returns.std(dim=1).clamp(min=1e-8)
    sharpe = (mean_return / std_return)
    
    rebalance_penalty = rebalance_cost * regime_change_prob
    
    return -(sharpe - rebalance_penalty).mean()
