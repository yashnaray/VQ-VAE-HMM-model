import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np


class MetaPortfolioOptimizer:
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001, n_inner=5):
        self.model = model
        self.inner_lr = inner_lr
        self.n_inner = n_inner
        self.meta_opt = optim.Adam(model.parameters(), lr=outer_lr)
        
    def adapt(self, support_data, loss_fn):
        adapted = {name: param.clone() for name, param in self.model.named_parameters()}
        
        for _ in range(self.n_inner):
            regime_probs, returns = support_data
            weights = self.model(regime_probs)
            loss = loss_fn(weights, returns)
            
            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            adapted = {name: param - self.inner_lr * grad 
                      for (name, param), grad in zip(self.model.named_parameters(), grads)}
        
        return adapted
    
    def meta_update(self, tasks, loss_fn):
        meta_loss = 0.0
        for support_data, query_data in tasks:
            adapted = self.adapt(support_data, loss_fn)
            
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.data = adapted[name]
            
            regime_probs, returns = query_data
            weights = self.model(regime_probs)
            meta_loss += loss_fn(weights, returns)
        
        self.meta_opt.zero_grad()
        meta_loss.backward()
        self.meta_opt.step()
        
        return meta_loss.item()


class OnlinePortfolioOptimizer:
    def __init__(self, model, lr=0.001, ema_decay=0.99):
        self.model = model
        self.opt = optim.Adam(model.parameters(), lr=lr)
        self.ema_decay = ema_decay
        self.ema_params = {name: param.clone().detach() 
                          for name, param in model.named_parameters()}
        
    def update(self, regime_probs, returns, loss_fn):
        weights = self.model(regime_probs)
        loss = loss_fn(weights, returns)
        
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        
        for name, param in self.model.named_parameters():
            self.ema_params[name] = (self.ema_decay * self.ema_params[name] + 
                                    (1 - self.ema_decay) * param.data)
        
        return loss.item()
    
    def use_ema(self):
        for name, param in self.model.named_parameters():
            param.data = self.ema_params[name]


class WalkForwardTrainer:
    def __init__(self, model, loss_fn, train_window=252, test_window=21, 
                 retrain_freq=21, lr=0.001):
        self.model = model
        self.loss_fn = loss_fn
        self.train_window = train_window
        self.test_window = test_window
        self.retrain_freq = retrain_freq
        self.lr = lr
        
    def train_test_split(self, data, start):
        train_end = start + self.train_window
        test_end = train_end + self.test_window
        return data[start:train_end], data[train_end:test_end]
    
    def train_epoch(self, train_data, n_epochs=10):
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        
        for _ in range(n_epochs):
            regime_probs, returns = train_data
            weights = self.model(regime_probs)
            loss = self.loss_fn(weights, returns)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        return loss.item()
    
    def evaluate(self, test_data):
        self.model.eval()
        with torch.no_grad():
            regime_probs, returns = test_data
            weights = self.model(regime_probs)
            port_ret = (weights.unsqueeze(1) * returns).sum(dim=-1)
            sharpe = port_ret.mean() / port_ret.std().clamp(min=1e-8)
        return sharpe.item()
    
    def run(self, full_data, n_periods):
        results = []
        for i in range(0, n_periods * self.retrain_freq, self.retrain_freq):
            train_data, test_data = self.train_test_split(full_data, i)
            train_loss = self.train_epoch(train_data)
            test_sharpe = self.evaluate(test_data)
            results.append({'train_loss': train_loss, 'test_sharpe': test_sharpe})
        return results


def train_advanced_portfolio(model, vae_hmm, dataloader, returns_data, 
                            num_epochs=100, lr=0.001, use_scheduler=True):
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(opt, T_max=num_epochs) if use_scheduler else None
    
    vae_hmm.eval()
    model.train()
    prev_weights = None
    
    for ep in range(num_epochs):
        epoch_loss = 0
        
        for idx, (x, u, lengths) in enumerate(dataloader):
            with torch.no_grad():
                regime_probs = torch.softmax(vae_hmm.encode(x), dim=1)
            
            batch_returns = returns_data[idx]
            
            opt.zero_grad()
            weights = model(regime_probs)
            
            from loss_functions import portfolio_loss
            loss = portfolio_loss(weights, batch_returns, prev_weights, 
                                          regime_probs, risk_free_rate=0.0)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            prev_weights = weights.detach()
            epoch_loss += loss.item()
        
        if scheduler:
            scheduler.step()
        
        print(f"Epoch {ep+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}, LR: {opt.param_groups[0]['lr']:.6f}")
    
    return model
