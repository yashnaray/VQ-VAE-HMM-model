import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Callable
from tqdm import tqdm


class Trainer:
    def __init__(self, model: nn.Module, lr: float = 1e-3, device: str = 'cuda'):
        self.model = model.to(device)
        self.opt = optim.Adam(model.parameters(), lr=lr)
        self.device = device
        
    def train_epoch(self, dataloader, loss_fn: Optional[Callable] = None, 
                   beta: float = 1.0) -> float:
        self.model.train()
        epoch_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            x, u, lengths = [b.to(self.device) for b in batch]
            
            self.opt.zero_grad()
            
            if loss_fn is None:
                loss = self.model.compute_loss(x, u, lengths, beta)
            else:
                loss = loss_fn(self.model, x, u, lengths)
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(dataloader)
    
    def train(self, dataloader, num_epochs: int = 100, 
             use_beta_warmup: bool = True) -> None:
        for ep in range(num_epochs):
            beta = min(1.0, 2.0 * (ep + 1) / num_epochs) if use_beta_warmup else 1.0
            avg_loss = self.train_epoch(dataloader, beta=beta)
            print(f"Epoch {ep+1}/{num_epochs}, Loss: {avg_loss:.4f}, Beta: {beta:.2f}")


class MetaTrainer:
    def __init__(self, model: nn.Module, inner_lr: float = 0.01, 
                 outer_lr: float = 0.001, n_inner: int = 5):
        self.model = model
        self.inner_lr = inner_lr
        self.n_inner = n_inner
        self.meta_opt = optim.Adam(model.parameters(), lr=outer_lr)
        
    def adapt(self, support_data, loss_fn: Callable):
        adapted = {name: param.clone() 
                  for name, param in self.model.named_parameters()}
        
        for _ in range(self.n_inner):
            loss = loss_fn(self.model, *support_data)
            grads = torch.autograd.grad(loss, self.model.parameters(), 
                                       create_graph=True)
            adapted = {name: param - self.inner_lr * grad 
                      for (name, param), grad in 
                      zip(self.model.named_parameters(), grads)}
        
        return adapted


class WalkForwardTrainer:
    def __init__(self, model: nn.Module, loss_fn: Callable,
                 train_window: int = 252, test_window: int = 21,
                 retrain_freq: int = 21, lr: float = 0.001):
        self.model = model
        self.loss_fn = loss_fn
        self.train_window = train_window
        self.test_window = test_window
        self.retrain_freq = retrain_freq
        self.lr = lr
        
    def run(self, data, n_periods: int):
        results = []
        
        for i in range(0, n_periods * self.retrain_freq, self.retrain_freq):
            train_data = data[i:i + self.train_window]
            test_data = data[i + self.train_window:i + self.train_window + self.test_window]
            
            # train
            opt = optim.Adam(self.model.parameters(), lr=self.lr)
            self.model.train()
            
            for _ in range(10):
                opt.zero_grad()
                loss = self.loss_fn(self.model, *train_data)
                loss.backward()
                opt.step()
            
            # eval
            self.model.eval()
            with torch.no_grad():
                test_loss = self.loss_fn(self.model, *test_data)
            
            results.append({'train_loss': loss.item(), 'test_loss': test_loss.item()})
        
        return results
