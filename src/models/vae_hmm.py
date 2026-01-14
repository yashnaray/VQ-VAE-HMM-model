import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_dim2: int, K: int):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim2, 3, padding=1)
        self.to_logits = nn.Conv1d(hidden_dim2, K, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        return self.to_logits(h)


class Prior(nn.Module):
    def __init__(self, K: int, u_dim: Optional[int] = None, trans_hidden: int = 128):
        super().__init__()
        self.K = K
        self.u_dim = u_dim
        self.log_prior = nn.Parameter(torch.zeros(K))

        if u_dim is None:
            raise ValueError('Stationary transitions not implemented')
        
        self.transition_net = nn.Sequential(
            nn.Linear(u_dim, trans_hidden),
            nn.ReLU(),
            nn.Linear(trans_hidden, K * K)
        )

    def forward(self, u: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if u is None:
            raise ValueError('u required for non-stationary transitions')

        # handle different input shapes
        if u.dim() == 3 and u.shape[1] == self.u_dim:
            u = u.permute(0, 2, 1)

        B, T, _ = u.shape
        logits = self.transition_net(u.reshape(B * T, -1))
        log_A = F.log_softmax(logits.view(B, T, self.K, self.K), dim=-1)
        
        return F.log_softmax(self.log_prior, dim=-1), log_A


class Decoder(nn.Module):
    def __init__(self, K: int, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(K, latent_dim)
        self.conv1 = nn.Conv1d(latent_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.to_params = nn.Conv1d(hidden_dim, output_dim * 2, 1)

    def forward(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q: (B, K, T) -> embed and decode
        e = torch.matmul(q.permute(0, 2, 1), self.embeddings.weight).permute(0, 2, 1)
        
        h = F.relu(self.conv1(e))
        h = F.relu(self.conv2(h))
        params = self.to_params(h)
        
        mid = params.shape[1] // 2
        return params[:, :mid, :], params[:, mid:, :]


class VAE_HMM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, K: int, 
                 hidden_dim2: int, u_dim: Optional[int] = None, 
                 trans_hidden: int = 128):
        super().__init__()
        self.K = K
        self.encoder = Encoder(input_dim, hidden_dim, hidden_dim2, K)
        self.prior = Prior(K, u_dim, trans_hidden)
        self.decoder = Decoder(K, hidden_dim, hidden_dim, input_dim)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(q)
    
    def compute_loss(self, x: torch.Tensor, u: Optional[torch.Tensor] = None, 
                    lengths: Optional[torch.Tensor] = None, beta: float = 1.0) -> torch.Tensor:
        B, C, T = x.shape
        if lengths is None:
            raise ValueError('lengths required')
        
        mask = torch.arange(T, device=x.device)[None, :] < lengths[:, None].to(x.device)
        log_pi, log_A = self.prior(u)
        logits = self.encoder(x)
        q = F.softmax(logits, dim=1)
        mu, logvar = self.decoder(q)

        # reconstruction: gaussian NLL
        var = logvar.exp().clamp(min=1e-8)
        nll = 0.5 * (torch.log(2 * math.pi * var) + (mu - x)**2 / var)
        recon_loss = (nll * mask.unsqueeze(1).float()).sum() / (mask.sum() * C).clamp(min=1.0)

        # HMM prior: initial + transitions
        init_loss = (q[:, :, 0] * log_pi.unsqueeze(0)).sum(dim=1)
        
        q_prev = q[:, :, :-1].permute(0, 2, 1).unsqueeze(-1)
        q_next = q[:, :, 1:].permute(0, 2, 1).unsqueeze(-2)
        trans_loss = (q_prev * q_next * log_A[:, 1:]).sum(dim=(2, 3))
        trans_mask = (mask[:, 1:] & mask[:, :-1]).float()
        trans_loss = (trans_loss * trans_mask).sum(dim=1)
        
        prior_loss = -(init_loss + trans_loss).mean()

        # entropy regularization
        entropy = -(q * F.log_softmax(logits, dim=1)).sum(dim=1)
        entropy = (entropy * mask.float()).sum() / B

        return recon_loss + beta * (prior_loss - entropy)

    def forward(self, x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        logits = self.encoder(x)
        q = F.softmax(logits, dim=1)
        mu, logvar = self.decoder(q)
        return (mu, logvar), q
