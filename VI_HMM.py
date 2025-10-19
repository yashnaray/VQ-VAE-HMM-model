import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim2, K):
        super().__init__()
        # x = (batch_size, input_dim, T)
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim2, kernel_size=3, padding=1)
        self.to_logits = nn.Conv1d(hidden_dim2, K, kernel_size=1)
        # logits = (batch_size, K, T)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        logits = self.to_logits(x) # (B, K, T)
        return logits

class Prior(nn.Module):
    def __init__(self, K, u_dim=None, trans_hidden=128):
        super().__init__()
        self.k_clusters = K
        self.u_dim = u_dim
        # initial state logits (unnormalized) with softmax -> pi (normalized state distribtuion)
        self.log_prior = nn.Parameter(torch.zeros(K), requires_grad=True)

        if u_dim is None:
            raise ValueError('Not supporting stationary transitions')
        else:
            # input-conditioned transitions: small MLP maps u_t -> K*K logits
            self.transition_net = nn.Sequential(
                nn.Linear(u_dim, trans_hidden),
                nn.ReLU(),
                nn.Linear(trans_hidden, K * K)
            )

    def forward(self, u=None):
        # pi = initial state distribution
        log_pi = F.log_softmax(self.log_prior, dim=-1)

        if self.u_dim is None or u is None:
            raise ValueError('Not supporting stationary transitions')

        # input-conditioned case
        if u.dim() == 3 and u.shape[1] == self.u_dim:
            # (B, U_dim, T) -> (B, T, U_dim)
            u = u.permute(0, 2, 1)

        B, T, U = u.shape
        u_flat = u.reshape(B * T, U)
        logits = self.transition_net(u_flat) # (B*T, K*K)
        logits = logits.view(B, T, self.k_clusters, self.k_clusters)
        # normalize by row 
        log_A = F.log_softmax(logits, dim=-1)
        return log_pi, log_A

class Decoder(nn.Module):
    def __init__(self, K, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.K = K
        self.latent_dim = latent_dim
        self.E = nn.Embedding(K, latent_dim)  # embedding for each discrete state
        self.conv1 = nn.Conv1d(latent_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.to_output = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, q):
        # q: (B, T) discrete latent states
        B, K, T = q.shape
        e = torch.matmul(q.permute(0, 2, 1), self.E.weight)
        e_t = e.permute(0, 2, 1)

        x = self.conv1(e_t)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        recon_x = self.to_output(x)
        return recon_x
    

class VAE_HMM(nn.Module):
    def __init__(self, input_dim, hidden_dim, K, hidden_dim2, u_dim=None, trans_hidden=128):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, hidden_dim2, K)
        self.prior = Prior(K, u_dim=u_dim, trans_hidden=trans_hidden)
        self.decoder = Decoder(K, latent_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=input_dim)
        self.K = K
    
    def encode(self, x):
        logits = self.encoder(x)
        return logits
    
    def decode(self, q):
        recon_x = self.decoder(q) 
        return recon_x
    
    def compute_loss(self, x, u=None, lengths=None, beta = 1.0):
        # ELBO loss = reconstruction loss + HMM prior loss - entropy of q -> minimize negative elbo

        B, C, T = x.shape
        # mask of valid timesteps (B: batch size, T)
        if lengths is None:
            raise ValueError('lengths must be provided')
        mask = (torch.arange(T, device=x.device)[None, :] < lengths[:, None].to(x.device))

        log_pi, log_A = self.prior(u)
        logits = self.encoder(x)  # (B, K, T)
        q_probs = F.softmax(logits, dim=1)  # (B, T)
        recon_x = self.decode(q_probs)

        # reconstruction loss (sum over data dim, sum over valid timesteps, average over batch)
        per_t_sq = ((recon_x - x) ** 2).sum(dim=1)  # (B, T)
        recon_loss = (per_t_sq * mask.float()).sum() / B

        # HMM prior loss
        prior_loss = 0.0
        q_probs = F.softmax(logits, dim=1)   # (B, K, T)
        q1 = q_probs[:, :, 0]
        initial_term = (q1 * log_pi.unsqueeze(0)).sum(dim=1)
        q_tm1 = q_probs[:, :, :-1]
        q_tp1 = q_probs[:, :, 1:]
        # reshape
        qm1 = q_tm1.permute(0, 2, 1).unsqueeze(-1)
        q_transition = q_tp1.permute(0, 2, 1).unsqueeze(-2)
        logA_bt = log_A[:, 1:, :, :]
        # elementwise product and sum over i,j
        joint = qm1 * q_transition * logA_bt # (B, T-1, K, K)
        trans_terms_bt = joint.sum(dim=(2, 3))  # (B, T-1)
        trans_mask = (mask[:, 1:] & mask[:, :-1]).float()
        trans_term = (trans_terms_bt * trans_mask).sum(dim=1)

        prior_per_batch = initial_term + trans_term
        prior_loss = - prior_per_batch.mean()

        # entropy (sum over time and states (T, K) -q_t,k log q_t,k) averaged over batch, for only valid timesteps
        log_q = F.log_softmax(logits, dim=1)  # (batch size, K, T)
        # compute per-(B,T)
        per_bt_entropy = - (q_probs * log_q).sum(dim=1)  # (batch size, T)
        entropy = (per_bt_entropy * mask.float()).sum() / B

        return recon_loss + beta * (prior_loss - entropy)


    def forward(self, x):
        logits = self.encode(x)
        q = F.softmax(logits, dim=1)
        recon_x = self.decode(q)
        return recon_x, q