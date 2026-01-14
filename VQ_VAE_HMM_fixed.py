import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RandomChunkDataset:
    def __init__(self, x_sequences, u_sequences, min_len=20, max_len=200):
        self.x_seqs = x_sequences
        self.u_seqs = u_sequences
        self.min_len = min_len
        self.max_len = max_len
    
    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        seq_idx = random.randint(0, len(self.x_seqs) - 1)
        x_seq = self.x_seqs[seq_idx]
        u_seq = self.u_seqs[seq_idx]
        
        seq_len = x_seq.shape[1]
        chunk_len = random.randint(self.min_len, min(self.max_len, seq_len))
        start = random.randint(0, seq_len - chunk_len)
        
        return x_seq[:, start:start + chunk_len], u_seq[:, start:start + chunk_len], chunk_len

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim2, K):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim2, 3, padding=1)
        self.to_logits = nn.Conv1d(hidden_dim2, K, 1)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        return self.to_logits(h)

class Prior(nn.Module):
    def __init__(self, K, u_dim=None, trans_hidden=128):
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

    def forward(self, u=None):
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
    def __init__(self, K, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.embeddings = nn.Embedding(K, latent_dim)
        self.conv1 = nn.Conv1d(latent_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.to_params = nn.Conv1d(hidden_dim, output_dim * 2, 1)

    def forward(self, q):
        # q: (B, K, T) -> embed and decode
        e = torch.matmul(q.permute(0, 2, 1), self.embeddings.weight).permute(0, 2, 1)
        
        h = F.relu(self.conv1(e))
        h = F.relu(self.conv2(h))
        params = self.to_params(h)
        
        mid = params.shape[1] // 2
        return params[:, :mid, :], params[:, mid:, :]

class VAE_HMM(nn.Module):
    def __init__(self, input_dim, hidden_dim, K, hidden_dim2, u_dim=None, trans_hidden=128):
        super().__init__()
        self.K = K
        self.encoder = Encoder(input_dim, hidden_dim, hidden_dim2, K)
        self.prior = Prior(K, u_dim, trans_hidden)
        self.decoder = Decoder(K, hidden_dim, hidden_dim, input_dim)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, q):
        return self.decoder(q)
    
    def compute_loss(self, x, u=None, lengths=None, beta=1.0):
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

    def forward(self, x):
        logits = self.encoder(x)
        q = F.softmax(logits, dim=1)
        mu, logvar = self.decoder(q)
        return (mu, logvar), q

def train_model(model, dataloader, num_epochs=10, lr=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for ep in range(num_epochs):
        epoch_loss = 0
        beta = min(1.0, 2.0 * (ep + 1) / num_epochs)  # KL annealing
        
        for x, u, lengths in dataloader:
            opt.zero_grad()
            loss = model.compute_loss(x, u, lengths, beta)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        print(f"Epoch {ep+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

    return model

def collate_fn(batch):
    lengths = torch.tensor([item[2] for item in batch], dtype=torch.long)
    max_len = lengths.max().item()
    
    B = len(batch)
    C = batch[0][0].shape[0]
    U = batch[0][1].shape[0]

    x_batch = torch.zeros(B, C, max_len, device=device)
    u_batch = torch.zeros(B, U, max_len, device=device)

    for i, (x, u, L) in enumerate(batch):
        x_batch[i, :, :L] = x
        u_batch[i, :, :L] = u

    return x_batch, u_batch, lengths


class RegimePortfolioOptimizer(nn.Module):
    def __init__(self, K, n_assets, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_assets)
        )
        
    def forward(self, regime_probs):
        if regime_probs.dim() == 3:
            regime_probs = regime_probs[:, :, -1]
        return F.softmax(self.net(regime_probs), dim=-1)

def sharpe_loss(weights, returns, rf=0.0):
    port_ret = (weights.unsqueeze(1) * returns).sum(dim=-1)
    mu = port_ret.mean(dim=1)
    sigma = port_ret.std(dim=1).clamp(min=1e-8)
    return -(mu - rf).div(sigma).mean()

class RegimeLSTMOptimizer(nn.Module):
    def __init__(self, K, n_assets, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(K, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, n_assets)
        
    def forward(self, regime_seq):
        out, _ = self.lstm(regime_seq)
        return F.softmax(self.head(out[:, -1]), dim=-1)

def regime_aware_sharpe_loss(weights, returns, regime_probs, trans_probs, rf=0.0):
    port_ret = (weights.unsqueeze(1) * returns).sum(dim=-1)
    confidence = regime_probs.max(dim=-1)[0]
    weighted_ret = port_ret * confidence
    
    # penalize short regime durations
    diag = trans_probs.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    duration = 1.0 / (1.0 - diag + 1e-8)
    penalty = 0.01 / duration.clamp(min=1.0)
    
    mu = weighted_ret.mean(dim=1)
    sigma = weighted_ret.std(dim=1).clamp(min=1e-8)
    sharpe = (mu - rf) / sigma
    
    return -(sharpe.mean() - penalty.mean())

def train_portfolio_optimizer(vae_hmm, port_model, dataloader, returns_data, num_epochs=50, lr=1e-3):
    opt = optim.Adam(port_model.parameters(), lr=lr)
    vae_hmm.eval()
    port_model.train()
    
    for ep in range(num_epochs):
        epoch_loss = 0
        for idx, (x, u, lengths) in enumerate(dataloader):
            with torch.no_grad():
                regime_probs = F.softmax(vae_hmm.encode(x), dim=1)
            
            opt.zero_grad()
            weights = port_model(regime_probs)
            loss = sharpe_loss(weights, returns_data[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {ep+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    
    return port_model
