import torch
from torch.utils.data import DataLoader
import pandas as pd

from src.models import VAE_HMM
from src.portfolio import RegimePortfolioOptimizer, sharpe_loss
from src.training import Trainer
from src.utils import RandomChunkDataset, collate_fn


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load data
    x_data = pd.read_csv('train_dataset_scaled.csv').drop(columns=['date']).values
    u_data = pd.read_csv('train_dataset_scaled.csv').drop(columns=['date', 'historical_vol']).values
    
    x_sequences = [torch.tensor(x_data, dtype=torch.float).permute(1, 0)]
    u_sequences = [torch.tensor(u_data, dtype=torch.float).permute(1, 0)]
    
    dataset = RandomChunkDataset(x_sequences, u_sequences, min_len=20, max_len=200)
    dataloader = DataLoader(dataset, batch_size=64, collate_fn=lambda b: collate_fn(b, device))
    
    # train VAE-HMM
    print("Training VAE-HMM...")
    vae_hmm = VAE_HMM(
        input_dim=x_data.shape[1],
        hidden_dim=64,
        K=3,
        hidden_dim2=32,
        u_dim=u_data.shape[1]
    )
    
    trainer = Trainer(vae_hmm, lr=1e-5, device=device)
    trainer.train(dataloader, num_epochs=150, use_beta_warmup=True)
    
    torch.save(vae_hmm.state_dict(), 'checkpoints/vae_hmm.pth')
    
    # train portfolio optimizer
    print("\nTraining Portfolio Optimizer...")
    port_model = RegimePortfolioOptimizer(K=3, n_assets=10, hidden_dim=64).to(device)
    opt = torch.optim.Adam(port_model.parameters(), lr=1e-3)
    vae_hmm.eval()
    
    for ep in range(50):
        epoch_loss = 0
        
        for idx, (x, u, lengths) in enumerate(dataloader):
            with torch.no_grad():
                regime_probs = torch.softmax(vae_hmm.encode(x), dim=1)
            
            # simulate returns (replace with actual data)
            returns = torch.randn(x.shape[0], x.shape[2], 10, device=device)
            
            opt.zero_grad()
            weights = port_model(regime_probs)
            loss = sharpe_loss(weights, returns)
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {ep+1}/50, Loss: {epoch_loss/len(dataloader):.4f}")
    
    torch.save(port_model.state_dict(), 'checkpoints/portfolio_model.pth')
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
