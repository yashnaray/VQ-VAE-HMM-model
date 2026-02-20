import torch
import torch.nn.functional as F
from data_loader import load_portfolio_data, create_dataloader
from VQ_VAE_HMM_fixed import VAE_HMM, train_model, RegimePortfolioOptimizer, sharpe_loss


TICKERS = [
    'SPY',   
    'QQQ',   
    'IWM',  
    'EFA',
    'TLT',    
    'LQD', 
    'GLD',   
    'DBC', 
    'XLE', 
    'XLF',
]

START_DATE = '2010-01-01'  
END_DATE = '2024-01-01'

# Training params
VQE_EPOCHS = 100 
PORT_EPOCHS = 100
BATCH_SIZE = 16
VQE_LR = 5e-6 
PORT_LR = 1e-3


def train_vqvae_hmm(data, num_epochs=100, lr=5e-5, batch_size=16):
    """Train VQ-VAE-HMM model."""
    print("\n Training VQ-VAE-HMM")
    
    model = VAE_HMM(input_dim=5, hidden_dim=64, K=3, hidden_dim2=32, u_dim=4)
    dataloader = create_dataloader(data['x_sequences'], data['u_sequences'], batch_size=batch_size)
    
    trained_model = train_model(model, dataloader, num_epochs=num_epochs, lr=lr)
    
    print("VQ-VAE-HMM training complete")
    return trained_model


def train_portfolio(vae_hmm, data, num_epochs=100, lr=1e-3, batch_size=16):
    """Train portfolio optimizer."""
    print("\n Training Portfolio Optimizer ")
    
    n_assets = data['returns'].shape[1]
    portfolio_model = RegimePortfolioOptimizer(K=3, n_assets=n_assets, hidden_dim=64)
    
    optimizer = torch.optim.Adam(portfolio_model.parameters(), lr=lr)
    vae_hmm.eval()
    portfolio_model.train()
    
    returns_tensor = torch.FloatTensor(data['returns'].values)
    dataloader = create_dataloader(data['x_sequences'], data['u_sequences'], batch_size=batch_size)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        n_batches = 0
        
        for x_batch, u_batch, lengths in dataloader:
            with torch.no_grad():
                regime_probs = F.softmax(vae_hmm.encode(x_batch), dim=1)
            
            optimizer.zero_grad()
            weights = portfolio_model(regime_probs)
            
            batch_size_actual = weights.shape[0]
            returns_batch = returns_tensor[torch.randint(0, len(returns_tensor), (batch_size_actual, 20))]
            
            loss = sharpe_loss(weights, returns_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(portfolio_model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/n_batches:.4f}")
    
    print("Portfolio optimizer training complete")
    return portfolio_model


def save_models(vae_hmm, portfolio_model, path='models/'):
    """Save trained models."""
    import os
    os.makedirs(path, exist_ok=True)
    torch.save(vae_hmm.state_dict(), f'{path}vae_hmm.pt')
    torch.save(portfolio_model.state_dict(), f'{path}portfolio.pt')
    print(f"\nModels saved to {path}")


if __name__ == '__main__':
    print("=" * 60)
    print("IMPROVED TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Assets: {len(TICKERS)}")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"VQE Epochs: {VQE_EPOCHS}, Portfolio Epochs: {PORT_EPOCHS}")
    print("=" * 60)
    
    data = load_portfolio_data(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    print(f"\nData loaded: {data['returns'].shape[0]} days, {data['x_sequences'].shape[0]} sequences")
    vae_hmm = train_vqvae_hmm(data, num_epochs=VQE_EPOCHS, lr=VQE_LR, batch_size=BATCH_SIZE)
    portfolio_model = train_portfolio(vae_hmm, data, num_epochs=PORT_EPOCHS, lr=PORT_LR, batch_size=BATCH_SIZE)
    save_models(vae_hmm, portfolio_model)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Run 'python inference.py' to test the trained models")
