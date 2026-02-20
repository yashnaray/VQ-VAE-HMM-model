import yfinance as yf
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from VQ_VAE_HMM_fixed import RandomChunkDataset, collate_fn


def download_data(tickers, regime_tickers, start_date='2015-01-01', end_date='2024-01-01'):
    """Download price data and regime indicators."""
    prices_raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, group_by='ticker')
    regime_raw = yf.download(regime_tickers, start=start_date, end=end_date, auto_adjust=True, group_by='ticker')
    
    # Extract Close prices
    if isinstance(prices_raw.columns, pd.MultiIndex):
        prices = prices_raw.xs('Close', level=1, axis=1)
    else:
        prices = prices_raw
    
    if isinstance(regime_raw.columns, pd.MultiIndex):
        regime_data = regime_raw.xs('Close', level=1, axis=1)
    else:
        regime_data = regime_raw
    
    return prices, regime_data


def prepare_sequences(prices, regime_data, lookback=20):
    """Prepare x_sequences (features) and u_sequences (regime indicators)."""
    returns = prices.pct_change().dropna()
    
    # x_sequences: [returns, volume_proxy, volatility, momentum, log_returns]
    volatility = returns.rolling(lookback).std()
    momentum = prices.pct_change(lookback)
    log_returns = np.log1p(returns)
    volume_proxy = returns.abs().rolling(lookback).mean()  # Proxy for volume
    
    x_features = pd.DataFrame({
        'returns': returns.mean(axis=1),
        'volume': volume_proxy.mean(axis=1),
        'volatility': volatility.mean(axis=1),
        'momentum': momentum.mean(axis=1),
        'log_returns': log_returns.mean(axis=1)
    })
    
    # u_sequences: [VIX, yield, market_return, volatility]
    u_features = pd.DataFrame({
        'vix': regime_data['^VIX'],
        'yield': regime_data['^TNX'],
        'market_return': regime_data['SPY'].pct_change(lookback),
        'market_vol': regime_data['SPY'].pct_change().rolling(lookback).std()
    })
    
    # Align and drop NaN
    combined = pd.concat([x_features, u_features], axis=1).dropna()
    x_data = combined[x_features.columns].values
    u_data = combined[u_features.columns].values
    
    return x_data, u_data, returns.loc[combined.index], prices.loc[combined.index]


def create_sequences(x_data, u_data, seq_len=100, stride=20):
    """Create overlapping sequences."""
    x_sequences, u_sequences = [], []
    
    for i in range(0, len(x_data) - seq_len, stride):
        x_sequences.append(x_data[i:i+seq_len])
        u_sequences.append(u_data[i:i+seq_len])
    
    return np.array(x_sequences), np.array(u_sequences)


def load_portfolio_data(tickers=None, start_date='2015-01-01', end_date='2024-01-01'):
    """Complete data loading pipeline."""
    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ', 'WMT', 'PG', 'V', 'UNH', 'HD']
    
    regime_tickers = ['^VIX', '^TNX', 'SPY']
    
    print("Downloading data...")
    prices, regime_data = download_data(tickers, regime_tickers, start_date, end_date)
    
    print("Preparing sequences...")
    x_data, u_data, returns, aligned_prices = prepare_sequences(prices, regime_data)
    
    print("Creating sequences...")
    x_sequences, u_sequences = create_sequences(x_data, u_data)
    
    # Convert to tensors and transpose to (batch, features, time)
    x_sequences = torch.FloatTensor(x_sequences).transpose(1, 2)
    u_sequences = torch.FloatTensor(u_sequences).transpose(1, 2)
    
    print(f"Data shape: x={x_sequences.shape}, u={u_sequences.shape}")
    print(f"Returns shape: {returns.shape}, Prices shape: {aligned_prices.shape}")
    
    return {
        'x_sequences': x_sequences,
        'u_sequences': u_sequences,
        'returns': returns,
        'prices': aligned_prices,
        'tickers': tickers
    }


def create_dataloader(x_sequences, u_sequences, batch_size=32, min_len=20, max_len=100):
    """Create DataLoader with RandomChunkDataset."""
    dataset = RandomChunkDataset(x_sequences, u_sequences, min_len=min_len, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader


if __name__ == '__main__':
    # Load data
    data = load_portfolio_data()
    
    # Create dataloader
    dataloader = create_dataloader(data['x_sequences'], data['u_sequences'])
    
    # Test
    for batch in dataloader:
        x_batch, u_batch, lengths = batch
        print(f"Batch: x={x_batch.shape}, u={u_batch.shape}, lengths={lengths[:5]}")
        break
    
    print("\nData ready for training!")
