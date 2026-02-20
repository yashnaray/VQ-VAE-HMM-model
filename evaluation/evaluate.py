#!/usr/bin/env python3
"""Simple evaluation routine for VQ-VAE-HMM.

Loads a trained VAE-HMM, runs a small evaluation on a dataset (or synthetic data),
and reports reconstruction mean-squared error (masked by sequence lengths).
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from VQ_VAE_HMM_fixed import VAE_HMM, RandomChunkDataset, collate_fn


def load_config(cfg_path: str) -> dict:
    if cfg_path.endswith('.json'):
        with open(cfg_path, 'r') as f:
            return json.load(f)
    raise ValueError("Config must be a .json file for this evaluator.")


def load_sequences(x_path: str, u_path: str):
    def load_one(p: str):
        if p.endswith('.pt') or p.endswith('.pth'):
            return torch.load(p)
        raise ValueError(f"Unsupported data format for {p}")
    x_seq = load_one(x_path)
    u_seq = load_one(u_path)
    return x_seq, u_seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='training_pipeline/train_config.json', help='Path to training config (JSON)')
    parser.add_argument('--checkpoint', required=True, help='Path to model state_dict (e.g., vae_hmm_trained.pth)')
    parser.add_argument('--data', nargs='*', default=None, help='Optional paths for data: x_sequences.pt u_sequences.pt')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--output', default='evaluation_reports/eval_results.txt', help='Output file for metrics')
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_args = cfg.get('model', {})
    input_dim = model_args.get('input_dim', 5)
    hidden_dim = model_args.get('hidden_dim', 64)
    K = model_args.get('K', 3)
    hidden_dim2 = model_args.get('hidden_dim2', 32)
    u_dim = model_args.get('u_dim', None)
    trans_hidden = model_args.get('trans_hidden', 128)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VAE_HMM(input_dim=input_dim, hidden_dim=hidden_dim, K=K,
                    hidden_dim2=hidden_dim2, u_dim=u_dim, trans_hidden=trans_hidden)
    model.to(device)
    # Load weights
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Load data (optional)
    if args.data and len(args.data) >= 2:
        x_path, u_path = args.data[0], args.data[1]
        x_seq, u_seq = load_sequences(x_path, u_path)
        min_len, max_len = 20, 200
        dataset = RandomChunkDataset(x_seq, u_seq, min_len=min_len, max_len=max_len)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, device))
    else:
        dataloader = None

    recon_mse_total = 0.0
    batches = 0
    if dataloader is not None:
        with torch.no_grad():
            for x, u, lengths in dataloader:
                x = x.to(device)
                logits = model.encode(x)
                q = F.softmax(logits, dim=1)
                mu, logvar = model.decode(q)
                T = x.shape[2]
                B = x.shape[0]
                mask = torch.arange(T, device=device)[None, :] < lengths[:, None]

                diff = (mu - x)
                recon = (diff ** 2) * mask[:, None, :]
                denom = (mask.sum() * x.shape[1]).item()
                if denom <= 0:
                    denom = 1.0
                mse = recon.sum().item() / denom
                recon_mse_total += mse
                batches += 1
    mean_recon_mse = (recon_mse_total / batches) if batches > 0 else float('nan')

    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(f"Mean Recon MSE: {mean_recon_mse}\n")
    print(f"Evaluation finished. Results saved to {args.output}")


if __name__ == '__main__':
    main()
