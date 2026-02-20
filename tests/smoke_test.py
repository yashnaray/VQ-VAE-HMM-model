#!/usr/bin/env python3
"""Smoke test for the VQ-VAE-HMM modules.

This test imports core components and runs a tiny forward pass to ensure
the modules cooperate end-to-end without requiring large datasets.
"""
import torch
import torch.nn.functional as F

try:
    from VQ_VAE_HMM_fixed import VAE_HMM
except Exception as e:
    raise SystemExit(f"Import failed for VAE_HMM_fixed: {e}")


def main():
    # Minimal model configuration for smoke test
    input_dim = 5
    hidden_dim = 8
    K = 3
    hidden_dim2 = 4
    u_dim = 2
    trans_hidden = 8
    model = VAE_HMM(input_dim=input_dim, hidden_dim=hidden_dim, K=K,
                    hidden_dim2=hidden_dim2, u_dim=u_dim, trans_hidden=trans_hidden)
    model.eval()
    device = 'cpu'
    model.to(device)

    # Tiny random input: [B, C, T]
    B, C, T = 1, input_dim, 16
    x = torch.randn(B, C, T).to(device)
    with torch.no_grad():
        logits = model.encode(x)
        q = F.softmax(logits, dim=1)
        mu, logvar = model.decode(q)
        # Basic shape checks
        assert mu.shape == x.shape
        assert logvar.shape == x.shape
        print("Smoke test passed: forward shapes OK.")


if __name__ == '__main__':
    main()
