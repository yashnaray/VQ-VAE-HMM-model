from flask import Flask, request, jsonify
import torch
import numpy as np
from VQ_VAE_HMM_fixed import VAE_HMM
from portfolio_optimizer import TransformerPortfolioOptimizer

app = Flask(__name__)

vae_hmm = VAE_HMM(input_dim=5, hidden_dim=64, K=3, hidden_dim2=32, u_dim=4)
portfolio_model = TransformerPortfolioOptimizer(K=3, n_assets=10, hidden_dim=64)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    x = torch.FloatTensor(data).unsqueeze(0)
    
    with torch.no_grad():
        regime_probs = vae_hmm.get_regime_probs(x, torch.zeros(1, x.size(1), 4))
        weights = portfolio_model(regime_probs, x)
    
    return jsonify({'weights': weights.squeeze().tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
