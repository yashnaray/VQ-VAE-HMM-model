import gradio as gr
import torch
import numpy as np
import pandas as pd
from VQ_VAE_HMM_fixed import VAE_HMM
from portfolio_optimizer import TransformerPortfolioOptimizer

vae_hmm = VAE_HMM(input_dim=5, hidden_dim=64, K=3, hidden_dim2=32, u_dim=4)
portfolio_model = TransformerPortfolioOptimizer(K=3, n_assets=10, hidden_dim=64)

try:
    vae_hmm.load_state_dict(torch.load('models/vae_hmm.pt', map_location='cpu'))
    portfolio_model.load_state_dict(torch.load('models/portfolio_improved.pt', map_location='cpu'))
    vae_hmm.eval()
    portfolio_model.eval()
except Exception as e:
    print(f"Model loading warning: {e}")

def predict_portfolio(market_data, n_assets):
    try:
        data = np.array([[float(x) for x in row.split(',')] for row in market_data.strip().split('\n')])
        x = torch.FloatTensor(data).unsqueeze(0)
        u = torch.zeros(1, x.size(1), 4)
        
        with torch.no_grad():
            regime_probs = vae_hmm.get_regime_probs(x, u)
            weights = portfolio_model(regime_probs, x[:, :, :n_assets])
        
        weights_np = weights.squeeze().cpu().numpy()
        result = "\n".join([f"Asset {i+1}: {w:.4f}" for i, w in enumerate(weights_np)])
        regime_str = "\n\nRegime Probabilities:\n" + "\n".join([f"Regime {i+1}: {p:.3f}" for i, p in enumerate(regime_probs.squeeze()[-1].cpu().numpy())])
        
        return result + regime_str
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(title="VQ-VAE-HMM Portfolio Optimizer") as demo:
    gr.Markdown("# VQ-VAE-HMM Portfolio Optimization")
    gr.Markdown("Enter market data (comma-separated values, one timestep per line)")
    
    with gr.Row():
        with gr.Column():
            market_input = gr.Textbox(label="Market Data", lines=10, placeholder="1.2,3.4,5.6,7.8,9.0\n1.3,3.5,5.7,7.9,9.1")
            n_assets = gr.Slider(minimum=1, maximum=10, value=10, step=1, label="Number of Assets")
            submit_btn = gr.Button("Optimize Portfolio")
        
        with gr.Column():
            output = gr.Textbox(label="Portfolio Weights & Regimes", lines=15)
    
    submit_btn.click(fn=predict_portfolio, inputs=[market_input, n_assets], outputs=output)
    
    gr.Examples(
        examples=[["1.2,3.4,5.6,7.8,9.0\n1.3,3.5,5.7,7.9,9.1\n1.1,3.3,5.5,7.7,8.9", 5]],
        inputs=[market_input, n_assets]
    )

if __name__ == "__main__":
    demo.launch()
