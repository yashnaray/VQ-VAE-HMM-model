import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from VQ_VAE_HMM_fixed import VAE_HMM
from portfolio_optimizer import TransformerPortfolioOptimizer

vae_hmm = VAE_HMM(input_dim=5, hidden_dim=64, K=3, hidden_dim2=32, u_dim=4)
portfolio_model = TransformerPortfolioOptimizer(K=3, n_assets=10, hidden_dim=64)

try:
    vae_hmm.load_state_dict(torch.load('models/vae_hmm.pt', map_location='cpu', weights_only=False))
    portfolio_model.load_state_dict(torch.load('models/portfolio_improved.pt', map_location='cpu', weights_only=False))
    vae_hmm.eval()
    portfolio_model.eval()
except Exception as e:
    print(f"Model loading warning: {e}")

def predict_portfolio(market_data, n_assets):
    try:
        lines = [line.strip() for line in market_data.strip().split('\n') if line.strip()]
        data = []
        for line in lines:
            if ',' in line:
                row = [float(x.strip()) for x in line.split(',')]
            elif '\t' in line:
                row = [float(x.strip()) for x in line.split('\t')]
            else:
                row = [float(x.strip()) for x in line.split()]
            data.append(row)
        
        data = np.array(data)
        x = torch.FloatTensor(data).unsqueeze(0).permute(0, 2, 1)
        
        with torch.no_grad():
            logits = vae_hmm.encode(x)
            regime_probs = F.softmax(logits, dim=1)
            weights = portfolio_model(regime_probs)
        
        weights_np = weights.squeeze().cpu().numpy()[:n_assets]
        weights_pct = weights_np * 100
        
        result = "Portfolio Allocation:\n" + "\n".join([f"  Asset {i+1}: {w:.2f}%" for i, w in enumerate(weights_pct)])
        result += f"\n\nTotal: {weights_pct.sum():.2f}%"
        
        regime_probs_avg = regime_probs.mean(dim=2).squeeze().cpu().numpy()
        regime_names = ["Bull Market", "Bear Market", "Neutral Market"]
        regime_str = "\n\nMarket Regime Detection:\n" + "\n".join(
            [f"  {regime_names[i]}: {p*100:.1f}%" for i, p in enumerate(regime_probs_avg)]
        )
        
        return result + regime_str
    except Exception as e:
        return f"Error: {str(e)}\n\nTip: Make sure your data is formatted correctly (comma-separated, one row per time period)"

with gr.Blocks(title="VQ-VAE-HMM Portfolio Optimizer") as demo:
    gr.Markdown("""# VQ-VAE-HMM Portfolio Optimizer
    
    This tool uses regime-switching models to optimize portfolio allocation across different market conditions.
    
    ### How to use:
    1. Enter historical market data (prices or returns) for your assets
    2. Select how many assets you want to allocate to
    3. Click "Optimize" to get optimal portfolio weights
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input Data")
            market_input = gr.Textbox(
                label="Market Data (one row per time period)", 
                lines=10, 
                placeholder="Example:\n1.2,3.4,5.6,7.8,9.0\n1.3,3.5,5.7,7.9,9.1\n1.1,3.3,5.5,7.7,8.9",
                info="Enter comma, tab, or space-separated values. Each row = one time period, each column = one asset."
            )
            n_assets = gr.Slider(
                minimum=1, 
                maximum=10, 
                value=5, 
                step=1, 
                label="Number of Assets to Allocate",
                info="How many assets should receive portfolio weight?"
            )
            submit_btn = gr.Button("Optimize Portfolio", variant="primary")
        
        with gr.Column():
            gr.Markdown("### Results")
            output = gr.Textbox(
                label="Portfolio Weights & Market Regimes", 
                lines=15,
                info="Optimal allocation percentages and detected market regime probabilities"
            )
    
    submit_btn.click(fn=predict_portfolio, inputs=[market_input, n_assets], outputs=output)
    
    gr.Markdown("### Example")
    gr.Examples(
        examples=[
            ["1.2,3.4,5.6,7.8,9.0\n1.3,3.5,5.7,7.9,9.1\n1.1,3.3,5.5,7.7,8.9\n1.4,3.6,5.8,8.0,9.2\n1.0,3.2,5.4,7.6,8.8", 5]
        ],
        inputs=[market_input, n_assets],
        label="Click to load sample data"
    )
    
    gr.Markdown("""---
    **Note:** This model detects 3 market regimes (bull, bear, neutral) and optimizes portfolio weights accordingly.
    Weights sum to 1.0 (100% allocation).""")

if __name__ == "__main__":
    demo.launch()
