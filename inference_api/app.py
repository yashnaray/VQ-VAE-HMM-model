import json
import torch
import torch.nn.functional as F
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from VQ_VAE_HMM_fixed import VAE_HMM
except Exception as e:
    raise RuntimeError("Failed to import VAE_HMM model. Ensure VQ_VAE_HMM_fixed.py is on PYTHONPATH. Error: {}".format(e))

app = FastAPI()

MODEL: VAE_HMM = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class InferRequest(BaseModel):
    x: List[List[float]]  # shape: [C, T]


def load_inference_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    # Load from config file in project root
    cfg_path = 'inference_config.json'
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    m_cfg = cfg.get('model', {})
    input_dim = m_cfg.get('input_dim', 5)
    hidden_dim = m_cfg.get('hidden_dim', 64)
    K = m_cfg.get('K', 3)
    hidden_dim2 = m_cfg.get('hidden_dim2', 32)
    u_dim = m_cfg.get('u_dim', None)
    trans_hidden = m_cfg.get('trans_hidden', 128)

    MODEL = VAE_HMM(input_dim=input_dim, hidden_dim=hidden_dim, K=K,
                    hidden_dim2=hidden_dim2, u_dim=u_dim, trans_hidden=trans_hidden)
    # Load checkpoint
    ckpt = cfg.get('checkpoint_path', 'checkpoints/vae_hmm_trained.pth')
    state = torch.load(ckpt, map_location=DEVICE)
    MODEL.load_state_dict(state)
    MODEL.eval()
    return MODEL


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/infer')
def infer(req: InferRequest):
    model = load_inference_model()
    x_list = req.x
    try:
        x_tensor = torch.tensor(x_list, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # [1, C, T]
        with torch.no_grad():
            logits = model.encode(x_tensor)
            q = F.softmax(logits, dim=1)
            mu, logvar = model.decode(q)
        mu_out = mu.squeeze(0).cpu().tolist()
        logvar_out = logvar.squeeze(0).cpu().tolist()
        q_out = q.squeeze(0).cpu().tolist()
        return {
            'mu': mu_out,
            'logvar': logvar_out,
            'regime_probs': q_out
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
