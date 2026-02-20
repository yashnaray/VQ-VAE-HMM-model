import os
import sys
import json
from pathlib import Path
import torch
import numpy as np

# Optional YAML support (not required if using JSON config)
try:
    import yaml
except ImportError:
    yaml = None

# Import the model and dataset utilities from the project (PyTorch version)
try:
    from VQ_VAE_HMM_fixed import VAE_HMM, RandomChunkDataset, collate_fn
except Exception as e:
    VAE_HMM = None
    RandomChunkDataset = None
    collate_fn = None
    print("Warning: could not import VQ_VAE_HMM_fixed. Ensure it's on PYTHONPATH. Error:", e)


def load_config(cfg_path: str):
    if cfg_path.endswith('.json'):
        with open(cfg_path, 'r') as f:
            return json.load(f)
    elif cfg_path.endswith('.yaml') or cfg_path.endswith('.yml'):
        if yaml is None:
            raise RuntimeError("YAML support not available. Install PyYAML or provide a .json config.")
        with open(cfg_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Config must be .json or .yaml/.yml")


def load_sequences(x_path: str, u_path: str):
    def load_one(p: str):
        if p.endswith('.pt') or p.endswith('.pth'):
            return torch.load(p)
        if p.endswith('.npz'):
            import numpy as np
            data = np.load(p)
            # Try common keys
            if 'x' in data:
                x = data['x']
            else:
                x = data[data.files[0]]
            if 'u' in data:
                u = data['u']
            else:
                u = data[data.files[1]]
            return torch.from_numpy(x), torch.from_numpy(u)
        if p.endswith('.npy'):
            arr = np.load(p)
            return torch.from_numpy(arr), None
        raise ValueError(f"Unsupported data format: {p}")
    x_seq = load_one(x_path)
    u_seq = load_one(u_path)
    return x_seq, u_seq


class TrainPipeline:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epochs = 1
        self.model = None
        self.dataloader = None
        self.x_seq = None
        self.u_seq = None

    def load_config(self):
        self.config = load_config(self.config_path)
        if 'training' in self.config:
            self.epochs = int(self.config['training'].get('epochs', 1))
        seed = int(self.config.get('training', {}).get('seed', 42))
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def build_model(self):
        if VAE_HMM is None:
            raise RuntimeError("VAE_HMM model not available. Ensure VQ_VAE_HMM_fixed.py is importable.")
        m = self.config.get('model', {})
        input_dim = m.get('input_dim', 5)
        hidden_dim = m.get('hidden_dim', 64)
        K = m.get('K', 3)
        hidden_dim2 = m.get('hidden_dim2', 32)
        u_dim = m.get('u_dim', None)
        trans_hidden = m.get('trans_hidden', 128)
        self.model = VAE_HMM(input_dim=input_dim, hidden_dim=hidden_dim, K=K,
                             hidden_dim2=hidden_dim2, u_dim=u_dim, trans_hidden=trans_hidden)
        self.model.to(self.device)
        return self.model

    def load_data(self):
        data_cfg = self.config.get('data', {})
        x_path = data_cfg.get('x_sequences_path')
        u_path = data_cfg.get('u_sequences_path')
        if x_path is None or u_path is None:
            raise ValueError("Data config must specify x_sequences_path and u_sequences_path.")
        self.x_seq, self.u_seq = load_sequences(x_path, u_path)
        # Normalize to a list of sequences to satisfy RandomChunkDataset API
        if isinstance(self.x_seq, torch.Tensor):
            self.x_seq = [self.x_seq]
        if isinstance(self.u_seq, torch.Tensor):
            self.u_seq = [self.u_seq]
        min_len = int(data_cfg.get('min_len', 20))
        max_len = int(data_cfg.get('max_len', 200))
        dataset = RandomChunkDataset(self.x_seq, self.u_seq, min_len=min_len, max_len=max_len)
        batch_size = int(self.config.get('training', {}).get('batch_size', 64))
        # Bind collate function to device
        if collate_fn is not None:
            def bound_collate(batch):
                return collate_fn(batch, self.device)
        else:
            def bound_collate(batch):
                return batch
        from torch.utils.data import DataLoader
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=bound_collate)
        return dataset

    def train(self):
        self.load_config()
        self.build_model()
        self.load_data()
        from VQ_VAE_HMM_fixed import train_model
        trained_model = train_model(self.model, self.dataloader, num_epochs=self.epochs,
                                    lr=float(self.config.get('training', {}).get('lr', 1e-3)))
        # Persist final model state
        out_dir = Path(self.config.get('training', {}).get('checkpoint_dir', 'checkpoints'))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'vae_hmm_trained.pth'
        torch.save(trained_model.state_dict(), str(out_path))
        print(f"Training finished. Model saved to {out_path}")
        return trained_model

def main():
    if len(sys.argv) < 2:
        print("Usage: python training_pipeline/train.py <path/to/config.json|config.yaml>")
        sys.exit(1)
    cfg_path = sys.argv[1]
    pipeline = TrainPipeline(cfg_path)
    pipeline.train()


if __name__ == '__main__':
    main()
