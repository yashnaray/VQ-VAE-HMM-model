import torch
from torch.utils.data import Dataset
import random
from typing import Tuple


class RandomChunkDataset(Dataset):
    def __init__(self, x_sequences, u_sequences, min_len: int = 20, max_len: int = 200):
        self.x_seqs = x_sequences
        self.u_seqs = u_sequences
        self.min_len = min_len
        self.max_len = max_len
    
    def __len__(self) -> int:
        return 1000
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        seq_idx = random.randint(0, len(self.x_seqs) - 1)
        x_seq = self.x_seqs[seq_idx]
        u_seq = self.u_seqs[seq_idx]
        
        seq_len = x_seq.shape[1]
        chunk_len = random.randint(self.min_len, min(self.max_len, seq_len))
        start = random.randint(0, seq_len - chunk_len)
        
        return x_seq[:, start:start + chunk_len], u_seq[:, start:start + chunk_len], chunk_len


def collate_fn(batch, device: str = 'cuda'):
    lengths = torch.tensor([item[2] for item in batch], dtype=torch.long)
    max_len = lengths.max().item()
    
    B = len(batch)
    C = batch[0][0].shape[0]
    U = batch[0][1].shape[0]
    
    x_batch = torch.zeros(B, C, max_len, device=device)
    u_batch = torch.zeros(B, U, max_len, device=device)
    
    for i, (x, u, L) in enumerate(batch):
        x_batch[i, :, :L] = x
        u_batch[i, :, :L] = u
    
    return x_batch, u_batch, lengths


def save_checkpoint(model, optimizer, epoch: int, loss: float, path: str):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(model, optimizer, path: str):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
