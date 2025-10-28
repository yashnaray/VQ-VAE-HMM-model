import random
import torch
from torch.utils.data import Dataset

class RandomChunkDataset(Dataset):
    """Dataset for sampling random-length chunks from long sequences.
    Assumes `sequences` is a list or array-like of tensors with shape (C, T_full)
    and `u_sequences` parallels it with shape (u_dim, T_full) or None.

    On __getitem__(i) we sample a random sequence index, a random length in [min_len,max_len]
    and a random start position such that the window fits.
    Returns (x_chunk, u_chunk, length) where x_chunk: (C, L), u_chunk: (u_dim, L)
    """
    def __init__(self, sequences, u_sequences=None, min_len=20, max_len=200):
        self.sequences = sequences
        self.u_sequences = u_sequences
        self.min_len = min_len
        self.max_len = max_len

    def __len__(self):
        return sum(seq.shape[1] for seq in self.sequences)  # rough

    def __getitem__(self, idx):
        # randomly pick a sequence
        i = random.randrange(len(self.sequences))
        x_full = self.sequences[i]
        T_full = x_full.shape[1]
        L = random.randint(self.min_len, min(self.max_len, T_full))
        if T_full == L:
            start = 0
        else:
            start = random.randint(0, T_full - L)
            
        x_chunk = x_full[:, start:start+L]
        u_chunk = self.u_sequences[i][:, start:start+L]
        return x_chunk, u_chunk, torch.tensor(L, dtype=torch.long)
