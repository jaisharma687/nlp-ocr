import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd=64, n_head=2, n_layer=2, block_size=128):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd, nhead=n_head, dim_feedforward=4*n_embd, activation='gelu', batch_first=True
            )
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
