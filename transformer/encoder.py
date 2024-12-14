import torch
import math
from torch import nn
from transformer.attention import TransformerBlock


def get_sinusoid_table(max_len, emb_dim):
    def get_angle(pos, i, emb_dim):
        return pos / 10000 ** ((2 * (i // 2)) / emb_dim)

    sinusoid_table = torch.zeros(max_len, emb_dim)
    for pos in range(max_len):
        for i in range(emb_dim):
            if i % 2 == 0:
                sinusoid_table[pos, i] = math.sin(get_angle(pos, i, emb_dim))
            else:
                sinusoid_table[pos, i] = math.cos(get_angle(pos, i, emb_dim))
    return sinusoid_table


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        num_layers,
        num_heads,
        forward_dim,
        dropout,
        max_len,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sinusoid_table(max_len + 1, emb_dim), freeze=True
        )
        self.dropout = nn.Dropout(dropout)
        self.trans_layers = nn.ModuleList(
            [TransformerBlock(emb_dim, num_heads, dropout, forward_dim)] * num_layers
        )

    def forward(self, x, mask):
        positions = torch.arange(x.size(1)).expand(x.size(0), x.size(1)).to(x.device)
        # Shift positions by +1
        positions = positions + 1
        x = self.tok_emb(x) + self.pos_emb(positions)
        x = self.dropout(x)
        # print("Passing through encoder attention heads")
        for layer in self.trans_layers:
            x = layer(x, x, x, mask)
        return x
