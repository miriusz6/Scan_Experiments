from torch import nn
import torch
from transformer.attention import MultiHeadAttention, TransformerBlock


class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, forward_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.attn = MultiHeadAttention(emb_dim, num_heads)
        self.transformer_block = TransformerBlock(
            emb_dim, num_heads, dropout, forward_dim
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, tgt_mask):
        self_attn = self.attn(x, x, x, tgt_mask)
        x = self.dropout(x + self_attn)
        query = self.ln1(x)
        x = self.transformer_block(query, value, key, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(
        self, vocab_size, emb_dim, num_layers, num_heads, forward_dim, dropout, max_len
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [DecoderBlock(emb_dim, num_heads, forward_dim, dropout) for _ in range(num_layers)]
        )

        self.fc1 = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        embeddings = self.token_emb(x)
        positions = torch.arange(x.shape[1]).expand(x.shape[0], x.shape[1]).to(x.device)
        pos_embeddings = self.pos_emb(positions)
        x = self.dropout(embeddings + pos_embeddings)

        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, src_mask, tgt_mask)
        out = self.fc1(x)
        return out
