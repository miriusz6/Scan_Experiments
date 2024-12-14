from torch import nn
import torch
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.query = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.key = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.value = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.out_proj = nn.Linear(self.head_dim * num_heads, emb_dim)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, emb_dim = x.size()
        # Reshape and transpose to get heads dim second
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = x.size()
        return (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, num_heads * head_dim)
        )

    def scaled_dot_product_attention(self, q, k, v: torch.Tensor, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        # (batch_size, num_heads, num_query_seq, num_key_seq)
        assert attn_scores.size() == (q.size(0), self.num_heads, q.size(2), k.size(2))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e20)
        attn_probs = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, v)

    def forward(self, query, key, value, mask=None):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        attention_out = self.scaled_dot_product_attention(q, k, v, mask)
        combined_out = self.combine_heads(attention_out)
        return self.out_proj(combined_out)


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, forward_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.ln2 = nn.LayerNorm(emb_dim, eps=1e-6)
        #self.dropout = dropout
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, forward_dim), nn.ReLU(), nn.Linear(forward_dim, emb_dim)
        )
        self.attn = MultiHeadAttention(emb_dim, num_heads)

        

    def forward(self, query, key, value, mask):
        attn_out = self.attn(query, key, value, mask)
        skip1 = query + attn_out
        skip1 = self.dropout(skip1)
        skip1 = self.ln1(skip1)
        ffn_out = self.ffn(skip1)
        skip2 = skip1 + ffn_out
        skip2 = self.dropout(skip2)
        out = self.ln2(skip2)
        return out
