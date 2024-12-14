import torch
from torch import nn
from transformer.encoder import Encoder
from transformer.decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        emb_dim=512,
        num_layers=6,
        num_heads=8,
        forward_dim=2048,
        dropout=0.0,
        max_len=128,
    ):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.num_heads = num_heads
        self.encoder = Encoder(
            src_vocab_size,
            emb_dim,
            num_layers,
            num_heads,
            forward_dim,
            dropout,
            max_len,
        )
        self.decoder = Decoder(
            tgt_vocab_size,
            emb_dim,
            num_layers,
            num_heads,
            forward_dim,
            dropout,
            max_len,
        )

    def create_src_mask(self, src):
        device = src.device
        # (batch_size, 1, 1, src_seq_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(device)

    def create_tgt_mask(self, tgt):
        device = tgt.device
        batch_size, tgt_len = tgt.shape
        tgt_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask * torch.tril(torch.ones((tgt_len, tgt_len))).expand(
            batch_size, 1, tgt_len, tgt_len
        ).to(device)
        return tgt_mask

    def forward(self, src, tgt):
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)

        encoder_out = self.encoder(src, src_mask)
        decoder_out = self.decoder(tgt, encoder_out, src_mask, tgt_mask)
        return decoder_out
