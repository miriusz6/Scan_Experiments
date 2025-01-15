import torch
from torch import nn
from transformers import BartModel

class Simplebart(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.bart = BartModel.from_pretrained('facebook/bart-base')
        self.up = nn.Linear(768, vocab_size)

    def forward(self, x):
        x = self.bart(x)
        x = self.up(x)
        return x
