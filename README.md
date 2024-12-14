### Transformer Boilerplate - Advanced Topics in NLP

The original notebook is included, but the code has been put into the `transformer` package which is recommended.
The notebook should only serve as a reference.

Example usage as taken from the original jupyer notebook provided:
```python
from transformer import Transformer

# Create a transformer model
model = Transformer(
    src_vocab_size=200, tgt_vocab_size=220, src_pad_idx=0, tgt_pad_idx=0, dropout=0.1
)

# source input: batch size 4, sequence length of 75
src_in = torch.randint(0, 200, (4, 75))  # .to(device)

# target input: batch size 4, sequence length of 80
tgt_in = torch.randint(0, 220, (4, 80))  # .to(device)

# expected output shape of the model
expected_out_shape = torch.Size([4, 80, 220])

with torch.no_grad():
    out = model(src_in, tgt_in)

assert (
    out.shape == expected_out_shape
), f"wrong output shape, expected: {expected_out_shape}"
```

