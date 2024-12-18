# %% [markdown]
# # Building the [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) "Vanilla" Transformer from scratch.

# %%
import math

import torch
import torch.nn.functional as F

# %% [markdown]
# # Task 1: Transformers from Scratch
# Here we will build the famous 2017 Transformer Encoder-Decoder from the Paper [Attention is All You Need](https://arxiv.org/abs/1706.03762).
#

# We will start by implementing Multi-Head Attention, which concatenates multiple single scaled dot-product attention (SDPA) modules along the number of attention heads we desire. However, as concatenation implies sequential procedures, we will directly implement multi-head attention as a tensor operation on `nn.Linear()` layers by dividing them into `num_heads` subparts and calculating SDPA on each of them. By doing this, we entirely avoided sequential calculations.
#
# In order to have trainable parameters, we can conveniently build all modules using torch's `nn.Module` functionality.
#
# * Our module's `__init__()` method takes in the embedding dimension `emb_dim` of our transformer, as well as the number of heads `num_heads`.
#     * It stores the `head_dim = emb_dim // num_heads`
# * We create 4 linear layers
#     * The linear layers for query, key, and value each have `(emb_dim, num_heads * head_dim)` size
#     * The output linear layer needs to take the `num_heads * head_dim` as input size, and outputs the original model embedding dimension `emb_dim`
# * The `forward()` method of this module takes in `query`, `key`, `value`, and an optional `mask`, and performs the calculations of the following formula:
#     * Remember that our input at this stage has dimensions `(batch_size, seq_len, emb_dim)`
#     * We pass `query`, `key`, `value` through their respective linear layers
#     * Then, we perform the multi-head splitting of the linearly projected outputs
#         * each projection's hidden dimension has to be reshaped to fit the `num_heads` and `head_dim` structure (in that order)
#         * Hint: Both `batch_size` and `seq_len` shouldn't be changed
#     * Afterwards, we perform the matrix multiplication step of queries with their transposed keys, visualized by the $QK^T$ in the above formula
#     * Hint: The output shape after this step should be `(batch_size, num_heads, num_query_seq, num_key_seq)`
#     * Call this output `key_out`
#     * After this step, add in the optional step to mask the `key_out` tensor. We provided this code snipped, just include it at this step in the forward pass
#     * Following this, we perform the softmax step on the result of the division from `key_out` with the square root of our `head_dim`
#         * Make sure to apply softmax to the correct dimension
#     * Now we need just need to matrix multiply this result with the values (which were passed through their respective linear layer earlier)
#     * The output shape of this operation is `(batch_size, seq_len, num_heads, head_dim)`
#     * Reshape it to fit the input shape of our output linear layer
#     * Pass it through the ouput linear layer

# %%
from torch import nn


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


# %% [markdown]
# ## Task 1.2: Transformer Blocks
# We will now create Transformer Blocks out of our `MultiHeadAttention` module, combined with Feedforward-Networks.
#

#
# * To create the blocks, our module takes as input in it's `__init__` method:
#     * the embedding dimension `emb_dim`, the number of heads `num_heads`, a dropout rate `dropout`, and the dimension of the hidden layer in the feedforward network, often called `forward_dim`
#     * in the `__init__` method, we further need two `nn.Layernorm` objects with an epsilon parameter `eps=1e-6`
#     * then, still in the `__init__` method, we set up the feedforward network
#     * we build it by creating an `nn.Sequential` module and filling it with:
#         * a linear layer projecting the input onto the `forward_dim`
#         * running it through `nn.ReLU`
#         * and projecting the `forward_dim` back to the embedding dimension with another linear layer
# * the `forward()` method takes `query`, `key`, `value` and the `mask`
#     * first, we run `query`, `key`, `value`, and the `mask` through multi-head attention
#     * secondly, we build a skip-connection by adding the `query` back to the output of multi-head attention
#         * dropout is applied to the sum, followed by our first layer norm
#     * third, the output is put through our FFN
#     * fourth, we build another skip-connection by adding the input of the FFN onto the output of the FFN
#     * apply dropout to the result of the skip-connection, apply normalization on the dropped-out result, and return it


# %%
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, forward_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.ln2 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.dropout = dropout
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, forward_dim), nn.ReLU(), nn.Linear(forward_dim, emb_dim)
        )
        self.attn = MultiHeadAttention(emb_dim, num_heads)

    def forward(self, query, key, value, mask):
        attn_out = self.attn(query, key, value, mask)
        skip1 = query + attn_out
        skip1 = F.dropout(skip1, p=self.dropout)
        skip1 = self.ln1(skip1)
        ffn_out = self.ffn(skip1)
        skip2 = skip1 + ffn_out
        skip2 = F.dropout(skip2, p=self.dropout)
        out = self.ln2(skip2)
        return out


# %% [markdown]
# This already convenes the encoder side of the transformer. We now just need to incorporate it into an appropriate format so that it can take input sequences, move them to the GPU, etc. To achieve this, we create another module called `Encoder`.
#
# ## Task 1.3 Encoder
# * The `Encoder` takes as input in its `__init__` method:
#     * the (source) vocabulary size `vocab_size`, embedding dimension `emb_dim`, number of layers `num_layers`, number of heads `num_heads`, feedforward-dimension `forward_dim`, the dropout rate `dropout`, and the maximum sequence length `max_len`
#     * Note that the preprocessing, in this case the truncation of sequences to the maximum allowed length, is handled in the data loading process that we performed in the first exercise while loading the sequences. Here, we define the model architecture that (usually) dictates the necessary preprocessing steps.
#     * We then define
#         * the token level embeddings with dimensions `vocab_size x emb_dim`
#         * positional encodings with the sinusoidal approach (function is given below)
#             * You need to create an additional `nn.Embedding` layer and load in the sinusoid table with the `.from_pretrained` method
#             * Freeze these embeddings
#         * a dropout layer
#         * and, lastly, instantiate `num_layers` many `TransformerBlock` modules inside an `nn.ModuleList`
# * In the `forward()` method, we take in the batched sequence inputs, as well as a mask
#     * Then, we create the input to the positional encodings by defining a matrix which represents the index of each token in the sequence
#         * Move the positions to the device on which the batched sequences are located
#         * Make sure to shift each index by `+1` (and the `max_len` in the creation of the sinusoidal table, too)
#         * This is done because index `0` is usually reserved for special tokens like `[PAD]`, which don't need a positional encoding.
#     * We then run our input through the embeddings, the above create positions are run through the positional encodings, and both results are summed up
#     * Apply dropout to the summed up result
#     * This will be our `query`, `key`, and `value` input that runs `num_layers` times through our encoder module list
#     * Return the last output


# %%
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


# %%
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


# %% [markdown]
# ## Task 1.4: Decoder Blocks
# Now to the decoder part!
#
#
# A `DecoderBlock` looks very similar to our previous `TransformerBlock`, but slightly extends the functionality because at its second stage, it receives inputs from both the encoder and its first stage (look closely at the input arrows in the picture!)
#
# * To build one, the module's `__init__` method takes as input:
#     * the embedding dimension `emb_dim`, number of heads `num_heads`, a feedforward dimension `forward_dim`, and a `dropout` rate
#     * It then initializes:
#         * an `nn.LayerNorm` with `eps=1e-6`, the `MultiHeadAttention` module, a `TransformerBlock`, and the dropout rate
# * The decoder block's `forward()` method takes:
#     * the batched sequence input, `value`, `key`, a source mask, and a target mask
#     * First, we compute *self-attention* representations of the input (i.e., the input serves as `query`, `key`, and `value`), and takes the *target mask* for the mask parameter
#         * This is the input that is symbolized by the arrow coming from the bottom of the image
#     * Secondly, we use a skip-connection by summing up the above self-attention result with the original input (again, apply dropout here and normalize the result)
#         * This output is our new `query`
#     * We now run this above created `query` as the query-input through a `TransformerBlock`, where the `value` and `key` arguments for the `TransformerBlock` come from the `Encoder` output
#         * This is called *cross-attention*
#         * Include the source mask as the `mask` argument in the `TransformerBlock`
#         * return the output of the `TransformerBlock`


# %%
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


# %% [markdown]
# As we could see from the large overview of the transformer architecture, this is already most of what is happening on the decoder side. Similar to our `Encoder`, we now must enable the `DecoderBlock` to take external input and embed its own sequences. We will do this in the `Decoder` module below.
#
# ## Task 1.5: Decoder
# The `Decoder`'s `__init__` method
# * takes as input:
#     * the (target) vocabulary size `vocab_size`, embedding dimension `emb_dim`, number of layers `num_layers`, number of heads `num_heads`, the hidden dimensionality `forward_dim` of the feedforward module, as well as the maximum sequence length `max_len`
#
#     * We then initialize:
#         * token embeddings, a dropout layer, and `num_layers` many `DecoderBlocks` inside another `nn.ModuleList`
#         * We also need positional encodings, but here we don't use sinusoidal embeddings, but instead something called *relative positional encodings*, which capture the relative position between the decoder input tokens and the output tokens at each decoding step
#             * They are trainable, and are implemented by another `nn.Embedding` layer, but with dimensions `max_len x emb_dim`
#         * lastly, we need a linear output layer which maps the embedding dimension back to the vocabulary size
#
# * The modules `forward()` pass then takes as input the batched sequence input, the encoder output, and a source and target mask
#     * The decoder then:
#         * processes the sequences through our normal embeddings
#         * creates inputs to the relative positional encodings by again creating a matrix of position indices from each token in the sequence (no `+1` shifting this time because we train each position relative to the current encoded sequence position output)
#         * The inputs again need to be moved to the batched sequence input's device
#         * runs these positions through the relative positional encodings, and sums them up with the token embeddings
#             * apply dropout on the sum
#         * the sum will be the input to the `num_layers` decoder block
#         * loop through all layers by passing the previous output as input through the next layer
#         * the last output will put through the linear output layer and returned


# %%
class Decoder(nn.Module):
    def __init__(
        self, vocab_size, emb_dim, num_layers, num_heads, forward_dim, dropout, max_len
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [DecoderBlock(emb_dim, num_heads, forward_dim, dropout)] * num_layers
        )

        self.fc1 = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        embeddings = self.token_emb(x)
        positions = torch.arange(x.shape[1]).expand(x.shape[0], x.shape[1]).to(x.device)
        pos_embeddings = self.pos_emb(positions)
        x = self.dropout(embeddings + pos_embeddings)

        print("Passing through decoder layers")
        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, src_mask, tgt_mask)
        out = self.fc1(x)
        return out


# %% [markdown]
# Now, we just need to put everything together into one `Transformer`.
#
# ## Task 1.6: Transformer
# * Gather all necessary arguments to initialize one `Encoder` and one `Decoder` in the `__init__` method
# * Additionally, we also need to include a source and a target padding index
# * For simplicity, we provide both mask creation functions
#
# During the `forward()` pass, we:
# * take in our batched source and target sequences
# * call both `create_mask` functions on the respective source and target sequence
# * encode the sequence using the initialized encoder and the source mask
# * input the original target sequences as input into the decoder, together with the encoder output and both masks
# * return the output of the decoder
#
# That's it - you made it!
#
# If you want to test the general functionality of your Transformer, we provide a test for you below. If the asserted shape is returned, you are on the right track.


# %%
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
        src_mask = self.create_src_mask(src)  # .permute(0, 3, 2, 1)
        tgt_mask = self.create_tgt_mask(tgt)  # .permute(0, 3, 2, 1)

        encoder_out = self.encoder(src, src_mask)
        decoder_out = self.decoder(tgt, encoder_out, src_mask, tgt_mask)
        return decoder_out


# %%
from transformer import Transformer
import torch

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

# %%
from dataset.scan_dataset import ScanDataset, DatasetType
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

train_dataset = ScanDataset(
    DatasetType.E1_TRAIN, in_seq_len=40, out_seq_len=40, device=device
)
test_dataset = ScanDataset(
    DatasetType.E1_TEST, in_seq_len=40, out_seq_len=40, device=device
)

# %% [markdown]
# # Experiment 1

# %%
from transformer import Transformer
from dataset.scan_dataset import ScanDataset, DatasetType
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
# device = "cpu"
max_len = 256
train_dataset = ScanDataset(
    DatasetType.E1_TRAIN,
    in_seq_len=max_len,
    out_seq_len=max_len + 5,
    device=device,
)
test_dataset = ScanDataset(
    DatasetType.E1_TEST,
    in_seq_len=max_len,
    out_seq_len=max_len + 5,
    device=device,
)

model = Transformer(
    src_vocab_size=len(train_dataset.vocab),
    tgt_vocab_size=len(train_dataset.vocab),
    src_pad_idx=train_dataset.vocab.pad_idx,
    tgt_pad_idx=train_dataset.vocab.pad_idx,
    dropout=0.05,
    emb_dim=128,
    num_layers=1,
    num_heads=8,
    forward_dim=512,
    max_len=max_len + 10,
)

# %%
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn import utils
import torch

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)
grad_clip = 1.0
criterion = CrossEntropyLoss()
optimizer = Adam(
    model.parameters(),
    lr=7e-4,
    weight_decay=0.0004,
)

from tqdm import tqdm

model.to(device)

for epoch in range(3):
    losses = []
    for step, batch in enumerate(tqdm(train_loader)):
        src, tgt = batch

        optimizer.zero_grad()
        out = model(src, tgt)
        loss = criterion(out.permute(0, 2, 1), tgt)
        loss.backward()
        utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        losses.append(loss.item())
        if step % 100 == 0:
            print(f"Epoch {epoch} Loss: {sum(losses) / len(losses)}")

            # if step % 50 == 0:
            with torch.no_grad():
                for batch in test_loader:
                    src, tgt = batch
                    out = model(src, tgt)
                    loss = criterion(out.permute(0, 2, 1), tgt)

                print(f"Test Loss: {loss.item()}")

# %%
test_dataset.inputs[0]

# %%
# command = "jump opposite right after walk around right thrice,"
command, target = test_dataset[0]
print(test_dataset.vocab.indxs_to_tokens(command.tolist()))
with torch.no_grad():
    out = model(command.unsqueeze(0), target.unsqueeze(0))
    indexes = out.argmax(2).squeeze(0).tolist()

out_string = test_dataset.vocab.indxs_to_tokens(indexes)
print("Predicted", out_string)
print("Target", test_dataset.vocab.indxs_to_tokens(target.tolist()))
