import torch
from torch import nn
from transformers import BartModel

class SimpleBart(nn.Module):
    def __init__(self, input_length = 80, vocab_size = 50264):
        super().__init__()
        self.bart = BartModel.from_pretrained('facebook/bart-base')
        self.up = nn.Linear(768, vocab_size)
        #self.pre = nn.Linear(input_length, input_length)
        #self.post = nn.Linear(vocab_size, input_length)

    def freeze_encoder(self):
        for param in self.bart.encoder.parameters():
            param.requires_grad = False
    
    def freeze_decoder(self):
        for param in self.bart.decoder.parameters():
            param.requires_grad = False

    #def forward(self, in_ids, in_mask, tgt_ids, tgt_mask):
    def forward(self, kwargs):
        """
        Perform a forward pass through the model.
        NOT autoregressive
        Args:
            in_ids (torch.Tensor): Input IDs for the encoder.
            in_mask (torch.Tensor): Attention mask for the encoder input.
            tgt_ids (torch.Tensor): Input IDs for the decoder.
            tgt_mask (torch.Tensor): Attention mask for the decoder input.
        Returns:
            torch.Tensor: The output of the model after passing through the encoder and decoder.
        """

        # enc_out = self.bart.encoder(input_ids=in_ids, attention_mask=in_mask).last_hidden_state
        # dec_out = self.bart.decoder(input_ids=tgt_ids,
        #                             attention_mask=tgt_mask,
        #                             encoder_hidden_states = enc_out).last_hidden_state
        #x = self.up(dec_out)

        #x = self.bart(input_ids=in_ids, attention_mask=in_mask, decoder_input_ids=tgt_ids, decoder_attention_mask=tgt_mask).last_hidden_state
        #x = self.bart(input_ids=in_ids, attention_mask=in_mask).last_hidden_state

        #kwargs['input_ids'] = self.pre(kwargs['input_ids'].to(torch.float32))

        x = self.bart( **kwargs).last_hidden_state
        x = self.up(x)
        #x = self.post(x)
        return x
    
    #def generate():

    
    # to be implemented
    # autoregressive forward pass
    #def generate():
