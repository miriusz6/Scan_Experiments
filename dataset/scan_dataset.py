import os.path
import os
from torch.utils.data import Dataset
from torch.nn.functional import pad
import torch
from dataset.vocab import Vocab

class ScanDataset(Dataset):
    def __init__(
        self,
        data = None,
        dataset_path: str = None,
        vocab: Vocab = None,
        in_seq_len: int = 75,
        out_seq_len: int = 80,
        device: str = "cuda",
    ):
        self.vocab = vocab
        if dataset_path is not None:
            self._verify_path(dataset_path)
            self.path = dataset_path
            self.in_seq_len = in_seq_len
            self.out_seq_len = out_seq_len
            # create and update vocab if not provided
            mk_vocab = vocab is None
            if mk_vocab:
                self.vocab = Vocab()
            inputs, decoder_inputs, targets = self._load_data(self.path, mk_vocab)
            self.inputs = torch.stack(inputs).to(device)
            self.decoder_input = torch.stack(decoder_inputs).to(device)
            self.targets = torch.stack(targets).to(device)
        elif data is not None:
            self.inputs, self.decoder_input, self.targets = data
        else:
            raise ValueError("Provide either data or dataset_path")
        self.length = self.inputs.size(0)
        self.command_length_span, self.action_sequence_span  = self._length_spans()


    def _load_data(self, path, mk_vocab):
        inputs, decoder_inputs, targets = [], [], []
        # one line at a time
        with open(path, "r") as f:
            eof = False
            while not eof:
                l = f.readline()
                eof = l == ""
                if eof:
                    break
                # to list of tokens(words)
                inp, targ = self._tokenize(l)
                # add tokens to vocab if no vocab provided
                if mk_vocab:
                    self.vocab.addTokens(inp)
                    self.vocab.addTokens(targ)

                # trim to max sequence length
                inp, targ = self._trim(inp, self.in_seq_len), self._trim(
                    targ, self.out_seq_len
                )
                # update the max/min lengths of the input and output sequences
                #self._update_length_spans(inp, targ)

                # add SOS and EOS tokens
                inp, dec_in, targ = (
                    self._add_eos(inp),
                    self._add_sos(targ),
                    self._add_eos(targ),
                )
                # convert tokens to corresponding indexes
                inp, dec_in, targ = (
                    self._tokens2indx(inp),
                    self._tokens2indx(dec_in),
                    self._tokens2indx(targ),
                )
                # to tensor
                inp, dec_in, targ = (
                    torch.tensor(inp, dtype=torch.long),
                    torch.tensor(dec_in, dtype=torch.long),
                    torch.tensor(targ, dtype=torch.long),
                )
                # pad to max sequence length
                inp = pad(inp, (0, self.in_seq_len - len(inp)))
                dec_in = pad(dec_in, (0, self.out_seq_len - len(dec_in)))
                targ = pad(targ, (0, self.out_seq_len - len(targ)))

                inputs.append(inp)
                decoder_inputs.append(dec_in)
                targets.append(targ)
        return inputs, decoder_inputs, targets

    def _length_spans(self):
        command_lens = torch.where(self.inputs == self.vocab.pad_idx, 0, 1).sum(1)
        action_lens = torch.where(self.targets == self.vocab.pad_idx, 0, 1).sum(1)
        command_span = (command_lens.min().item(), command_lens.max().item())
        action_span = (action_lens.min().item(), action_lens.max().item())
        return command_span, action_span

    def _verify_path(self, path):
        if not os.path.isfile(path):
            # check if file exists
            raise FileNotFoundError("File not found: {}".format(path))
        elif not os.access(path, os.R_OK):
            # check if permission to read
            raise PermissionError("File not readable: {}".format(path))
            # check if txt
        elif not path.endswith(".txt"):
            raise ValueError("File is not a txt file: {}".format(path))

    def _add_sos(self, tokens):
        return ["<SOS>"] + tokens

    def _add_eos(self, tokens):
        return tokens + ["<EOS>"]

    def _add_sos_eos(self, tokens):
        return ["<SOS>"] + tokens + ["<EOS>"]

    def _trim(self, tokens, max_len):
        return tokens[:max_len] if len(tokens) > max_len else tokens

    def _tokens2indx(self, tokens):
        return [self.vocab._token2index[token] for token in tokens]

    def _tokenize(self, line):
        # "IN: SEQ OUT: SEQ" -> ["SEQ"], ["SEQ"]
        inp, targ = line.split("OUT:")
        _, inp = inp.split("IN:")
        inp, targ = inp.strip(), targ.strip()
        inp, targ = inp.split(" "), targ.split(" ")
        return inp, targ

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if isinstance(idx, int) and idx >= self.length:
            raise IndexError("Index out of range")
        return self.inputs[idx], self.decoder_input[idx], self.targets[idx]

    def pull_nth_fold_out(self, n_folds, fold_num):
        if fold_num >= n_folds:
            raise ValueError("fold_num must be less than n_folds")
        fold_size = self.length // n_folds
        start = fold_size * fold_num
        end = start + fold_size
        v = self.vocab
        chosen_fold = ScanDataset(self[start:end], vocab=v)
        
        if start == 0:
            # return the tail, chosen fold
            tail = ScanDataset(self[end:], vocab=v)
            return tail, chosen_fold
        elif end == self.length:
            # return the head, chosen fold
            head = ScanDataset(self[:start], vocab=v)
            return head, chosen_fold
        else:
            # return merged tail and head, chosen fold
            head_in, head_dec, head_targ = self[:start]
            tail_in, tail_dec, tail_targ = self[end:]
            inputs = torch.cat((head_in, tail_in))
            decoder_input = torch.cat((head_dec, tail_dec))
            targets = torch.cat((head_targ, tail_targ))
            headNtail = ScanDataset((inputs, decoder_input, targets), vocab=v)
            return headNtail, chosen_fold
        

from transformers import PreTrainedTokenizer

class ScanDatasetHF(Dataset):
    def __init__(
        self,
        dataset_path: str,
        tokenizer:PreTrainedTokenizer,
        in_seq_len: int = 120,#75,
        out_seq_len: int = 120,
        device: str = 'cuda',
    ):
        
        self.tokenizer = tokenizer
        self._verify_path(dataset_path)
        self.path = dataset_path
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len

        # create and update vocab if not provided
        inputs, dec_in, targets = self._load_data(self.path)
        inputs = self.tokenizer(inputs, 
                                padding="max_length", 
                                truncation=True, 
                                max_length=in_seq_len,
                                padding_side='right',
                                return_tensors="pt")
        targets = self.tokenizer(targets,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=out_seq_len,
                                    padding_side='right',
                                    return_tensors="pt")    
        decoder_input = self.tokenizer(dec_in,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=out_seq_len,
                                    padding_side='right',
                                    return_tensors="pt")

        


        self.inputs = inputs['input_ids']
        self.targets = targets['input_ids']
        self.decoder_input = decoder_input['input_ids']
        self.inputs_msks = inputs['attention_mask']
        self.targets_msks = targets['attention_mask']
        self.decoder_input_msks = decoder_input['attention_mask']

        self.output_vocab = torch.sort(torch.unique(self.inputs.flatten())).values
        self.length = self.inputs.size(0)
        

    
    def _verify_path(self, path):
        if not os.path.isfile(path):
            # check if file exists
            raise FileNotFoundError("File not found: {}".format(path))
        elif not os.access(path, os.R_OK):
            # check if permission to read
            raise PermissionError("File not readable: {}".format(path))
            # check if txt
        elif not path.endswith(".txt"):
            raise ValueError("File is not a txt file: {}".format(path))

    def _load_data(self, path):
        inputs, dec_input, targets = [], [], []
        # one line at a time
        with open(path, "r") as f:
            eof = False
            while not eof:
                l = f.readline()
                eof = l == ""
                if eof:
                    break
                # clean
                inp, targ = self._clean_raw(l)
                dec_in = targ
                targ = self._add_eos(targ)
                inp = self._add_eos(inp)
                inputs.append(inp)
                targets.append(targ)
                dec_input.append(dec_in)
        return inputs, dec_input, targets

    def _clean_raw(self, line):
        # "IN: SEQ OUT: SEQ" -> ["SEQ"], ["SEQ"]
        inp, targ = line.split("OUT:")
        _, inp = inp.split("IN:")
        inp, targ = inp.strip(), targ.strip()
        return inp, targ
    
    def _add_start(self, tokens, token):
        return [token] + tokens

    def _add_sos(self, tokens):
        return self.tokenizer.special_tokens_map['bos_token'] + tokens

    def _add_eos(self, tokens):
        return tokens + self.tokenizer.special_tokens_map['eos_token']
        

    def _add_sos_eos(self, tokens):
        return self._add_eos(self._add_sos(tokens))


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if isinstance(idx, int) and idx >= self.length:
            raise IndexError("Index out of range")
        return self.inputs[idx], self.inputs_msks[idx], self.decoder_input[idx], self.decoder_input_msks[idx], self.targets[idx], self.targets_msks[idx]
