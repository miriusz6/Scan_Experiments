import pickle


class Vocab:
    def __init__(self):
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self._token2index = {self.pad_token: 0, self.sos_token: 1, self.eos_token: 2}
        self._index2token = {0: self.pad_token, 1: self.sos_token, 2: self.eos_token}
        self.n_tokens = 3  # Count SOS and EOS

    def __len__(self):
        return len(self._token2index)

    def addTokens(self, tokens):
        [self._addToken(token) for token in tokens]

    def _addToken(self, token):
        if token in self._token2index:
            return
        self._token2index[token] = self.n_tokens
        self._index2token[self.n_tokens] = token
        self.n_tokens += 1

    @property
    def pad_idx(self) -> int:
        return self._token2index[self.pad_token]

    @property
    def sos_idx(self) -> int:
        return self._token2index[self.sos_token]

    @property
    def eos_idx(self) -> int:
        return self._token2index[self.eos_token]

    def tokens_to_indxs(self, tokens):
        return [self._token2index[token] for token in tokens]

    def indxs_to_tokens(self, indxs):
        ret = []
        for indx in indxs:
            if self._index2token[indx] == "":
                continue
            ret.append(self._index2token[indx])
        return ret

    def save_to_file(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def from_file(path):
        with open(path, "rb") as f:
            return pickle.load(f)
