import os
import re

from transformers import PreTrainedTokenizer


class GenoJEPATokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        self.pad_token = "N"

        self.base_chars = ["A", "T", "C", "G"]
        self.special_tokens = [self.pad_token]

        self.vocab = {token: i for i, token in enumerate(self.base_chars + self.special_tokens)}
        self.ids_to_tokens = {i: token for token, i in self.vocab.items()}

        self.dna_pattern = re.compile(r"[ATCG]")
        self.pad_token_id = self._convert_token_to_id(self.pad_token)

        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab[self.pad_token])

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.pad_token)

    def _tokenize(self, seq, **kwargs):
        tokens = []
        pos = 0
        while pos < len(seq):
            dna_match = self.dna_pattern.match(seq, pos)
            if dna_match:
                dna_seq = dna_match.group()
                tokens.append(dna_seq)
                pos = dna_match.end()
            else:
                tokens.append(self.pad_token)
                pos += 1

        return tokens

    def save_vocabulary(self, save_directory, filename_prefix=None):
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.txt")
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                writer.write(token + "\n")

        return (vocab_file,)
