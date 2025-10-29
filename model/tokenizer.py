class SimpleTokenizer:
    def __init__(self, texts):
        # Build vocabulary from dataset text + special tokens
        special_tokens = ["<PAD>"]
        chars = sorted(list(set("".join(texts))))
        vocab = special_tokens + chars
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.pad_token = "<PAD>"
        self.pad_id = self.stoi[self.pad_token]

    def encode(self, text):
        return [self.stoi.get(ch, self.pad_id) for ch in text]

    def decode(self, tokens):
        # Drop PAD tokens during decoding
        return "".join([self.itos.get(t, "") for t in tokens if t != self.pad_id])

    @property
    def vocab_size(self):
        return len(self.stoi)
