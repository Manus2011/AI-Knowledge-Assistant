# lstm text classifier in pytorch. torch is optional, the app falls back to the sklearn
# categorizer if it isn't installed (see categorization_service)
import re

import torch
import torch.nn as nn

PAD, UNK = 0, 1


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int, emb_dim: int = 48, hidden: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x):
        lengths = (x != PAD).sum(dim=1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.out(h[-1])


class NeuralCategorizer:
    def __init__(self, max_len: int = 80, seed: int = 0):
        self.max_len = max_len
        self.seed = seed
        self.vocab: dict[str, int] = {}
        self.labels: list[str] = []
        self.model: LSTMClassifier | None = None

    def _encode(self, text: str) -> list[int]:
        ids = [self.vocab.get(t, UNK) for t in tokenize(text)][: self.max_len]
        return ids + [PAD] * (self.max_len - len(ids))

    def fit(self, data: list[tuple[str, str]], epochs: int = 25, lr: float = 0.01):
        torch.manual_seed(self.seed)
        self.labels = sorted({l for _, l in data})
        words = sorted({w for t, _ in data for w in tokenize(t)})
        self.vocab = {w: i + 2 for i, w in enumerate(words)}
        X = torch.tensor([self._encode(t) for t, _ in data])
        y = torch.tensor([self.labels.index(l) for _, l in data])
        self.model = LSTMClassifier(len(self.vocab) + 2, len(self.labels))
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(epochs):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 32):
                idx = perm[i:i + 32]
                opt.zero_grad()
                loss = loss_fn(self.model(X[idx]), y[idx])
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, text: str) -> dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            p = torch.softmax(self.model(torch.tensor([self._encode(text)])), dim=1)[0]
        return {l: float(p[i]) for i, l in enumerate(self.labels)}

    def predict(self, texts: list[str]) -> list[str]:
        return [max((p := self.predict_proba(t)), key=p.get) for t in texts]
