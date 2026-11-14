# small cnn on the 8x8 digits set, compared with the numpy net from digit_recognizer.py
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                      # 8x8 -> 4x4
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                      # 4x4 -> 2x2
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 2 * 2, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.head(self.features(x))


def main():
    d = load_digits()
    X = torch.tensor(d.data / 16.0, dtype=torch.float32).reshape(-1, 1, 8, 8)
    y = torch.tensor(d.target)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = CNN()
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, 31):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(Xte).argmax(1) == yte).float().mean().item()
            print(f"epoch {epoch:2d}  test acc {acc:.3f}")

    print(f"\ncnn test accuracy: {acc:.3f}  (numpy net got 0.969)")


if __name__ == "__main__":
    main()
