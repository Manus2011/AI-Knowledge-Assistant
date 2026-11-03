# small neural net written with numpy only, to see how backprop actually works.
# trained on sklearn's 8x8 digits set, then compared to sklearn's MLPClassifier.
# (torch wasn't available in my setup so no cnn here)

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

rng = np.random.default_rng(0)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TinyNet:
    def __init__(self, n_in, n_hidden, n_out):
        self.W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, n_out))
        self.b2 = np.zeros(n_out)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # relu
        return softmax(self.a1 @ self.W2 + self.b2)

    def step(self, X, y_onehot, lr):
        n = len(X)
        probs = self.forward(X)
        d2 = (probs - y_onehot) / n
        dW2 = self.a1.T @ d2
        db2 = d2.sum(axis=0)
        d1 = (d2 @ self.W2.T) * (self.z1 > 0)
        dW1 = X.T @ d1
        db1 = d1.sum(axis=0)
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        return -np.log(probs[np.arange(n), y_onehot.argmax(axis=1)] + 1e-9).mean()

    def predict(self, X):
        return self.forward(X).argmax(axis=1)


def main():
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    Y = np.eye(10)[y_train]

    net = TinyNet(64, 48, 10)
    for epoch in range(1, 301):
        loss = net.step(X_train, Y, lr=0.5)
        if epoch % 50 == 0:
            acc = (net.predict(X_test) == y_test).mean()
            print(f"epoch {epoch:3d}  loss {loss:.4f}  test acc {acc:.3f}")

    scratch_acc = (net.predict(X_test) == y_test).mean()

    sk = MLPClassifier(hidden_layer_sizes=(48,), max_iter=500, random_state=0)
    sk.fit(X_train, y_train)
    sk_acc = sk.score(X_test, y_test)

    print(f"\nfrom scratch numpy net: {scratch_acc:.3f}")
    print(f"sklearn MLPClassifier:  {sk_acc:.3f}")


if __name__ == "__main__":
    main()
