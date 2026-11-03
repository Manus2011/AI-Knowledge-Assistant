# compare logistic regression vs a small neural net (MLP) on the document categories.
# data is synthetic (templated) so treat these numbers as a sanity check only.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from app.services.training_data import generate

# train on the first 5 fragments per class, test on docs built only from the last 3
# (never seen in training) so it is a real generalization check
train = generate(60, seed=1, frag_slice=slice(0, 5))
test = generate(25, seed=999, frag_slice=slice(5, None))

vec = TfidfVectorizer(stop_words="english")
X_train = vec.fit_transform([t for t, _ in train])
y_train = [l for _, l in train]
X_test = vec.transform([t for t, _ in test])
y_test = [l for _, l in test]

lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=0).fit(X_train, y_train)

print(f"logreg  test acc: {lr.score(X_test, y_test):.3f}")
print(f"mlp     test acc: {mlp.score(X_test, y_test):.3f}")
