# lstm vs logistic regression vs mlp on the document categories, same held-out-phrase test
# as neural_text_classifier.py (synthetic data, so just a sanity check)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from app.services.neural_categorizer import NeuralCategorizer
from app.services.training_data import generate

train = generate(60, seed=1, frag_slice=slice(0, 5))
test = generate(25, seed=999, frag_slice=slice(5, None))
texts_test, y_test = [t for t, _ in test], [l for _, l in test]

lstm = NeuralCategorizer().fit(train)
lstm_acc = sum(p == y for p, y in zip(lstm.predict(texts_test), y_test)) / len(y_test)

vec = TfidfVectorizer(stop_words="english")
Xtr = vec.fit_transform([t for t, _ in train])
Xte = vec.transform(texts_test)
ytr = [l for _, l in train]
lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr).score(Xte, y_test)
mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=0).fit(Xtr, ytr).score(Xte, y_test)

print(f"logreg {lr:.3f}   mlp {mlp:.3f}   lstm {lstm_acc:.3f}")
