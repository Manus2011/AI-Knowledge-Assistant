import pytest

torch = pytest.importorskip("torch")

from app.services.categorization_service import CategorizationService
from app.services.neural_categorizer import NeuralCategorizer
from app.services.training_data import generate


def test_lstm_learns_the_training_task():
    data = generate(40, seed=3)
    model = NeuralCategorizer().fit(data, epochs=15)
    held_out = generate(10, seed=11)
    acc = sum(p == y for p, (_, y) in zip(model.predict([t for t, _ in held_out]), held_out)) / len(held_out)
    assert acc > 0.9


def test_probabilities_sum_to_one():
    model = NeuralCategorizer().fit(generate(20, seed=3), epochs=5)
    probs = model.predict_proba("meeting notes with action items")
    assert abs(sum(probs.values()) - 1.0) < 1e-4


def test_lstm_backend_in_categorizer():
    svc = CategorizationService(backend="lstm")
    assert svc.backend == "lstm"
    assert svc.categorize("This agreement may be terminated by either party with 30 days written notice.") == "contract"
    assert svc.categorize("   ") == "other"


def test_cnn_forward_shape():
    import importlib.util, os
    path = os.path.join(os.path.dirname(__file__), "..", "exercises", "week4_dl", "cnn_digits.py")
    spec = importlib.util.spec_from_file_location("cnn_digits", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.CNN()(torch.zeros(4, 1, 8, 8)).shape == (4, 10)
