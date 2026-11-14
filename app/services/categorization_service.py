# sorts uploaded docs into a category using tf-idf + logistic regression,
# same idea as the spam classifier exercise just applied for real here

from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from app.services.training_data import generate

# small training set, would add more real examples over time
TRAINING_DATA = [
    ("Experienced software engineer with 5 years in Python and backend development.", "resume"),
    ("Objective: seeking a data analyst role. Skills include SQL and Excel.", "resume"),
    ("Education: BS Computer Science. Work experience includes internships at two startups.", "resume"),
    ("Meeting notes: discussed Q3 roadmap, action items assigned to team leads.", "meeting_notes"),
    ("Attendees: John, Sarah, Mike. Topics covered: budget review, next steps.", "meeting_notes"),
    ("Agenda for today's standup: blockers, progress updates, upcoming deadlines.", "meeting_notes"),
    ("Quarterly report summarizing revenue growth and key performance indicators.", "report"),
    ("This report outlines findings from the user research study conducted in June.", "report"),
    ("Annual report detailing company performance across all business units.", "report"),
    ("This agreement is entered into between the parties for the purpose of service delivery.", "contract"),
    ("Terms and conditions: payment due within 30 days of invoice date.", "contract"),
    ("Non-disclosure agreement covering confidential information shared between parties.", "contract"),
]


class CategorizationService:
    def __init__(self, backend: str | None = None):
        # backend is "logreg" (default), "mlp" (sklearn neural net) or "lstm" (pytorch, week 4)
        self.backend = backend or os.environ.get("CATEGORIZER", "logreg")
        data = TRAINING_DATA + generate(40, seed=7)
        texts = [t for t, _ in data]
        labels = [l for _, l in data]

        self.neural = None
        if self.backend == "lstm":
            try:
                from app.services.neural_categorizer import NeuralCategorizer
                self.neural = NeuralCategorizer().fit(data)
                return
            except ImportError:
                # torch not installed, use the sklearn model instead
                self.backend = "logreg"

        self.vectorizer = TfidfVectorizer(stop_words="english")
        X = self.vectorizer.fit_transform(texts)

        if self.backend == "mlp":
            self.model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=0)
        else:
            self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, labels)

    def categorize(self, text: str) -> str:
        if not text or not text.strip():
            return "other"

        if self.neural is not None:
            probs = self.neural.predict_proba(text)
            best = max(probs, key=probs.get)
            return best if probs[best] >= 0.35 else "other"

        X = self.vectorizer.transform([text])
        prediction = self.model.predict(X)[0]

        # don't force a guess if the model isn't actually confident
        probabilities = self.model.predict_proba(X)[0]
        if max(probabilities) < 0.35:
            return "other"

        return prediction


categorization_service = CategorizationService()
