"""
Week 2 capstone enhancement: document categorization.

Trains a small text classifier on startup that sorts uploaded documents
into categories (resume, meeting_notes, report, contract, other). Same
core idea as the spam classifier exercise, just applied to a real feature
in the app now instead of a standalone script.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# small labeled training set, enough to get reasonable behavior for
# common document types. would grow this with real examples over time
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
    def __init__(self):
        texts = [t for t, _ in TRAINING_DATA]
        labels = [l for _, l in TRAINING_DATA]

        self.vectorizer = TfidfVectorizer(stop_words="english")
        X = self.vectorizer.fit_transform(texts)

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, labels)

    def categorize(self, text: str) -> str:
        if not text or not text.strip():
            return "other"

        X = self.vectorizer.transform([text])
        prediction = self.model.predict(X)[0]

        # if the model isn't confident in any class, fall back to "other"
        # rather than force a guess on something that doesn't look like
        # any of the trained categories
        probabilities = self.model.predict_proba(X)[0]
        if max(probabilities) < 0.35:
            return "other"

        return prediction


categorization_service = CategorizationService()
