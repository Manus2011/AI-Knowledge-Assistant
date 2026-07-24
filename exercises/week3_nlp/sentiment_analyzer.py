"""
Week 3 - Day 2: Sentiment analyzer

Classifies short text as positive or negative. Same TF-IDF + Naive Bayes
combo as the other exercises this week, the real point here is seeing
that the exact same pipeline (clean -> vectorize -> classify) works
across pretty different NLP tasks, spam, resumes, and now sentiment.

Note: the first version of this training set was too small and caused
misclassifications, e.g. "support" only showed up once, in a negative
example, so any test sentence mentioning support got dragged negative
regardless of context. Expanded the training data to fix it. Good
reminder that small/unbalanced vocabulary is usually the real problem,
not the model itself.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

TRAINING_DATA = [
    ("This product completely changed how I work, absolutely love it.", "positive"),
    ("Great customer service, they resolved my issue in minutes.", "positive"),
    ("Really happy with the quality, exceeded my expectations.", "positive"),
    ("The team was super helpful and the onboarding was smooth.", "positive"),
    ("Fantastic experience overall, would recommend to anyone.", "positive"),
    ("This is exactly what I needed, works perfectly.", "positive"),
    ("Terrible experience, the app crashes constantly.", "negative"),
    ("Customer support never responded, very frustrating.", "negative"),
    ("Waste of money, the product broke after two days.", "negative"),
    ("Really disappointed with the quality, not worth it.", "negative"),
    ("The whole process was confusing and took way too long.", "negative"),
    ("Would not recommend, had a bad experience from the start.", "negative"),
    ("Support responded quickly and fixed the issue right away.", "positive"),
    ("The interface is clean, simple, and easy to use.", "positive"),
    ("Everything worked smoothly and the setup was quick.", "positive"),
    ("I'm really enjoying this so far, works great.", "positive"),
    ("The app kept crashing and nothing worked as advertised.", "negative"),
    ("I'm pretty annoyed, the product stopped working after a week.", "negative"),
    ("This was a complete waste of time, would not buy again.", "negative"),
    ("Setup was confusing and support never actually helped.", "negative"),
]


def run():
    texts = [t for t, _ in TRAINING_DATA]
    labels = [l for _, l in TRAINING_DATA]

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)

    model = MultinomialNB()
    model.fit(X, labels)

    test_reviews = [
        "The interface is clean and easy to use, really enjoying it so far.",
        "Nothing worked as advertised, honestly a waste of time.",
        "Support got back to me quickly and fixed everything.",
        "The product stopped working after a week, pretty annoyed.",
    ]

    for review in test_reviews:
        vec = vectorizer.transform([review])
        prediction = model.predict(vec)[0]
        confidence = max(model.predict_proba(vec)[0])
        print(f"'{review}'")
        print(f"  -> {prediction} (confidence: {confidence:.2f})\n")


if __name__ == "__main__":
    run()
