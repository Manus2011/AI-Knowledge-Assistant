"""
Week 3 - Day 1: Resume classifier

Classifies a resume snippet into a job field (engineering, marketing,
finance). Builds on the same TF-IDF + classifier pattern from Week 2,
just applied to a proper NLP preprocessing pipeline this time: cleaning
text before vectorizing it instead of feeding raw text straight in.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

TRAINING_DATA = [
    ("Built REST APIs using Python and FastAPI, worked with SQL databases and Git.", "engineering"),
    ("Developed machine learning models for fraud detection using scikit-learn.", "engineering"),
    ("Led frontend development using React, collaborated with backend engineers on API design.", "engineering"),
    ("Automated testing pipelines and improved CI/CD workflows for the platform team.", "engineering"),
    ("Managed social media campaigns and increased engagement by analyzing audience data.", "marketing"),
    ("Created content strategy for product launches across email and paid ad channels.", "marketing"),
    ("Ran A/B tests on landing pages to improve conversion rate for marketing campaigns.", "marketing"),
    ("Coordinated brand partnerships and managed influencer outreach programs.", "marketing"),
    ("Prepared financial models and forecasts for quarterly investor presentations.", "finance"),
    ("Performed variance analysis on budget vs actuals for department spending.", "finance"),
    ("Analyzed portfolio performance and prepared risk assessment reports.", "finance"),
    ("Reconciled monthly accounts and supported the annual audit process.", "finance"),
    ("Wrote unit tests and debugged production issues in a Django backend.", "engineering"),
    ("Designed database schemas and optimized slow SQL queries for the app.", "engineering"),
    ("Built email marketing funnels and tracked open rates and conversions.", "marketing"),
    ("Managed the company's SEO strategy and grew organic search traffic.", "marketing"),
    ("Built valuation models for M&A deals and presented findings to leadership.", "finance"),
    ("Tracked cash flow and prepared monthly financial statements.", "finance"),
]


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace.

    This is the "text cleaning" step from this week's topics, doing it
    before vectorizing keeps the model from treating 'API' and 'api.' as
    two different tokens.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run():
    texts = [clean_text(t) for t, _ in TRAINING_DATA]
    labels = [l for _, l in TRAINING_DATA]

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.3, random_state=1, stratify=labels
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    print(f"Test accuracy: {model.score(X_test, y_test):.2f}\n")

    new_resumes = [
        "Designed dashboards and wrote Python scripts to automate reporting pipelines.",
        "Planned digital ad spend and tracked campaign ROI across channels.",
        "Built pricing models and evaluated investment opportunities for the fund.",
    ]

    for resume in new_resumes:
        cleaned = clean_text(resume)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        print(f"'{resume[:50]}...' -> {prediction}")


if __name__ == "__main__":
    run()
