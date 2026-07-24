"""
Week 3 capstone enhancement: keyword extraction.

Runs the same TF-IDF idea from the keyword_extractor exercise, but wired
into the real app now. Every uploaded document gets a set of keywords
attached, which is a small step toward "document understanding", it's
the first hint of the semantic search work coming in Week 6.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

# generic reference texts so TF-IDF has something to compare against when
# there aren't many real documents uploaded yet. once there are enough
# real documents in the system, those should be used as the comparison
# corpus instead, since that gives more meaningful scores
FALLBACK_CORPUS = [
    "Machine learning models learn patterns from data to make predictions.",
    "Cloud computing lets companies rent servers instead of owning hardware.",
    "Project management involves planning, tracking, and delivering work on time.",
    "Financial reports summarize revenue, expenses, and overall performance.",
]


def extract_keywords(text: str, comparison_corpus: list[str] | None = None, top_n: int = 8) -> list[str]:
    if not text or not text.strip():
        return []

    corpus = (comparison_corpus or FALLBACK_CORPUS) + [text]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    scores = tfidf_matrix[-1].toarray()[0]
    feature_names = vectorizer.get_feature_names_out()

    word_scores = [(word, score) for word, score in zip(feature_names, scores) if score > 0]
    word_scores.sort(key=lambda x: x[1], reverse=True)

    return [word for word, _ in word_scores[:top_n]]
