# pulls keywords out of a doc using tf-idf, same as the exercise
# but wired into the actual upload flow now

from sklearn.feature_extraction.text import TfidfVectorizer

# reference texts to compare against when we don't have enough real
# docs uploaded yet for a meaningful tf-idf comparison
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
