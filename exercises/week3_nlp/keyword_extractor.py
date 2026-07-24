# day 3 - pulling keywords out of a document using tf-idf
# feeds into the categorization/keyword feature in the actual app

from sklearn.feature_extraction.text import TfidfVectorizer

DOCUMENT = """
Retrieval-Augmented Generation, or RAG, combines a retrieval system with
a language model to answer questions using external documents. Instead
of relying only on what the model learned during training, RAG retrieves
relevant chunks of text from a knowledge base and feeds them into the
model as context. This makes answers more accurate and grounded in real
source material, especially for domain-specific or up-to-date information
that the base model wouldn't otherwise know about.
"""

OTHER_DOCS_FOR_CONTEXT = [
    "Machine learning models learn patterns from data to make predictions.",
    "Deep learning uses neural networks with many layers to process data.",
    "Cloud computing lets companies rent servers instead of owning hardware.",
]


def extract_keywords(text: str, top_n: int = 8) -> list[str]:
    # tfidf needs more than one doc to compare against, so pad with a few unrelated ones
    corpus = OTHER_DOCS_FOR_CONTEXT + [text]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # last row corresponds to our target document
    scores = tfidf_matrix[-1].toarray()[0]
    feature_names = vectorizer.get_feature_names_out()

    word_scores = list(zip(feature_names, scores))
    word_scores.sort(key=lambda x: x[1], reverse=True)

    return [word for word, score in word_scores[:top_n] if score > 0]


if __name__ == "__main__":
    keywords = extract_keywords(DOCUMENT)
    print("Top keywords:")
    for kw in keywords:
        print(f"  - {kw}")
