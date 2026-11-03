import re
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, ENGLISH_STOP_WORDS

DIM = 4096  # 1024 gave hash collisions (junk queries matched real chunks)

# note: this is a hashed bag of words/bigrams, so it matches on shared wording.
# it is NOT a semantic model (no synonyms), swapping in sentence-transformers or an
# embeddings api later only means changing embed()
SUFFIXES = ("ations", "ation", "ing", "ed", "es", "s", "ly")


def _stem(w: str) -> str:
    # crude suffix stripping so rotated/rotation/rotates land on the same token
    for suf in SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def _analyze(text: str) -> list[str]:
    words = [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in ENGLISH_STOP_WORDS]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


_vec = HashingVectorizer(n_features=DIM, analyzer=_analyze, alternate_sign=False, norm="l2")


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    return _vec.transform(texts).toarray().astype(np.float32)


def to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
