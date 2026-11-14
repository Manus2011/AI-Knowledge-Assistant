# week 6 embedding experiments: how do chunk size and embedding dimension change retrieval?
# uses the same 5 docs / 15 questions as eval_retrieval.py. temp db, nothing real is touched.
import os, sys, tempfile, importlib
tmp = tempfile.mkdtemp()
os.environ["DATA_DIR"] = tmp
os.environ["DB_PATH"] = os.path.join(tmp, "exp.db")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from app.services import embedding_service as emb
from app.services.chunking_service import chunk_text

sys.path.insert(0, os.path.dirname(__file__))
from retrieval_data import DOCS, QUERIES


def run(size, overlap, dim):
    emb._vec = HashingVectorizer(n_features=dim, analyzer=emb._analyze, alternate_sign=False, norm="l2")
    chunks, owners = [], []
    for name, text in DOCS.items():
        for c in chunk_text(text, size=size, overlap=overlap):
            chunks.append(c); owners.append(name)
    M = emb._vec.transform(chunks).toarray()
    top1 = 0
    for q, want in QUERIES:
        scores = M @ emb._vec.transform([q]).toarray()[0]
        top1 += owners[int(np.argmax(scores))] == want
    return top1


print("chunk size / overlap / dim -> top-1 (out of 15)")
for size, overlap in [(100, 20), (200, 40), (500, 80)]:
    for dim in [64, 512, 4096]:
        print(f"{size:4d} / {overlap:3d} / {dim:4d} -> {run(size, overlap, dim)}")
