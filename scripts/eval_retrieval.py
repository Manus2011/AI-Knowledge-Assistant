# quick retrieval check on a small hand written set. temp db so it doesn't touch real data
import os, sys, tempfile
tmp = tempfile.mkdtemp()
os.environ["DATA_DIR"] = tmp
os.environ["DB_PATH"] = os.path.join(tmp, "eval.db")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.db import init_db, get_conn
from app.services import vector_store

sys.path.insert(0, os.path.dirname(__file__))
from retrieval_data import DOCS, QUERIES

init_db()
conn = get_conn()
for name, text in DOCS.items():
    conn.execute("INSERT INTO documents (id, filename, uploaded_at) VALUES (?, ?, 'x')", (name, name + ".txt"))
conn.commit(); conn.close()
for name, text in DOCS.items():
    vector_store.index_document(name, text)

top1 = top3 = 0
for q, want in QUERIES:
    hits = vector_store.search(q, top_k=3, min_score=0.0)
    ids = [h["doc_id"] for h in hits]
    top1 += ids[:1] == [want]
    top3 += want in ids
print(f"queries: {len(QUERIES)}  top-1: {top1}/{len(QUERIES)}  top-3: {top3}/{len(QUERIES)}")
