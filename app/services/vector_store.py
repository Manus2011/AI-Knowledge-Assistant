import numpy as np
from app.core.db import get_conn
from app.services import embedding_service as emb
from app.services.chunking_service import chunk_text


def index_document(doc_id: str, text: str) -> int:
    # replaces any old chunks, so re-indexing is safe
    chunks = chunk_text(text)
    vectors = emb.embed(chunks)
    conn = get_conn()
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    conn.executemany(
        "INSERT INTO chunks (doc_id, idx, text, embedding) VALUES (?, ?, ?, ?)",
        [(doc_id, i, c, emb.to_blob(v)) for i, (c, v) in enumerate(zip(chunks, vectors))],
    )
    conn.commit()
    conn.close()
    return len(chunks)


def search(query: str, top_k: int = 4, owner: str | None = None, min_score: float = 0.08) -> list[dict]:
    conn = get_conn()
    sql = ("SELECT c.doc_id, c.idx, c.text, c.embedding, d.filename FROM chunks c "
           "JOIN documents d ON d.id = c.doc_id")
    params: tuple = ()
    if owner is not None:
        sql += " WHERE d.owner = ?"
        params = (owner,)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    if not rows:
        return []

    matrix = np.stack([emb.from_blob(r["embedding"]) for r in rows])
    q = emb.embed([query])[0]
    # both sides are l2 normalized so the dot product is cosine similarity
    scores = matrix @ q
    order = np.argsort(-scores)[:top_k]
    return [
        {
            "doc_id": rows[i]["doc_id"],
            "filename": rows[i]["filename"],
            "chunk_index": rows[i]["idx"],
            "text": rows[i]["text"],
            "score": round(float(scores[i]), 4),
        }
        for i in order if scores[i] >= min_score
    ]
