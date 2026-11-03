import numpy as np
from app.services.chunking_service import chunk_text
from app.services import embedding_service as emb
from app.services.llm_service import llm_service

DOCS = {
    "vacation_policy.txt": "Employees receive 20 days of paid vacation each year. Vacation requests must be approved by a manager two weeks in advance. Unused vacation days can be carried over up to five days.",
    "expense_policy.txt": "Travel expenses must be submitted within 30 days with receipts. Meals are reimbursed up to 50 dollars per day. Flights should be booked in economy class.",
    "security_policy.txt": "All laptops must use full disk encryption. Passwords must be rotated every 90 days and multi factor authentication is required for the VPN.",
}


def test_chunking_overlap_and_size():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 3
    assert all(len(c) <= 260 for c in chunks)
    assert chunks[0].split()[-1] in chunks[1] or chunks[1].startswith(chunks[0][-30:].split()[0])


def test_chunk_empty():
    assert chunk_text("") == []


def test_chunk_no_punctuation():
    assert len(chunk_text("word " * 500, size=100, overlap=10)) > 5


def test_embeddings_normalized():
    v = emb.embed(["hello world policy", "another sentence about vacation"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)


def _load(client):
    for name, body in DOCS.items():
        r = client.post("/documents/upload", files={"file": (name, body.encode(), "text/plain")})
        assert r.json()["chunks_indexed"] >= 1


def test_retrieval_top1_correct(client):
    _load(client)
    cases = [
        ("how many vacation days do employees get", "vacation_policy"),
        ("when are travel expenses due with receipts", "expense_policy"),
        ("how often do passwords need rotation", "security_policy"),
        ("is multi factor authentication required for vpn", "security_policy"),
    ]
    for q, expected in cases:
        res = client.get("/documents/search", params={"q": q}).json()["results"]
        assert res and expected in res[0]["filename"], q


def test_rag_answer_has_sources(client):
    _load(client)
    llm_service.set_provider(None)  # extractive fallback
    r = client.post("/chat/documents", json={"message": "how many vacation days do employees get"}).json()
    assert "20 days" in r["reply"]
    assert r["sources"] and "vacation" in r["sources"][0]["filename"]


def test_rag_nothing_found(client):
    _load(client)
    r = client.post("/chat/documents", json={"message": "zzzqqq xylophone"}).json()
    assert r["sources"] == []


def test_delete_removes_chunks(client):
    _load(client)
    docs = client.get("/documents/").json()["documents"]
    target = next(d for d in docs if d["filename"].startswith("security_policy"))
    assert client.delete(f"/documents/{target['id']}").status_code == 200
    res = client.get("/documents/search", params={"q": "passwords rotated every 90 days"}).json()["results"]
    assert all(target["id"] != h["doc_id"] for h in res)
