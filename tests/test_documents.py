SAMPLE = b"Meeting notes: discussed Q3 roadmap. Attendees: John, Sarah. Action items assigned to team leads."


def upload(client, name="notes.txt", data=SAMPLE):
    return client.post("/documents/upload", files={"file": (name, data, "text/plain")})


def test_health(client):
    assert client.get("/health").json() == {"status": "healthy"}


def test_upload_and_categorize(client):
    r = upload(client)
    assert r.status_code == 200
    doc = r.json()["document"]
    assert doc["category"] == "meeting_notes"
    assert len(doc["keywords"]) > 0


def test_empty_upload_rejected(client):
    r = upload(client, data=b"")
    assert r.status_code == 400


def test_get_and_list(client):
    doc = upload(client, "a.txt").json()["document"]
    assert client.get(f"/documents/{doc['id']}").status_code == 200
    assert client.get("/documents/").json()["count"] >= 1


def test_missing_doc_404(client):
    assert client.get("/documents/nope").status_code == 404


def test_keywords_survive_reload(client):
    from app.services.document_service import document_service
    doc = upload(client, "persist.txt").json()["document"]
    fresh = document_service.get_document(doc["id"])
    assert fresh.keywords == doc["keywords"]
    assert fresh.extracted_text


def test_stats(client):
    upload(client, "s.txt")
    stats = client.get("/documents/stats/summary").json()
    assert stats["total_documents"] >= 1
