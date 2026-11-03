from app.services import auth_service


def test_register_login_flow(anon_client):
    r = anon_client.post("/auth/register", json={"username": "carol", "password": "longenough1"})
    assert r.status_code == 201
    r = anon_client.post("/auth/login", json={"username": "carol", "password": "longenough1"})
    assert r.status_code == 200 and r.json()["token_type"] == "bearer"


def test_duplicate_username(anon_client):
    body = {"username": "dave", "password": "longenough1"}
    assert anon_client.post("/auth/register", json=body).status_code == 201
    assert anon_client.post("/auth/register", json=body).status_code == 409


def test_wrong_password(anon_client):
    anon_client.post("/auth/register", json={"username": "erin", "password": "longenough1"})
    r = anon_client.post("/auth/login", json={"username": "erin", "password": "wrongpass99"})
    assert r.status_code == 401


def test_short_password_rejected(anon_client):
    assert anon_client.post("/auth/register", json={"username": "frank", "password": "short"}).status_code == 422


def test_password_not_stored_plain():
    from app.core.db import get_conn
    auth_service.register("gina", "plainpass123")
    row = get_conn().execute("SELECT * FROM users WHERE username='gina'").fetchone()
    assert "plainpass123" not in (row["password_hash"] + row["salt"])


def test_routes_require_token(anon_client):
    for method, path in [("get", "/documents/"), ("get", "/documents/search?q=x"),
                         ("post", "/chat/"), ("post", "/agent/run")]:
        r = getattr(anon_client, method)(path)
        assert r.status_code == 401, path


def test_bad_token_rejected(anon_client):
    r = anon_client.get("/documents/", headers={"Authorization": "Bearer not.a.token"})
    assert r.status_code == 401


def test_expired_token_rejected(anon_client, monkeypatch):
    from app.core import config
    monkeypatch.setattr(config, "JWT_EXPIRE_MINUTES", -5)
    token = auth_service.create_token("alice")
    r = anon_client.get("/documents/", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


def test_users_cant_see_each_others_documents(client, other_client):
    doc = client.post("/documents/upload", files={"file": ("private.txt", b"alice secret salary budget plan", "text/plain")}).json()["document"]
    assert other_client.get(f"/documents/{doc['id']}").status_code == 404
    assert other_client.delete(f"/documents/{doc['id']}").status_code == 404
    assert all(d["id"] != doc["id"] for d in other_client.get("/documents/").json()["documents"])
    assert other_client.get("/documents/search", params={"q": "secret salary budget"}).json()["results"] == []
    assert client.get(f"/documents/{doc['id']}").status_code == 200


def test_agent_approval_is_per_user(client, other_client):
    client.post("/documents/upload", files={"file": ("mine.txt", b"alice file to be deleted later maybe", "text/plain")})
    r = client.post("/agent/run", json={"message": "delete mine.txt"}).json()
    assert r["status"] == "awaiting_approval"
    assert other_client.post("/agent/approve", json={"thread_id": r["thread_id"], "approved": True}).status_code == 409
    # other user also can't target alice's file by name
    r2 = other_client.post("/agent/run", json={"message": "delete mine.txt"}).json()
    assert r2["status"] == "done"


def test_unhandled_error_is_hidden(client, monkeypatch):
    from app.services import vector_store
    monkeypatch.setattr(vector_store, "search", lambda *a, **k: 1 / 0)
    r = client.get("/documents/search", params={"q": "boom"})
    assert r.status_code == 500 and r.json() == {"detail": "Internal server error"}


def test_request_id_header(client):
    assert len(client.get("/health").headers["X-Request-ID"]) == 8


def test_openapi_docs_available(anon_client):
    spec = anon_client.get("/openapi.json").json()
    assert spec["info"]["version"] == "1.0.0"
    assert "/chat/documents" in spec["paths"]
