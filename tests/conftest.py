import os
import tempfile

# has to happen before the app gets imported so it doesn't touch real data
_tmp = tempfile.mkdtemp()
os.environ["DATA_DIR"] = _tmp
os.environ["DB_PATH"] = os.path.join(_tmp, "test.db")
os.environ["JWT_SECRET"] = "test-secret-that-is-long-enough-for-hs256"
os.environ.pop("ANTHROPIC_API_KEY", None)

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import auth_service

PASSWORD = "password123"


def make_client(username: str) -> TestClient:
    c = TestClient(app, raise_server_exceptions=False)
    try:
        auth_service.register(username, PASSWORD)
    except auth_service.AuthError:
        pass
    token = auth_service.authenticate(username, PASSWORD)
    c.headers.update({"Authorization": f"Bearer {token}"})
    return c


@pytest.fixture()
def client():
    return make_client("alice")


@pytest.fixture()
def other_client():
    return make_client("bob")


@pytest.fixture()
def anon_client():
    return TestClient(app, raise_server_exceptions=False)
