import os
import tempfile

# has to happen before the app gets imported so it doesn't touch real data
_tmp = tempfile.mkdtemp()
os.environ["DATA_DIR"] = _tmp
os.environ["DB_PATH"] = os.path.join(_tmp, "test.db")
os.environ.pop("ANTHROPIC_API_KEY", None)

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    return TestClient(app)
